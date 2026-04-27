import logging
import time
from dataclasses import dataclass
from typing import Optional

from kubernetes import client, config
from kubernetes.client import ApiException

LOGGER = logging.getLogger(__name__)


@dataclass
class ScalingDecision:
    action: str
    desired_replicas: int
    reason: str
    cooldown_active: bool


class K8sScaler:
    def __init__(
        self,
        namespace: str,
        deployment_name: str,
        min_replicas: int = 1,
        max_replicas: int = 20,
        up_threshold: float = 70.0,
        down_threshold: float = 30.0,
        scale_step: int = 1,
        cooldown_seconds: int = 60,
        scale_up_margin: float = 10.0,
        burst_rps_delta_threshold: float = 5.0,
        absolute_burst_rps_threshold: float = 20.0,
        burst_scale_step: int = 2,
        max_scale_up_step: int = 4,
        cooldown_bypass_margin: float = 8.0,
        scale_down_streak_required: int = 2,
    ) -> None:
        self.namespace = namespace
        self.deployment_name = deployment_name
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold
        self.scale_step = scale_step
        self.cooldown_seconds = cooldown_seconds
        self.scale_up_margin = max(0.0, scale_up_margin)
        self.burst_rps_delta_threshold = max(0.0, burst_rps_delta_threshold)
        self.absolute_burst_rps_threshold = max(0.0, absolute_burst_rps_threshold)
        self.burst_scale_step = max(1, burst_scale_step)
        self.max_scale_up_step = max(self.scale_step, max_scale_up_step)
        self.cooldown_bypass_margin = max(0.0, cooldown_bypass_margin)
        self.scale_down_streak_required = max(1, scale_down_streak_required)
        self.last_scale_ts = 0.0
        self.low_utilization_streak = 0

        config.load_incluster_config()
        LOGGER.info("Loaded in-cluster Kubernetes config")
        self.apps_api = client.AppsV1Api()
        self.autoscaling_api = client.AutoscalingV2Api()

    def get_current_replicas(self) -> int:
        deployment = self.apps_api.read_namespaced_deployment(
            name=self.deployment_name,
            namespace=self.namespace,
        )
        return int(deployment.spec.replicas or 1)

    def get_hpa_desired_replicas(self) -> Optional[int]:
        try:
            hpa = self.autoscaling_api.read_namespaced_horizontal_pod_autoscaler(
                name=self.deployment_name,
                namespace=self.namespace,
            )
            if hpa.status and hpa.status.desired_replicas is not None:
                return int(hpa.status.desired_replicas)
            if hpa.spec and hpa.spec.min_replicas is not None:
                return int(hpa.spec.min_replicas)
            return None
        except ApiException as exc:
            if exc.status == 404:
                return None
            LOGGER.warning("Failed to read HPA: %s", exc)
            return None

    def decide(self, predicted_cpu_percent: float, current_replicas: int) -> ScalingDecision:
        return self.decide_with_signals(
            predicted_cpu_percent=predicted_cpu_percent,
            projected_cpu_percent=predicted_cpu_percent,
            current_replicas=current_replicas,
            actual_cpu_percent=None,
            request_rate_rps=None,
            request_rate_delta_rps=0.0,
        )

    def decide_with_actual(
        self,
        predicted_cpu_percent: float,
        current_replicas: int,
        actual_cpu_percent: Optional[float],
    ) -> ScalingDecision:
        return self.decide_with_signals(
            predicted_cpu_percent=predicted_cpu_percent,
            projected_cpu_percent=predicted_cpu_percent,
            current_replicas=current_replicas,
            actual_cpu_percent=actual_cpu_percent,
            request_rate_rps=None,
            request_rate_delta_rps=0.0,
        )

    def decide_with_signals(
        self,
        predicted_cpu_percent: float,
        projected_cpu_percent: float,
        current_replicas: int,
        actual_cpu_percent: Optional[float],
        request_rate_rps: Optional[float],
        request_rate_delta_rps: float,
    ) -> ScalingDecision:
        now = time.time()
        cooldown_active = (now - self.last_scale_ts) < self.cooldown_seconds
        proactive_up_threshold = max(self.down_threshold + 1.0, self.up_threshold - self.scale_up_margin)
        absolute_burst_detected = (
            request_rate_rps is not None and request_rate_rps >= self.absolute_burst_rps_threshold
        )
        burst_detected = request_rate_delta_rps >= self.burst_rps_delta_threshold or absolute_burst_detected
        cooldown_bypass = projected_cpu_percent >= (proactive_up_threshold + self.cooldown_bypass_margin)

        effective_cpu = projected_cpu_percent
        if actual_cpu_percent is not None and actual_cpu_percent >= proactive_up_threshold:
            effective_cpu = max(effective_cpu, actual_cpu_percent)

        can_scale_down = (
            predicted_cpu_percent < self.down_threshold
            and projected_cpu_percent < self.down_threshold
            and (actual_cpu_percent is None or actual_cpu_percent < self.down_threshold)
            and request_rate_delta_rps <= 0.0
        )

        if can_scale_down:
            self.low_utilization_streak += 1
        else:
            self.low_utilization_streak = 0

        if burst_detected:
            dynamic_step = self._compute_dynamic_scale_up_step(
                effective_cpu=effective_cpu,
                proactive_up_threshold=proactive_up_threshold,
                request_rate_delta_rps=request_rate_delta_rps,
                request_rate_rps=request_rate_rps,
                base_step=self.burst_scale_step,
            )
            desired = min(self.max_replicas, current_replicas + dynamic_step)
            action = "scale_up" if desired > current_replicas else "hold"
            reason = (
                f"burst_detected req_delta={request_rate_delta_rps:.3f} rps "
                f"req_rate={0.0 if request_rate_rps is None else request_rate_rps:.3f} "
                f"projected_cpu={projected_cpu_percent:.2f}% step={dynamic_step} "
                f"absolute_trigger={absolute_burst_detected}"
            )
            return ScalingDecision(
                action=action,
                desired_replicas=desired,
                reason=reason,
                cooldown_active=False,
            )

        if cooldown_active and not cooldown_bypass and effective_cpu < proactive_up_threshold:
            return ScalingDecision(
                action="hold",
                desired_replicas=current_replicas,
                reason=(
                    f"cooldown (pred={predicted_cpu_percent:.2f}%, projected={projected_cpu_percent:.2f}%"
                    + (
                        f", actual={actual_cpu_percent:.2f}%"
                        if actual_cpu_percent is not None
                        else ""
                    )
                    + ")"
                ),
                cooldown_active=True,
            )

        if effective_cpu >= proactive_up_threshold:
            dynamic_step = self._compute_dynamic_scale_up_step(
                effective_cpu=effective_cpu,
                proactive_up_threshold=proactive_up_threshold,
                request_rate_delta_rps=request_rate_delta_rps,
                request_rate_rps=request_rate_rps,
                base_step=self.scale_step,
            )
            desired = min(self.max_replicas, current_replicas + dynamic_step)
            action = "scale_up" if desired > current_replicas else "hold"
            reason = (
                f"effective_cpu={effective_cpu:.2f}% (pred={predicted_cpu_percent:.2f}%, "
                f"projected={projected_cpu_percent:.2f}%) >= proactive threshold {proactive_up_threshold:.2f}% "
                f"step={dynamic_step}"
            )
            return ScalingDecision(action=action, desired_replicas=desired, reason=reason, cooldown_active=False)

        if can_scale_down and self.low_utilization_streak >= self.scale_down_streak_required:
            desired = max(self.min_replicas, current_replicas - self.scale_step)
            action = "scale_down" if desired < current_replicas else "hold"
            reason = (
                f"pred={predicted_cpu_percent:.2f}% projected={projected_cpu_percent:.2f}% "
                f"actual={0.0 if actual_cpu_percent is None else actual_cpu_percent:.2f}% "
                f"< {self.down_threshold:.2f}% and req_delta={request_rate_delta_rps:.3f} "
                f"streak={self.low_utilization_streak}"
            )
            return ScalingDecision(action=action, desired_replicas=desired, reason=reason, cooldown_active=False)

        if can_scale_down:
            return ScalingDecision(
                action="hold",
                desired_replicas=current_replicas,
                reason=(
                    f"waiting_for_scale_down_stability pred={predicted_cpu_percent:.2f}% "
                    f"projected={projected_cpu_percent:.2f}% actual={0.0 if actual_cpu_percent is None else actual_cpu_percent:.2f}% "
                    f"streak={self.low_utilization_streak}/{self.scale_down_streak_required}"
                ),
                cooldown_active=False,
            )

        return ScalingDecision(
            action="hold",
            desired_replicas=current_replicas,
            reason=(
                f"effective_cpu={effective_cpu:.2f}% (pred={predicted_cpu_percent:.2f}%, "
                f"projected={projected_cpu_percent:.2f}%) within hold band "
                f"[{self.down_threshold:.2f}%, {proactive_up_threshold:.2f}%]"
            ),
            cooldown_active=False,
        )

    def _compute_dynamic_scale_up_step(
        self,
        effective_cpu: float,
        proactive_up_threshold: float,
        request_rate_delta_rps: float,
        request_rate_rps: Optional[float],
        base_step: int,
    ) -> int:
        step = max(1, base_step)
        cpu_gap = max(0.0, effective_cpu - proactive_up_threshold)
        if cpu_gap >= 15.0:
            step += 2
        elif cpu_gap >= 8.0:
            step += 1

        if request_rate_delta_rps >= self.burst_rps_delta_threshold * 2:
            step += 1
        if request_rate_rps is not None and request_rate_rps >= self.absolute_burst_rps_threshold * 1.5:
            step += 1

        return min(self.max_scale_up_step, max(self.scale_step, step))

    def apply(self, decision: ScalingDecision) -> bool:
        if decision.action == "hold":
            LOGGER.info("No scaling action: %s", decision.reason)
            return False

        try:
            body = {"spec": {"replicas": decision.desired_replicas}}
            self.apps_api.patch_namespaced_deployment_scale(
                name=self.deployment_name,
                namespace=self.namespace,
                body=body,
            )
            self.last_scale_ts = time.time()
            LOGGER.warning(
                "Scaling action=%s deployment=%s/%s replicas=%d reason=%s",
                decision.action,
                self.namespace,
                self.deployment_name,
                decision.desired_replicas,
                decision.reason,
            )
            return True
        except ApiException:
            LOGGER.exception("Failed to scale deployment")
            return False
