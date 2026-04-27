import logging
import os
import signal
import sys
import time
import csv
from datetime import datetime
from typing import Optional

from data_collector import PrometheusCollector
from evaluator import EvalPoint, Evaluator
from model_loader import LSTMModelHandler, NormalizationConfig
from predictor import CPUPredictor
from scaler import K8sScaler

RUNNING = True


def _setup_logging() -> None:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
    )


def _signal_handler(signum, frame) -> None:
    del signum, frame
    global RUNNING
    RUNNING = False


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logging.getLogger(__name__).warning("Invalid float env %s=%s, using %s", name, value, default)
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logging.getLogger(__name__).warning("Invalid int env %s=%s, using %s", name, value, default)
        return default


def _optional_float_env(name: str) -> Optional[float]:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        logging.getLogger(__name__).warning("Invalid optional float env %s=%s, ignoring", name, value)
        return None


def run() -> None:
    _setup_logging()
    logger = logging.getLogger(__name__)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    prometheus_url = os.getenv("PROMETHEUS_URL", "http://prometheus-server.monitoring.svc.cluster.local")
    namespace = os.getenv("NAMESPACE", "autoscale-demo")
    deployment_name = os.getenv("DEPLOYMENT_NAME", "cpu-demo")
    model_path = os.getenv("MODEL_PATH", "models/offline_lstm_model.h5")

    interval_seconds = _env_int("LOOP_INTERVAL_SECONDS", 10)
    cooldown_seconds = _env_int("COOLDOWN_SECONDS", 15)
    min_replicas = _env_int("MIN_REPLICAS", 1)
    max_replicas = _env_int("MAX_REPLICAS", 20)
    scale_step = _env_int("SCALE_STEP", 1)
    burst_scale_step = _env_int("BURST_SCALE_STEP", 2)
    max_scale_up_step = _env_int("MAX_SCALE_UP_STEP", 4)

    up_threshold = _env_float("UP_THRESHOLD", 70.0)
    down_threshold = _env_float("DOWN_THRESHOLD", 30.0)
    scale_up_margin = _env_float("SCALE_UP_MARGIN", 10.0)
    burst_rps_delta_threshold = _env_float("BURST_RPS_DELTA_THRESHOLD", 5.0)
    absolute_burst_rps_threshold = _env_float("ABSOLUTE_BURST_RPS_THRESHOLD", 20.0)
    cooldown_bypass_margin = _env_float("COOLDOWN_BYPASS_MARGIN", 8.0)
    scale_down_streak_required = _env_int("SCALE_DOWN_STREAK_REQUIRED", 2)
    cpu_trend_weight = _env_float("CPU_TREND_WEIGHT", 0.60)
    request_rate_boost_percent = _env_float("REQUEST_RATE_BOOST_PERCENT", 15.0)
    cpu_rate_window = os.getenv("CPU_RATE_WINDOW", "30s")
    request_rate_window = os.getenv("REQUEST_RATE_WINDOW", "30s")
    request_rate_query = os.getenv("REQUEST_RATE_QUERY", "").strip()

    norm_cfg = NormalizationConfig(
        cpu_min=_env_float("CPU_MIN", 0.0),
        cpu_max=_env_float("CPU_MAX", 100.0),
        memory_min=_env_float("MEMORY_MIN", 0.0),
        memory_max=_env_float("MEMORY_MAX", 100.0),
        request_min=_env_float("REQUEST_MIN", 0.0),
        request_max=_env_float("REQUEST_MAX", 2000.0),
    )

    ai_latency_ms = _optional_float_env("AI_LATENCY_MS")
    hpa_latency_ms = _optional_float_env("HPA_LATENCY_MS")

    collector = PrometheusCollector(
        prometheus_url=prometheus_url,
        namespace=namespace,
        deployment_name=deployment_name,
        cpu_rate_window=cpu_rate_window,
        request_rate_window=request_rate_window,
        request_rate_queries=[request_rate_query] if request_rate_query else [],
    )

    model_handler = LSTMModelHandler(model_path=model_path, norm_cfg=norm_cfg)
    model_handler.load()

    predictor = CPUPredictor(
        model_handler=model_handler,
        cpu_trend_weight=cpu_trend_weight,
        request_rate_boost_percent=request_rate_boost_percent,
    )

    scaler = K8sScaler(
        namespace=namespace,
        deployment_name=deployment_name,
        min_replicas=min_replicas,
        max_replicas=max_replicas,
        up_threshold=up_threshold,
        down_threshold=down_threshold,
        scale_step=scale_step,
        cooldown_seconds=cooldown_seconds,
        scale_up_margin=scale_up_margin,
        burst_rps_delta_threshold=burst_rps_delta_threshold,
        absolute_burst_rps_threshold=absolute_burst_rps_threshold,
        burst_scale_step=burst_scale_step,
        max_scale_up_step=max_scale_up_step,
        cooldown_bypass_margin=cooldown_bypass_margin,
        scale_down_streak_required=scale_down_streak_required,
    )

    evaluator = Evaluator(output_dir=os.getenv("EVAL_OUTPUT_DIR", "artifacts"))
    metrics_log_path = "/app/metrics_log.csv"

    if not os.path.exists(metrics_log_path):
        try:
            with open(metrics_log_path, "w", newline="", encoding="utf-8") as fp:
                writer = csv.writer(fp)
                writer.writerow(
                    [
                        "timestamp",
                        "cpu",
                        "predicted_cpu",
                        "projected_cpu",
                        "replicas",
                        "request_rate",
                        "req_delta",
                        "reason",
                    ]
                )
        except Exception:
            logger.exception("Failed to initialize metrics CSV at %s", metrics_log_path)

    logger.info(
        "Starting predictive autoscaler for %s/%s with interval=%ss cooldown=%ss cpu_window=%s req_window=%s",
        namespace,
        deployment_name,
        interval_seconds,
        cooldown_seconds,
        cpu_rate_window,
        request_rate_window,
    )

    first_burst_logged = False

    while RUNNING:
        loop_start = time.time()
        try:
            snapshot = collector.collect()
            prediction = predictor.update_and_predict(snapshot)

            current_replicas = scaler.get_current_replicas()
            decision = scaler.decide_with_signals(
                predicted_cpu_percent=prediction.predicted_cpu_percent,
                projected_cpu_percent=prediction.projected_cpu_percent,
                current_replicas=current_replicas,
                actual_cpu_percent=prediction.latest_cpu_percent,
                request_rate_rps=prediction.latest_request_rate_rps,
                request_rate_delta_rps=prediction.request_rate_delta_rps,
            )
            scaler.apply(decision)

            hpa_replicas = scaler.get_hpa_desired_replicas()

            evaluator.add_point(
                EvalPoint(
                    timestamp=snapshot.timestamp,
                    actual_cpu=prediction.latest_cpu_percent,
                    predicted_cpu=prediction.predicted_cpu_percent,
                    ai_replicas=decision.desired_replicas,
                    hpa_replicas=hpa_replicas,
                    request_rate=prediction.latest_request_rate_rps,
                    ai_latency_ms=ai_latency_ms,
                    hpa_latency_ms=hpa_latency_ms,
                )
            )

            logger.info(
                "Loop summary actual_cpu=%.2f predicted_cpu=%.2f projected_cpu=%.2f action=%s replicas=%d hpa=%s",
                prediction.latest_cpu_percent,
                prediction.predicted_cpu_percent,
                prediction.projected_cpu_percent,
                decision.action,
                decision.desired_replicas,
                hpa_replicas,
            )
            logger.info(
                "Decision details: actual_cpu=%.2f predicted_cpu=%.2f projected_cpu=%.2f "
                "req_rate=%.3f req_delta=%.3f scale_up_threshold=%.2f proactive_margin=%.2f "
                "absolute_burst_rps_threshold=%.2f cooldown_bypass_margin=%.2f "
                "scale_down_threshold=%.2f current_replicas=%d decision_reason=%s",
                prediction.latest_cpu_percent,
                prediction.predicted_cpu_percent,
                prediction.projected_cpu_percent,
                prediction.latest_request_rate_rps,
                prediction.request_rate_delta_rps,
                up_threshold,
                scale_up_margin,
                absolute_burst_rps_threshold,
                cooldown_bypass_margin,
                down_threshold,
                current_replicas,
                decision.reason,
            )
            logger.info(
                "RUNTIME_METRIC timestamp=%s cpu=%.4f predicted_cpu=%.4f projected_cpu=%.4f "
                "replicas=%d request_rate=%.4f req_delta=%.4f reason=%s",
                datetime.utcnow().isoformat(),
                prediction.latest_cpu_percent,
                prediction.predicted_cpu_percent,
                prediction.projected_cpu_percent,
                decision.desired_replicas,
                prediction.latest_request_rate_rps,
                prediction.request_rate_delta_rps,
                decision.reason,
            )

            burst_detected = (
                prediction.request_rate_delta_rps >= burst_rps_delta_threshold
                or prediction.latest_request_rate_rps >= absolute_burst_rps_threshold
            )
            if burst_detected and not first_burst_logged:
                first_burst_logged = True
                logger.warning(
                    "First burst detected at %s req_rate=%.3f req_delta=%.3f projected_cpu=%.2f",
                    datetime.utcnow().isoformat(),
                    prediction.latest_request_rate_rps,
                    prediction.request_rate_delta_rps,
                    prediction.projected_cpu_percent,
                )

            try:
                with open(metrics_log_path, "a", newline="", encoding="utf-8") as fp:
                    writer = csv.writer(fp)
                    writer.writerow(
                        [
                            datetime.utcnow().isoformat(),
                            f"{prediction.latest_cpu_percent:.4f}",
                            f"{prediction.predicted_cpu_percent:.4f}",
                            f"{prediction.projected_cpu_percent:.4f}",
                            decision.desired_replicas,
                            f"{prediction.latest_request_rate_rps:.4f}",
                            f"{prediction.request_rate_delta_rps:.4f}",
                            decision.reason,
                        ]
                    )
            except Exception:
                logger.exception("Failed to append metrics CSV row")
        except Exception:
            logger.exception("Main loop iteration failed")

        elapsed = time.time() - loop_start
        sleep_for = max(0.0, interval_seconds - elapsed)
        time.sleep(sleep_for)

    logger.info("Shutdown signal received. Finalizing evaluation artifacts...")
    evaluator.finalize()


if __name__ == "__main__":
    run()
