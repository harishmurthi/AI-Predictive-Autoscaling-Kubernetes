import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np

from data_collector import MetricSnapshot
from model_loader import LSTMModelHandler

LOGGER = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    predicted_cpu_percent: float
    projected_cpu_percent: float
    latest_cpu_percent: float
    latest_memory_percent: float
    latest_request_rate_rps: float
    previous_request_rate_rps: float
    request_rate_delta_rps: float
    cpu_trend_percent: float
    warmup: bool


class CPUPredictor:
    def __init__(
        self,
        model_handler: LSTMModelHandler,
        cpu_trend_weight: float = 0.60,
        request_rate_boost_percent: float = 15.0,
    ) -> None:
        self.model_handler = model_handler
        if self.model_handler.input_timesteps is None:
            raise RuntimeError("Model handler must be loaded before predictor init")
        self.history: Deque[np.ndarray] = deque(maxlen=max(256, self.model_handler.input_timesteps * 4))
        self.previous_snapshot: Optional[MetricSnapshot] = None
        self.cpu_trend_weight = max(0.0, cpu_trend_weight)
        self.request_rate_boost_percent = max(0.0, request_rate_boost_percent)

    def _compute_projected_cpu(
        self,
        predicted_cpu: float,
        cpu_trend_percent: float,
        request_rate_delta_rps: float,
    ) -> float:
        request_max = max(1.0, self.model_handler.norm_cfg.request_max)
        request_rate_boost = max(0.0, request_rate_delta_rps) / request_max
        request_rate_boost *= self.request_rate_boost_percent
        cpu_trend_boost = max(0.0, cpu_trend_percent) * self.cpu_trend_weight
        projected_cpu = predicted_cpu + cpu_trend_boost + request_rate_boost
        return float(np.clip(projected_cpu, 0.0, 100.0))

    def update_and_predict(self, snapshot: MetricSnapshot) -> PredictionResult:
        previous_cpu = self.previous_snapshot.cpu_usage_percent if self.previous_snapshot is not None else snapshot.cpu_usage_percent
        previous_request_rate = (
            self.previous_snapshot.request_rate_rps if self.previous_snapshot is not None else snapshot.request_rate_rps
        )
        cpu_trend_percent = snapshot.cpu_usage_percent - previous_cpu
        request_rate_delta_rps = snapshot.request_rate_rps - previous_request_rate

        normalized = self.model_handler.normalize(
            cpu=snapshot.cpu_usage_percent,
            memory=snapshot.memory_usage_percent,
            request_rate=snapshot.request_rate_rps,
            current_replicas=snapshot.current_replicas,
        )
        self.history.append(normalized)

        timesteps = self.model_handler.input_timesteps
        assert timesteps is not None

        if len(self.history) < timesteps:
            projected_cpu = self._compute_projected_cpu(
                predicted_cpu=snapshot.cpu_usage_percent,
                cpu_trend_percent=cpu_trend_percent,
                request_rate_delta_rps=request_rate_delta_rps,
            )
            LOGGER.info(
                "Warmup mode: %d/%d points collected. Using current CPU as prediction and %.2f%% as projected CPU.",
                len(self.history),
                timesteps,
                projected_cpu,
            )
            self.previous_snapshot = snapshot
            return PredictionResult(
                predicted_cpu_percent=snapshot.cpu_usage_percent,
                projected_cpu_percent=projected_cpu,
                latest_cpu_percent=snapshot.cpu_usage_percent,
                latest_memory_percent=snapshot.memory_usage_percent,
                latest_request_rate_rps=snapshot.request_rate_rps,
                previous_request_rate_rps=previous_request_rate,
                request_rate_delta_rps=request_rate_delta_rps,
                cpu_trend_percent=cpu_trend_percent,
                warmup=True,
            )

        input_tensor = self.model_handler.to_window_tensor(self.history)
        pred = self.model_handler.model.predict(input_tensor, verbose=0)

        if isinstance(pred, list):
            pred = pred[0]

        predicted_norm_cpu = float(np.ravel(pred)[0])
        predicted_cpu = self.model_handler.denormalize_cpu(predicted_norm_cpu)
        projected_cpu = self._compute_projected_cpu(
            predicted_cpu=predicted_cpu,
            cpu_trend_percent=cpu_trend_percent,
            request_rate_delta_rps=request_rate_delta_rps,
        )

        LOGGER.info(
            "Predicted next CPU: %.2f%% projected CPU: %.2f%% (actual now: %.2f%%, cpu_trend=%.2f, req_delta=%.3f)",
            predicted_cpu,
            projected_cpu,
            snapshot.cpu_usage_percent,
            cpu_trend_percent,
            request_rate_delta_rps,
        )
        self.previous_snapshot = snapshot

        return PredictionResult(
            predicted_cpu_percent=predicted_cpu,
            projected_cpu_percent=projected_cpu,
            latest_cpu_percent=snapshot.cpu_usage_percent,
            latest_memory_percent=snapshot.memory_usage_percent,
            latest_request_rate_rps=snapshot.request_rate_rps,
            previous_request_rate_rps=previous_request_rate,
            request_rate_delta_rps=request_rate_delta_rps,
            cpu_trend_percent=cpu_trend_percent,
            warmup=False,
        )
