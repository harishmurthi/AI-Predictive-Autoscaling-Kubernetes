import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List

import numpy as np
from tensorflow.keras.models import load_model

LOGGER = logging.getLogger(__name__)


@dataclass
class NormalizationConfig:
    cpu_min: float = 0.0
    cpu_max: float = 100.0
    memory_min: float = 0.0
    memory_max: float = 100.0
    request_min: float = 0.0
    request_max: float = 2000.0
    replica_min: float = 1.0
    replica_max: float = 20.0


class LSTMModelHandler:
    def __init__(self, model_path: str, norm_cfg: NormalizationConfig) -> None:
        self.model_path = Path(model_path)
        self.norm_cfg = norm_cfg
        self.model = None
        self.input_timesteps = None
        self.input_features = None

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        # Inference-only service: avoid deserializing training-time compile objects
        # from legacy H5 models (e.g., keras.metrics.mse incompatibilities).
        self.model = load_model(self.model_path, compile=False)
        shape = self.model.input_shape
        if len(shape) != 3:
            raise ValueError(f"Unexpected model input shape: {shape}")

        self.input_timesteps = int(shape[1]) if shape[1] is not None else 10
        self.input_features = int(shape[2]) if shape[2] is not None else 3
        LOGGER.info(
            "Loaded model from %s with input shape=%s (timesteps=%s, features=%s)",
            self.model_path,
            shape,
            self.input_timesteps,
            self.input_features,
        )

    def normalize(self, cpu: float, memory: float, request_rate: float, current_replicas: float) -> np.ndarray:
        cfg = self.norm_cfg
        norm_cpu = (cpu - cfg.cpu_min) / (cfg.cpu_max - cfg.cpu_min + 1e-9)
        norm_mem = (memory - cfg.memory_min) / (cfg.memory_max - cfg.memory_min + 1e-9)
        norm_req = (request_rate - cfg.request_min) / (cfg.request_max - cfg.request_min + 1e-9)
        norm_rep = (current_replicas - cfg.replica_min) / (cfg.replica_max - cfg.replica_min + 1e-9)

        values = np.array([norm_cpu, norm_mem, norm_req, norm_rep], dtype=np.float32)
        values = np.clip(values, 0.0, 1.0)
        return values

    def denormalize_cpu(self, normalized_cpu: float) -> float:
        cfg = self.norm_cfg
        cpu = normalized_cpu * (cfg.cpu_max - cfg.cpu_min) + cfg.cpu_min
        return float(np.clip(cpu, 0.0, 100.0))

    def to_window_tensor(self, sequence: Deque[np.ndarray]) -> np.ndarray:
        if self.input_timesteps is None or self.input_features is None:
            raise RuntimeError("Model must be loaded before preprocessing input")

        if len(sequence) < self.input_timesteps:
            raise ValueError(
                f"Need at least {self.input_timesteps} observations, got {len(sequence)}"
            )

        window: List[np.ndarray] = list(sequence)[-self.input_timesteps :]
        matrix = np.stack(window, axis=0)

        if matrix.shape[1] != self.input_features:
            raise ValueError(
                f"Feature mismatch. model expects {self.input_features}, got {matrix.shape[1]}"
            )

        return np.expand_dims(matrix, axis=0).astype(np.float32)
