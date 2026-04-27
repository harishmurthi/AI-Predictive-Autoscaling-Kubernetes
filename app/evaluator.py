import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class EvalPoint:
    timestamp: float
    actual_cpu: float
    predicted_cpu: float
    ai_replicas: int
    hpa_replicas: Optional[int]
    request_rate: float
    ai_latency_ms: Optional[float]
    hpa_latency_ms: Optional[float]


class Evaluator:
    def __init__(self, output_dir: str = "artifacts") -> None:
        self.points: List[EvalPoint] = []
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def add_point(self, point: EvalPoint) -> None:
        self.points.append(point)

    def _safe_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        non_zero = np.where(np.abs(y_true) > 1e-6, y_true, np.nan)
        mape = np.nanmean(np.abs((y_true - y_pred) / non_zero)) * 100.0
        if np.isnan(mape):
            return 0.0
        return float(mape)

    def compute_metrics(self) -> dict:
        if not self.points:
            return {
                "rmse": 0.0,
                "mae": 0.0,
                "mape": 0.0,
                "resource_savings_percent": 0.0,
                "latency_reduction_percent": 0.0,
            }

        actual = np.array([p.actual_cpu for p in self.points], dtype=np.float32)
        predicted = np.array([p.predicted_cpu for p in self.points], dtype=np.float32)
        ai_replicas = np.array([p.ai_replicas for p in self.points], dtype=np.float32)
        hpa_replicas = np.array([
            p.hpa_replicas if p.hpa_replicas is not None else p.ai_replicas for p in self.points
        ], dtype=np.float32)

        rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
        mae = float(np.mean(np.abs(actual - predicted)))
        mape = self._safe_mape(actual, predicted)

        hpa_sum = float(np.sum(hpa_replicas)) + 1e-9
        ai_sum = float(np.sum(ai_replicas))
        resource_savings = max(0.0, ((hpa_sum - ai_sum) / hpa_sum) * 100.0)

        ai_latency = np.array([
            p.ai_latency_ms for p in self.points if p.ai_latency_ms is not None
        ], dtype=np.float32)
        hpa_latency = np.array([
            p.hpa_latency_ms for p in self.points if p.hpa_latency_ms is not None
        ], dtype=np.float32)

        if ai_latency.size > 0 and hpa_latency.size > 0:
            ai_lat_mean = float(np.mean(ai_latency))
            hpa_lat_mean = float(np.mean(hpa_latency)) + 1e-9
            latency_reduction = max(0.0, ((hpa_lat_mean - ai_lat_mean) / hpa_lat_mean) * 100.0)
        else:
            latency_reduction = 0.0

        return {
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "resource_savings_percent": resource_savings,
            "latency_reduction_percent": latency_reduction,
        }

    def save_csv(self, filename: str = "evaluation_log.csv") -> Path:
        out = self.output_dir / filename
        with out.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(
                [
                    "timestamp",
                    "actual_cpu",
                    "predicted_cpu",
                    "ai_replicas",
                    "hpa_replicas",
                    "request_rate",
                    "ai_latency_ms",
                    "hpa_latency_ms",
                ]
            )
            for p in self.points:
                writer.writerow(
                    [
                        p.timestamp,
                        p.actual_cpu,
                        p.predicted_cpu,
                        p.ai_replicas,
                        p.hpa_replicas,
                        p.request_rate,
                        p.ai_latency_ms,
                        p.hpa_latency_ms,
                    ]
                )
        return out

    def _plot_xy(
        self,
        x: np.ndarray,
        y1: np.ndarray,
        y2: np.ndarray,
        y1_label: str,
        y2_label: str,
        title: str,
        out_name: str,
        y_label: str,
    ) -> None:
        plt.figure(figsize=(12, 5))
        plt.plot(x, y1, label=y1_label, linewidth=2)
        plt.plot(x, y2, label=y2_label, linewidth=2)
        plt.title(title)
        plt.xlabel("Time index")
        plt.ylabel(y_label)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / out_name, dpi=150)
        plt.close()

    def generate_plots(self) -> None:
        if not self.points:
            LOGGER.warning("No evaluation points available; skipping plots.")
            return

        x = np.arange(len(self.points))
        actual = np.array([p.actual_cpu for p in self.points], dtype=np.float32)
        pred = np.array([p.predicted_cpu for p in self.points], dtype=np.float32)

        self._plot_xy(
            x,
            pred,
            actual,
            "Predicted CPU (%)",
            "Actual CPU (%)",
            "Prediction vs Actual CPU",
            "prediction_vs_actual.png",
            "CPU (%)",
        )

        self._plot_xy(
            x,
            actual,
            pred,
            "Actual CPU (%)",
            "Predicted CPU (%)",
            "CPU Usage Comparison",
            "cpu_usage_comparison.png",
            "CPU (%)",
        )

        ai_replicas = np.array([p.ai_replicas for p in self.points], dtype=np.float32)
        hpa_replicas = np.array([
            p.hpa_replicas if p.hpa_replicas is not None else p.ai_replicas for p in self.points
        ], dtype=np.float32)

        self._plot_xy(
            x,
            ai_replicas,
            hpa_replicas,
            "AI Replicas",
            "HPA Replicas",
            "Replica Count Comparison",
            "replica_count_comparison.png",
            "Replicas",
        )

    def finalize(self) -> dict:
        metrics = self.compute_metrics()
        csv_path = self.save_csv()
        self.generate_plots()
        metrics_path = self.output_dir / "metrics.txt"
        with metrics_path.open("w", encoding="utf-8") as fp:
            for key, value in metrics.items():
                fp.write(f"{key}: {value:.4f}\n")

        LOGGER.info("Evaluation CSV saved to %s", csv_path)
        LOGGER.info("Evaluation metrics saved to %s", metrics_path)
        LOGGER.info("Metrics: %s", metrics)
        return metrics
