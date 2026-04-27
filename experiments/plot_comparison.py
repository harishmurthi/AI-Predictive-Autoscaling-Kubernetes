#!/usr/bin/env python3
import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class MetricsSeries:
    timestamps: np.ndarray
    seconds: np.ndarray
    cpu: np.ndarray
    replicas: np.ndarray
    request_rate: np.ndarray
    predicted_cpu: Optional[np.ndarray] = None
    cpu_label: str = "CPU (%)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate predictive vs HPA comparison graphs for research reporting."
    )
    parser.add_argument("--predictive", required=True, help="Path to predictive_metrics.csv")
    parser.add_argument("--hpa", required=True, help="Path to hpa_metrics.csv")
    parser.add_argument(
        "--outdir",
        default=".",
        help="Directory where cpu_comparison.png, replica_comparison.png, and prediction_vs_actual.png are written.",
    )
    parser.add_argument(
        "--predictive-cpu-limit-millicores",
        type=float,
        default=None,
        help="Optional CPU limit in millicores used to convert predictive CPU millicores to percent when needed.",
    )
    parser.add_argument(
        "--hpa-cpu-limit-millicores",
        type=float,
        default=None,
        help="Optional CPU limit in millicores used to convert HPA CPU millicores to percent when needed.",
    )
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (13, 6),
            "figure.dpi": 180,
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "legend.fontsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.35,
            "lines.linewidth": 2.6,
        }
    )


def parse_timestamp(value: str) -> datetime:
    value = value.strip()
    if value.endswith("Z"):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return datetime.fromisoformat(value)


def get_float(row: dict, candidates: List[str], default: float = 0.0) -> float:
    for key in candidates:
        value = row.get(key)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return default


def has_any_field(fieldnames: List[str], candidates: List[str]) -> bool:
    return any(name in fieldnames for name in candidates)


def load_metrics(path: Path) -> MetricsSeries:
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if not rows:
        raise ValueError(f"No rows found in {path}")

    timestamps: List[datetime] = []
    cpu_values: List[float] = []
    replica_values: List[float] = []
    request_rate_values: List[float] = []
    predicted_values: List[float] = []

    uses_percent_cpu = has_any_field(fieldnames, ["cpu", "actual_cpu", "cpu_actual"])
    cpu_label = "CPU (%)" if uses_percent_cpu else "CPU Load (millicores)"

    has_predicted = has_any_field(fieldnames, ["predicted_cpu", "cpu_predicted"])

    for row in rows:
        timestamps.append(parse_timestamp(row["timestamp"]))
        cpu_values.append(
            get_float(
                row,
                ["cpu", "actual_cpu", "cpu_actual", "cpu_millicores"],
            )
        )
        replica_values.append(get_float(row, ["replicas", "ai_replicas", "hpa_replicas"], default=1.0))
        request_rate_values.append(get_float(row, ["request_rate", "request_rate_rps"], default=0.0))
        if has_predicted:
            predicted_values.append(get_float(row, ["predicted_cpu", "cpu_predicted"]))

    base_ts = timestamps[0]
    seconds = np.array([(ts - base_ts).total_seconds() for ts in timestamps], dtype=np.float64)

    return MetricsSeries(
        timestamps=np.array(timestamps, dtype=object),
        seconds=seconds,
        cpu=np.array(cpu_values, dtype=np.float64),
        replicas=np.array(replica_values, dtype=np.float64),
        request_rate=np.array(request_rate_values, dtype=np.float64),
        predicted_cpu=np.array(predicted_values, dtype=np.float64) if has_predicted else None,
        cpu_label=cpu_label,
    )


def maybe_convert_cpu_to_percent(series: MetricsSeries, cpu_limit_millicores: Optional[float]) -> MetricsSeries:
    if series.cpu_label == "CPU (%)":
        return series
    if cpu_limit_millicores is None or cpu_limit_millicores <= 0:
        return series

    converted_cpu = (series.cpu / cpu_limit_millicores) * 100.0
    return MetricsSeries(
        timestamps=series.timestamps,
        seconds=series.seconds,
        cpu=converted_cpu,
        replicas=series.replicas,
        request_rate=series.request_rate,
        predicted_cpu=series.predicted_cpu,
        cpu_label="CPU (%)",
    )


def to_minutes(seconds: np.ndarray) -> np.ndarray:
    return seconds / 60.0


def detect_first_scale_up(series: MetricsSeries) -> Optional[int]:
    if len(series.replicas) == 0:
        return None
    initial = series.replicas[0]
    for idx, value in enumerate(series.replicas):
        if value > initial:
            return idx
    return None


def detect_spike_start(series_list: List[MetricsSeries]) -> Optional[float]:
    combined_seconds: List[float] = []
    combined_cpu: List[float] = []
    for series in series_list:
        combined_seconds.extend(series.seconds.tolist())
        combined_cpu.extend(series.cpu.tolist())

    if len(combined_cpu) < 5:
        return None

    order = np.argsort(np.array(combined_seconds))
    ordered_seconds = np.array(combined_seconds, dtype=np.float64)[order]
    ordered_cpu = np.array(combined_cpu, dtype=np.float64)[order]

    baseline_window = max(3, len(ordered_cpu) // 10)
    baseline = float(np.median(ordered_cpu[:baseline_window]))
    peak = float(np.max(ordered_cpu))
    if peak <= baseline:
        return None

    threshold = baseline + max((peak - baseline) * 0.25, 5.0 if peak <= 100.0 else 250.0)
    for idx in range(baseline_window, len(ordered_cpu)):
        value = ordered_cpu[idx]
        if value >= threshold:
            return float(ordered_seconds[idx])
    return None


def detect_rise_start(values: np.ndarray, seconds: np.ndarray) -> Optional[float]:
    if len(values) < 5:
        return None
    baseline_window = max(3, len(values) // 10)
    baseline = float(np.median(values[:baseline_window]))
    peak = float(np.max(values))
    if peak <= baseline:
        return None
    threshold = baseline + max((peak - baseline) * 0.25, 5.0)
    for idx in range(baseline_window, len(values)):
        value = values[idx]
        if value >= threshold:
            return float(seconds[idx])
    return None


def add_event_line(ax, x_minutes: float, label: str, color: str, linestyle: str = "--") -> None:
    ax.axvline(x=x_minutes, color=color, linestyle=linestyle, alpha=0.8, linewidth=2.0, label=label)


def save_cpu_comparison(
    predictive: MetricsSeries,
    hpa: MetricsSeries,
    out_path: Path,
    spike_start_seconds: Optional[float],
) -> None:
    if predictive.cpu_label != hpa.cpu_label:
        raise ValueError(
            "CPU units do not match between predictive and HPA series. "
            "Provide CPU limits in millicores so both can be converted to percent."
        )

    fig, ax = plt.subplots()
    ax.plot(to_minutes(predictive.seconds), predictive.cpu, color="#d62728", label="Predictive CPU", linewidth=3.2)
    ax.plot(to_minutes(hpa.seconds), hpa.cpu, color="#1f77b4", label="HPA CPU", linewidth=3.2)

    if spike_start_seconds is not None:
        spike_start_minutes = spike_start_seconds / 60.0
        max_minutes = max(
            float(np.max(to_minutes(predictive.seconds))),
            float(np.max(to_minutes(hpa.seconds))),
        )
        ax.axvspan(spike_start_minutes, max_minutes, color="#ffcc80", alpha=0.20, label="Spike region")
        add_event_line(ax, spike_start_minutes, "CPU spike start", "#ff7f0e")

    ax.set_title("CPU Comparison: Predictive Autoscaler vs HPA")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel(predictive.cpu_label)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_replica_comparison(
    predictive: MetricsSeries,
    hpa: MetricsSeries,
    out_path: Path,
    predictive_first_idx: Optional[int],
    hpa_first_idx: Optional[int],
) -> None:
    fig, ax = plt.subplots()
    ax.step(
        to_minutes(predictive.seconds),
        predictive.replicas,
        where="post",
        color="#d62728",
        label="Predictive replicas",
        linewidth=3.2,
    )
    ax.step(
        to_minutes(hpa.seconds),
        hpa.replicas,
        where="post",
        color="#1f77b4",
        label="HPA replicas",
        linewidth=3.2,
    )

    predictive_first_minutes = None
    hpa_first_minutes = None

    if predictive_first_idx is not None:
        predictive_first_minutes = predictive.seconds[predictive_first_idx] / 60.0
        ax.scatter(
            [predictive_first_minutes],
            [predictive.replicas[predictive_first_idx]],
            color="#d62728",
            s=70,
            zorder=4,
            label="Predictive first scale-up",
        )
        add_event_line(ax, predictive_first_minutes, "Predictive first scale-up", "#d62728")

    if hpa_first_idx is not None:
        hpa_first_minutes = hpa.seconds[hpa_first_idx] / 60.0
        ax.scatter(
            [hpa_first_minutes],
            [hpa.replicas[hpa_first_idx]],
            color="#1f77b4",
            s=70,
            zorder=4,
            label="HPA first scale-up",
        )
        add_event_line(ax, hpa_first_minutes, "HPA first scale-up", "#1f77b4")

    if predictive_first_minutes is not None and hpa_first_minutes is not None:
        if predictive_first_minutes < hpa_first_minutes:
            note = f"Predictive scales earlier by {(hpa_first_minutes - predictive_first_minutes) * 60.0:.1f}s"
            note_color = "#2e7d32"
        elif predictive_first_minutes > hpa_first_minutes:
            note = f"HPA scales earlier by {(predictive_first_minutes - hpa_first_minutes) * 60.0:.1f}s"
            note_color = "#c62828"
        else:
            note = "Both controllers scale at the same observed time"
            note_color = "#6a1b9a"

        x_text = min(predictive_first_minutes, hpa_first_minutes)
        y_text = max(float(np.max(predictive.replicas)), float(np.max(hpa.replicas))) * 0.92
        ax.annotate(
            note,
            xy=(x_text, y_text),
            xytext=(x_text + 0.15, y_text + 0.4),
            arrowprops={"arrowstyle": "->", "color": note_color, "lw": 1.8},
            color=note_color,
            fontsize=12,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": note_color, "alpha": 0.92},
        )

        if predictive_first_minutes < hpa_first_minutes:
            ax.annotate(
                "Predictive scales earlier",
                xy=(predictive_first_minutes, predictive.replicas[predictive_first_idx]),
                xytext=(predictive_first_minutes + 0.2, predictive.replicas[predictive_first_idx] + 1.5),
                arrowprops={"arrowstyle": "->", "color": "#2e7d32", "lw": 1.8},
                color="#2e7d32",
                fontsize=12,
            )
        else:
            ax.annotate(
                "HPA reacts earlier",
                xy=(hpa_first_minutes, hpa.replicas[hpa_first_idx]),
                xytext=(hpa_first_minutes + 0.2, hpa.replicas[hpa_first_idx] + 1.5),
                arrowprops={"arrowstyle": "->", "color": "#c62828", "lw": 1.8},
                color="#c62828",
                fontsize=12,
            )

    ax.set_title("Replica Count Comparison")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Replicas")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_prediction_vs_actual(
    predictive: MetricsSeries,
    out_path: Path,
    spike_start_seconds: Optional[float],
) -> None:
    if predictive.predicted_cpu is None:
        raise ValueError("Predictive metrics file must include predicted_cpu for the prediction plot.")

    fig, ax = plt.subplots()
    ax.plot(to_minutes(predictive.seconds), predictive.cpu, color="#1f77b4", label="Actual CPU", linewidth=3.2)
    ax.plot(to_minutes(predictive.seconds), predictive.predicted_cpu, color="#d62728", label="Predicted CPU", linewidth=3.2)

    predicted_rise = detect_rise_start(predictive.predicted_cpu, predictive.seconds)
    actual_rise = detect_rise_start(predictive.cpu, predictive.seconds)

    if spike_start_seconds is not None:
        add_event_line(ax, spike_start_seconds / 60.0, "Spike start", "#ff7f0e")

    if predicted_rise is not None:
        add_event_line(ax, predicted_rise / 60.0, "Predicted rise", "#d62728", linestyle=":")
    if actual_rise is not None:
        add_event_line(ax, actual_rise / 60.0, "Actual rise", "#1f77b4", linestyle=":")

    if predicted_rise is not None and actual_rise is not None:
        if predicted_rise < actual_rise:
            note = f"Prediction leads actual CPU rise by {(actual_rise - predicted_rise):.1f}s"
            note_color = "#2e7d32"
        elif predicted_rise > actual_rise:
            note = f"Prediction lags actual CPU rise by {(predicted_rise - actual_rise):.1f}s"
            note_color = "#c62828"
        else:
            note = "Prediction and actual rise together"
            note_color = "#6a1b9a"

        x_text = min(predicted_rise, actual_rise) / 60.0
        y_text = max(float(np.max(predictive.cpu)), float(np.max(predictive.predicted_cpu))) * 0.88
        ax.annotate(
            note,
            xy=(x_text, y_text),
            xytext=(x_text + 0.15, y_text),
            arrowprops={"arrowstyle": "->", "color": note_color, "lw": 1.8},
            color=note_color,
            fontsize=12,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": note_color, "alpha": 0.92},
        )

    ax.set_title("Prediction vs Actual CPU")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel(predictive.cpu_label)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def print_summary(
    predictive: MetricsSeries,
    hpa: MetricsSeries,
    predictive_first_idx: Optional[int],
    hpa_first_idx: Optional[int],
    spike_start_seconds: Optional[float],
) -> None:
    def format_event(idx: Optional[int], series: MetricsSeries) -> str:
        if idx is None:
            return "not detected"
        return f"{series.seconds[idx]:.1f}s"

    predictive_first = predictive.seconds[predictive_first_idx] if predictive_first_idx is not None else None
    hpa_first = hpa.seconds[hpa_first_idx] if hpa_first_idx is not None else None

    print("Graph event summary")
    print(f"- Predictive first scale-up: {format_event(predictive_first_idx, predictive)}")
    print(f"- HPA first scale-up: {format_event(hpa_first_idx, hpa)}")
    print(f"- CPU spike start: {'not detected' if spike_start_seconds is None else f'{spike_start_seconds:.1f}s'}")

    if predictive_first is not None and hpa_first is not None:
        if predictive_first < hpa_first:
            print(f"- Result: Predictive scales earlier by {hpa_first - predictive_first:.1f}s")
        elif predictive_first > hpa_first:
            print(f"- Result: HPA scales earlier by {predictive_first - hpa_first:.1f}s")
        else:
            print("- Result: Both scale at the same observed time")


def main() -> int:
    args = parse_args()
    configure_style()

    predictive_path = Path(args.predictive)
    hpa_path = Path(args.hpa)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictive = maybe_convert_cpu_to_percent(load_metrics(predictive_path), args.predictive_cpu_limit_millicores)
    hpa = maybe_convert_cpu_to_percent(load_metrics(hpa_path), args.hpa_cpu_limit_millicores)

    predictive_first_idx = detect_first_scale_up(predictive)
    hpa_first_idx = detect_first_scale_up(hpa)
    spike_start_seconds = detect_spike_start([predictive, hpa])

    save_cpu_comparison(predictive, hpa, out_dir / "cpu_comparison.png", spike_start_seconds)
    save_replica_comparison(
        predictive,
        hpa,
        out_dir / "replica_comparison.png",
        predictive_first_idx,
        hpa_first_idx,
    )
    save_prediction_vs_actual(predictive, out_dir / "prediction_vs_actual.png", spike_start_seconds)

    print_summary(predictive, hpa, predictive_first_idx, hpa_first_idx, spike_start_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
