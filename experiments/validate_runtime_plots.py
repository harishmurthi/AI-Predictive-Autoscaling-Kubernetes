#!/usr/bin/env python3
import argparse
import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class RuntimeSeries:
    name: str
    timestamps: np.ndarray
    seconds: np.ndarray
    cpu: np.ndarray
    replicas: np.ndarray
    request_rate: Optional[np.ndarray] = None
    req_delta: Optional[np.ndarray] = None
    projected_cpu: Optional[np.ndarray] = None
    predicted_cpu: Optional[np.ndarray] = None
    reason: Optional[List[str]] = None
    cpu_unit: str = "percent"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict validation and publication-quality plotting for predictive vs HPA runtime logs."
    )
    parser.add_argument("--predictive", required=True, help="Path to predictive_runtime.csv")
    parser.add_argument("--hpa", required=True, help="Path to hpa_runtime.csv")
    parser.add_argument("--outdir", default="artifacts", help="Output directory")
    parser.add_argument("--resample-seconds", type=int, default=5, help="Uniform resampling grid step")
    parser.add_argument("--predictive-cpu-limit-millicores", type=float, default=None)
    parser.add_argument("--hpa-cpu-limit-millicores", type=float, default=None)
    parser.add_argument("--cpu-spike-threshold", type=float, default=65.0)
    parser.add_argument("--burst-rate-threshold", type=float, default=None)
    parser.add_argument("--burst-delta-threshold", type=float, default=None)
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (13.5, 6.8),
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "legend.fontsize": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.35,
            "lines.linewidth": 3.2,
        }
    )


def parse_timestamp(value: str) -> datetime:
    value = value.strip()
    if value.endswith("Z"):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return datetime.fromisoformat(value.replace(",", "."))


def get_first_present(row: Dict[str, str], names: Sequence[str]) -> Optional[str]:
    for name in names:
        if name in row and row[name] not in ("", None):
            return row[name]
    return None


def get_float(row: Dict[str, str], names: Sequence[str], default: Optional[float] = None) -> Optional[float]:
    value = get_first_present(row, names)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def infer_cpu_unit(fieldnames: Sequence[str]) -> str:
    if any(name in fieldnames for name in ("cpu_millicores", "cpu_mcore", "cpu_mc")):
        return "millicores"
    return "percent"


def load_runtime_csv(path: Path, name: str) -> RuntimeSeries:
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if not rows:
        raise ValueError(f"No rows found in {path}")

    cpu_unit = infer_cpu_unit(fieldnames)
    timestamps: List[datetime] = []
    cpu: List[float] = []
    replicas: List[float] = []
    request_rate: List[float] = []
    req_delta: List[float] = []
    projected_cpu: List[float] = []
    predicted_cpu: List[float] = []
    reasons: List[str] = []

    for row in rows:
        timestamps.append(parse_timestamp(row["timestamp"]))
        cpu_value = get_float(row, ["cpu", "actual_cpu", "cpu_actual", "cpu_millicores"])
        replica_value = get_float(row, ["replicas", "ai_replicas", "hpa_replicas"], 1.0)
        if cpu_value is None or replica_value is None:
            continue
        cpu.append(cpu_value)
        replicas.append(replica_value)
        request_rate.append(get_float(row, ["request_rate", "request_rate_rps"], 0.0) or 0.0)
        req_delta.append(get_float(row, ["req_delta", "request_rate_delta", "request_rate_delta_rps"], 0.0) or 0.0)
        projected_cpu.append(get_float(row, ["projected_cpu", "projected_cpu_percent"], math.nan) or math.nan)
        predicted_cpu.append(get_float(row, ["predicted_cpu", "cpu_predicted"], math.nan) or math.nan)
        reasons.append(get_first_present(row, ["reason", "decision_reason"]) or "")

    base_ts = timestamps[0]
    seconds = np.array([(ts - base_ts).total_seconds() for ts in timestamps], dtype=np.float64)

    return RuntimeSeries(
        name=name,
        timestamps=np.array(timestamps, dtype=object),
        seconds=seconds,
        cpu=np.array(cpu, dtype=np.float64),
        replicas=np.array(replicas, dtype=np.float64),
        request_rate=np.array(request_rate, dtype=np.float64),
        req_delta=np.array(req_delta, dtype=np.float64),
        projected_cpu=np.array(projected_cpu, dtype=np.float64),
        predicted_cpu=np.array(predicted_cpu, dtype=np.float64),
        reason=reasons,
        cpu_unit=cpu_unit,
    )


def convert_cpu_to_percent(series: RuntimeSeries, cpu_limit_millicores: Optional[float]) -> RuntimeSeries:
    if series.cpu_unit == "percent":
        return series
    if cpu_limit_millicores is None or cpu_limit_millicores <= 0:
        raise ValueError(
            f"CPU unit conversion required for {series.name}, but no valid CPU limit in millicores was provided."
        )
    converted_cpu = (series.cpu / cpu_limit_millicores) * 100.0
    return RuntimeSeries(
        name=series.name,
        timestamps=series.timestamps,
        seconds=series.seconds,
        cpu=converted_cpu,
        replicas=series.replicas,
        request_rate=series.request_rate,
        req_delta=series.req_delta,
        projected_cpu=series.projected_cpu,
        predicted_cpu=series.predicted_cpu,
        reason=series.reason,
        cpu_unit="percent",
    )


def resample_series(series: RuntimeSeries, step_seconds: int) -> RuntimeSeries:
    max_seconds = float(series.seconds[-1])
    target_seconds = np.arange(0.0, max_seconds + step_seconds, step_seconds, dtype=np.float64)

    def interp(values: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if values is None:
            return None
        if len(values) == 0:
            return None
        if np.all(np.isnan(values)):
            return values
        valid_mask = ~np.isnan(values)
        if not np.any(valid_mask):
            return np.full_like(target_seconds, np.nan, dtype=np.float64)
        return np.interp(target_seconds, series.seconds[valid_mask], values[valid_mask])

    resampled_timestamps = np.array(
        [series.timestamps[0] + (series.timestamps[1] - series.timestamps[0]) * 0 for _ in target_seconds],
        dtype=object,
    )

    return RuntimeSeries(
        name=series.name,
        timestamps=resampled_timestamps,
        seconds=target_seconds,
        cpu=np.interp(target_seconds, series.seconds, series.cpu),
        replicas=np.interp(target_seconds, series.seconds, series.replicas),
        request_rate=interp(series.request_rate),
        req_delta=interp(series.req_delta),
        projected_cpu=interp(series.projected_cpu),
        predicted_cpu=interp(series.predicted_cpu),
        reason=series.reason,
        cpu_unit=series.cpu_unit,
    )


def detect_first_scale_up(series: RuntimeSeries) -> Optional[float]:
    initial = series.replicas[0]
    for sec, replicas in zip(series.seconds, series.replicas):
        if replicas > initial + 1e-9:
            return float(sec)
    return None


def detect_cpu_spike(series: RuntimeSeries, threshold: float) -> Optional[float]:
    for sec, cpu in zip(series.seconds, series.cpu):
        if cpu >= threshold:
            return float(sec)
    return None


def detect_burst_start(
    predictive: RuntimeSeries,
    rate_threshold: Optional[float],
    delta_threshold: Optional[float],
) -> Optional[float]:
    if predictive.request_rate is None or predictive.req_delta is None:
        return None

    request_rate = predictive.request_rate
    req_delta = predictive.req_delta
    if np.allclose(request_rate, 0.0) and np.allclose(req_delta, 0.0):
        return None

    non_zero_rates = request_rate[request_rate > 0]
    positive_deltas = req_delta[req_delta > 0]
    if rate_threshold is None:
        rate_threshold = max(1.0, float(np.percentile(non_zero_rates, 70))) if non_zero_rates.size else 1.0
    if delta_threshold is None:
        delta_threshold = max(0.5, float(np.percentile(positive_deltas, 70))) if positive_deltas.size else 0.5

    for sec, rate, delta in zip(predictive.seconds, request_rate, req_delta):
        if rate >= rate_threshold or delta >= delta_threshold:
            return float(sec)
    return None


def validate_telemetry(predictive: RuntimeSeries, t_burst_start: Optional[float], t_cpu_spike: Optional[float]) -> List[str]:
    errors: List[str] = []
    if predictive.request_rate is None or predictive.req_delta is None:
        errors.append("predictive_runtime is missing request_rate or req_delta columns")
        return errors
    if np.allclose(predictive.request_rate, 0.0):
        errors.append("request_rate is all zeros")
    if not np.any(predictive.req_delta > 0):
        errors.append("req_delta never shows an early positive jump")
    if t_burst_start is None:
        errors.append("could not detect burst start from request_rate/req_delta")
    if t_burst_start is not None and t_cpu_spike is not None and t_burst_start >= t_cpu_spike:
        errors.append("request_rate signal does not rise before CPU spike")
    return errors


def area_above_threshold(series: RuntimeSeries, threshold: float) -> float:
    excess = np.maximum(series.cpu - threshold, 0.0)
    if len(excess) < 2:
        return 0.0
    return float(np.trapz(excess, series.seconds))


def seconds_to_minutes(seconds: np.ndarray) -> np.ndarray:
    return seconds / 60.0


def add_vertical_line(ax, seconds_value: Optional[float], label: str, color: str, linestyle: str = "--") -> None:
    if seconds_value is None:
        return
    ax.axvline(seconds_value / 60.0, color=color, linestyle=linestyle, linewidth=2.0, alpha=0.85, label=label)


def save_replica_plot(
    predictive: RuntimeSeries,
    hpa: RuntimeSeries,
    out_path: Path,
    t_burst_start: Optional[float],
    t_pred_scale: Optional[float],
    t_hpa_scale: Optional[float],
) -> None:
    fig, ax = plt.subplots()
    ax.step(seconds_to_minutes(predictive.seconds), predictive.replicas, where="post", label="Predictive replicas", color="#d62728")
    ax.step(seconds_to_minutes(hpa.seconds), hpa.replicas, where="post", label="HPA replicas", color="#1f77b4")

    add_vertical_line(ax, t_burst_start, "Burst start", "#ff7f0e")
    add_vertical_line(ax, t_pred_scale, "Predictive first scale-up", "#d62728", ":")
    add_vertical_line(ax, t_hpa_scale, "HPA first scale-up", "#1f77b4", ":")

    if t_pred_scale is not None and t_hpa_scale is not None:
        if t_pred_scale < t_hpa_scale:
            text = f"Predictive scales earlier by {t_hpa_scale - t_pred_scale:.1f}s"
            color = "#2e7d32"
        else:
            text = f"HPA scales earlier by {t_pred_scale - t_hpa_scale:.1f}s"
            color = "#c62828"
        y_anchor = max(float(np.max(predictive.replicas)), float(np.max(hpa.replicas))) * 0.9
        x_anchor = min(t_pred_scale, t_hpa_scale) / 60.0
        ax.annotate(
            text,
            xy=(x_anchor, y_anchor),
            xytext=(x_anchor + 0.2, y_anchor + 0.5),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.8},
            color=color,
            fontsize=12,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": color, "alpha": 0.92},
        )

    ax.set_title("Replica Comparison: Predictive vs HPA")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Replicas")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_cpu_plot(
    predictive: RuntimeSeries,
    hpa: RuntimeSeries,
    out_path: Path,
    t_burst_start: Optional[float],
    t_cpu_spike_pred: Optional[float],
    t_cpu_spike_hpa: Optional[float],
    threshold: float,
) -> None:
    fig, ax = plt.subplots()
    ax.plot(seconds_to_minutes(predictive.seconds), predictive.cpu, label="Predictive CPU", color="#d62728")
    ax.plot(seconds_to_minutes(hpa.seconds), hpa.cpu, label="HPA CPU", color="#1f77b4")

    if t_burst_start is not None:
        max_minutes = max(float(np.max(seconds_to_minutes(predictive.seconds))), float(np.max(seconds_to_minutes(hpa.seconds))))
        ax.axvspan(t_burst_start / 60.0, max_minutes, color="#ffcc80", alpha=0.18, label="Burst window")

    add_vertical_line(ax, t_burst_start, "Burst start", "#ff7f0e")
    add_vertical_line(ax, t_cpu_spike_pred, "Predictive CPU spike", "#d62728", ":")
    add_vertical_line(ax, t_cpu_spike_hpa, "HPA CPU spike", "#1f77b4", ":")
    ax.axhline(threshold, color="#6d4c41", linestyle="--", linewidth=1.8, alpha=0.85, label=f"High CPU threshold ({threshold:.0f}%)")

    pred_peak = float(np.max(predictive.cpu))
    hpa_peak = float(np.max(hpa.cpu))
    ax.annotate(
        f"Predictive peak: {pred_peak:.1f}%",
        xy=(seconds_to_minutes(predictive.seconds[np.argmax(predictive.cpu)]), pred_peak),
        fontsize=13,
    )
    ax.annotate(
        f"HPA peak: {hpa_peak:.1f}%",
        xy=(seconds_to_minutes(hpa.seconds[np.argmax(hpa.cpu)]), hpa_peak),
        fontsize=13,
    )

    ax.set_title("CPU Comparison")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("CPU (%)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_prediction_plot(
    predictive: RuntimeSeries,
    out_path: Path,
    t_burst_start: Optional[float],
) -> None:
    if predictive.projected_cpu is None or np.all(np.isnan(predictive.projected_cpu)):
        raise ValueError("predictive_runtime must include projected_cpu for prediction_vs_actual plot")

    fig, ax = plt.subplots()
    ax.plot(seconds_to_minutes(predictive.seconds), predictive.cpu, label="Actual CPU", color="#1f77b4")
    ax.plot(seconds_to_minutes(predictive.seconds), predictive.projected_cpu, label="Projected CPU", color="#d62728")
    if predictive.predicted_cpu is not None and not np.all(np.isnan(predictive.predicted_cpu)):
        ax.plot(seconds_to_minutes(predictive.seconds), predictive.predicted_cpu, label="Predicted CPU", color="#9467bd", linewidth=2.4)

    add_vertical_line(ax, t_burst_start, "Burst start", "#ff7f0e")
    ax.set_title("Prediction vs Actual CPU")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("CPU (%)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    out_path: Path,
    telemetry_errors: List[str],
    t_burst_start: Optional[float],
    t_pred_scale: Optional[float],
    t_hpa_scale: Optional[float],
    t_cpu_spike_pred: Optional[float],
    t_cpu_spike_hpa: Optional[float],
    predictive_delay: Optional[float],
    hpa_delay: Optional[float],
    claim_supported: bool,
    predictive_area: float,
    hpa_area: float,
) -> None:
    with out_path.open("w", encoding="utf-8") as fp:
        fp.write("Runtime validation summary\n")
        fp.write(f"telemetry_errors={telemetry_errors}\n")
        fp.write(f"t_burst_start={t_burst_start}\n")
        fp.write(f"t_pred_scale={t_pred_scale}\n")
        fp.write(f"t_hpa_scale={t_hpa_scale}\n")
        fp.write(f"t_cpu_spike_predictive={t_cpu_spike_pred}\n")
        fp.write(f"t_cpu_spike_hpa={t_cpu_spike_hpa}\n")
        fp.write(f"predictive_delay={predictive_delay}\n")
        fp.write(f"hpa_delay={hpa_delay}\n")
        fp.write(f"claim_supported={claim_supported}\n")
        fp.write(f"predictive_high_cpu_area={predictive_area}\n")
        fp.write(f"hpa_high_cpu_area={hpa_area}\n")
        if hpa_area > 0:
            reduction = ((hpa_area - predictive_area) / hpa_area) * 100.0
            fp.write(f"high_cpu_area_reduction_percent={reduction}\n")


def print_summary_table(
    telemetry_errors: List[str],
    t_burst_start: Optional[float],
    t_pred_scale: Optional[float],
    t_hpa_scale: Optional[float],
    predictive_delay: Optional[float],
    hpa_delay: Optional[float],
    claim_supported: bool,
) -> None:
    print("=== Runtime Validation Summary ===")
    print(f"t_burst_start: {t_burst_start}")
    print(f"t_pred_scale: {t_pred_scale}")
    print(f"t_hpa_scale: {t_hpa_scale}")
    print(f"predictive_delay: {predictive_delay}")
    print(f"hpa_delay: {hpa_delay}")
    print(f"claim_supported: {claim_supported}")
    if telemetry_errors:
        print(f"telemetry_errors: {telemetry_errors}")


def main() -> int:
    args = parse_args()
    configure_style()
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictive = load_runtime_csv(Path(args.predictive), "predictive")
    hpa = load_runtime_csv(Path(args.hpa), "hpa")

    predictive = convert_cpu_to_percent(predictive, args.predictive_cpu_limit_millicores)
    hpa = convert_cpu_to_percent(hpa, args.hpa_cpu_limit_millicores)

    if predictive.cpu_unit != hpa.cpu_unit:
        raise ValueError("CPU units still do not match after conversion")

    predictive = resample_series(predictive, args.resample_seconds)
    hpa = resample_series(hpa, args.resample_seconds)

    t_cpu_spike_pred = detect_cpu_spike(predictive, args.cpu_spike_threshold)
    t_cpu_spike_hpa = detect_cpu_spike(hpa, args.cpu_spike_threshold)
    t_cpu_spike = min([t for t in [t_cpu_spike_pred, t_cpu_spike_hpa] if t is not None], default=None)
    t_burst_start = detect_burst_start(predictive, args.burst_rate_threshold, args.burst_delta_threshold)

    telemetry_errors = validate_telemetry(predictive, t_burst_start, t_cpu_spike)
    if telemetry_errors:
        print("INVALID RUN: telemetry does not support proactive scaling")
        for error in telemetry_errors:
            print(f"- {error}")
        write_summary(
            out_dir / "summary.txt",
            telemetry_errors,
            t_burst_start,
            None,
            None,
            t_cpu_spike_pred,
            t_cpu_spike_hpa,
            None,
            None,
            False,
            area_above_threshold(predictive, args.cpu_spike_threshold),
            area_above_threshold(hpa, args.cpu_spike_threshold),
        )
        return 1

    t_pred_scale = detect_first_scale_up(predictive)
    t_hpa_scale = detect_first_scale_up(hpa)

    predictive_delay = None if t_burst_start is None or t_pred_scale is None else t_pred_scale - t_burst_start
    hpa_delay = None if t_burst_start is None or t_hpa_scale is None else t_hpa_scale - t_burst_start
    claim_supported = (
        t_pred_scale is not None
        and t_hpa_scale is not None
        and t_pred_scale < t_hpa_scale
    )

    if not claim_supported:
        print("CLAIM NOT SUPPORTED: HPA scales earlier in this run")
    if predictive_delay is not None and predictive_delay > 10.0:
        print("WARNING: Predictive not early enough (<10s expected)")

    save_replica_plot(
        predictive,
        hpa,
        out_dir / "replica_comparison.png",
        t_burst_start,
        t_pred_scale,
        t_hpa_scale,
    )
    save_cpu_plot(
        predictive,
        hpa,
        out_dir / "cpu_comparison.png",
        t_burst_start,
        t_cpu_spike_pred,
        t_cpu_spike_hpa,
        args.cpu_spike_threshold,
    )
    save_prediction_plot(
        predictive,
        out_dir / "prediction_vs_actual.png",
        t_burst_start,
    )

    predictive_area = area_above_threshold(predictive, args.cpu_spike_threshold)
    hpa_area = area_above_threshold(hpa, args.cpu_spike_threshold)

    write_summary(
        out_dir / "summary.txt",
        telemetry_errors,
        t_burst_start,
        t_pred_scale,
        t_hpa_scale,
        t_cpu_spike_pred,
        t_cpu_spike_hpa,
        predictive_delay,
        hpa_delay,
        claim_supported,
        predictive_area,
        hpa_area,
    )

    print_summary_table(
        telemetry_errors,
        t_burst_start,
        t_pred_scale,
        t_hpa_scale,
        predictive_delay,
        hpa_delay,
        claim_supported,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
