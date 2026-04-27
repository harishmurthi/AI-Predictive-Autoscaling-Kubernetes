#!/usr/bin/env python3
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class TimelinePoint:
    ts: float
    replicas: float
    cpu_millicores: float


def parse_iso_utc(ts: str) -> float:
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").timestamp()


def load_timeline(path: Path) -> List[TimelinePoint]:
    rows: List[TimelinePoint] = []
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows.append(
                TimelinePoint(
                    ts=float(row["unix_ts"]),
                    replicas=float(row["replicas"]),
                    cpu_millicores=float(row["cpu_millicores"]),
                )
            )
    return rows


def replica_seconds(points: List[TimelinePoint]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(points, points[1:]):
        dt = max(0.0, b.ts - a.ts)
        total += a.replicas * dt
    return total


def first_scale_up_delay(points: List[TimelinePoint]) -> float:
    if not points:
        return math.inf
    initial = points[0].replicas
    start = points[0].ts
    for p in points:
        if p.replicas > initial:
            return p.ts - start
    return math.inf


def count_scale_events(points: List[TimelinePoint]) -> Tuple[int, int]:
    up = 0
    down = 0
    if len(points) < 2:
        return up, down
    for a, b in zip(points, points[1:]):
        if b.replicas > a.replicas:
            up += 1
        elif b.replicas < a.replicas:
            down += 1
    return up, down


def load_predictive_metrics(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    actual, predicted, replicas = [], [], []
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            try:
                actual.append(float(row["actual_cpu"]))
                predicted.append(float(row["predicted_cpu"]))
                replicas.append(float(row["replicas"]))
            except (KeyError, ValueError):
                continue
    return np.array(actual), np.array(predicted), np.array(replicas)


def rmse(y, yhat) -> float:
    if len(y) == 0:
        return 0.0
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mae(y, yhat) -> float:
    if len(y) == 0:
        return 0.0
    return float(np.mean(np.abs(y - yhat)))


def mape(y, yhat) -> float:
    if len(y) == 0:
        return 0.0
    denom = np.where(np.abs(y) > 1e-9, y, np.nan)
    value = np.nanmean(np.abs((y - yhat) / denom)) * 100.0
    return 0.0 if np.isnan(value) else float(value)


def plot_actual_vs_pred(actual: np.ndarray, predicted: np.ndarray, out: Path) -> None:
    plt.figure(figsize=(10, 4))
    x = np.arange(len(actual))
    plt.plot(x, actual, label="Actual CPU (%)", linewidth=2)
    plt.plot(x, predicted, label="Predicted CPU (%)", linewidth=2)
    plt.title("Actual vs Predicted CPU")
    plt.xlabel("Sample")
    plt.ylabel("CPU (%)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_replicas(hpa: List[TimelinePoint], pred: List[TimelinePoint], out: Path) -> None:
    plt.figure(figsize=(10, 4))
    hpa_x = np.arange(len(hpa))
    pred_x = np.arange(len(pred))
    plt.plot(hpa_x, [p.replicas for p in hpa], label="HPA Replicas", linewidth=2)
    plt.plot(pred_x, [p.replicas for p in pred], label="Predictive Replicas", linewidth=2)
    plt.title("HPA vs Predictive Replica Timeline")
    plt.xlabel("Sample")
    plt.ylabel("Replicas")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_resource_usage(hpa: List[TimelinePoint], pred: List[TimelinePoint], out: Path) -> None:
    hpa_avg_cpu = np.mean([p.cpu_millicores for p in hpa]) if hpa else 0.0
    pred_avg_cpu = np.mean([p.cpu_millicores for p in pred]) if pred else 0.0
    hpa_rep_s = replica_seconds(hpa)
    pred_rep_s = replica_seconds(pred)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(["HPA", "Predictive"], [hpa_avg_cpu, pred_avg_cpu])
    ax[0].set_title("Average CPU Usage (millicores)")
    ax[0].set_ylabel("millicores")
    ax[0].grid(True, axis="y", linestyle="--", alpha=0.4)

    ax[1].bar(["HPA", "Predictive"], [hpa_rep_s, pred_rep_s])
    ax[1].set_title("Total Replica-Seconds")
    ax[1].set_ylabel("replica-seconds")
    ax[1].grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def main() -> int:
    if len(sys.argv) != 6:
        print(
            "Usage: evaluate.py <hpa_results.csv> <predictive_results.csv> <predictive.csv> <output_dir> <meta.json>",
            file=sys.stderr,
        )
        return 2

    hpa_csv = Path(sys.argv[1])
    pred_csv = Path(sys.argv[2])
    predictive_metrics_csv = Path(sys.argv[3])
    out_dir = Path(sys.argv[4])
    meta_json = Path(sys.argv[5])
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in [hpa_csv, pred_csv, predictive_metrics_csv, meta_json]:
        if not path.exists():
            print(f"ERROR: Missing required file: {path}", file=sys.stderr)
            return 1

    hpa_points = load_timeline(hpa_csv)
    pred_points = load_timeline(pred_csv)
    actual, predicted, _ = load_predictive_metrics(predictive_metrics_csv)

    with meta_json.open("r", encoding="utf-8") as fp:
        meta = json.load(fp)

    rmse_v = rmse(actual, predicted)
    mae_v = mae(actual, predicted)
    mape_v = mape(actual, predicted)

    hpa_replica_seconds = replica_seconds(hpa_points)
    pred_replica_seconds = replica_seconds(pred_points)

    hpa_react = first_scale_up_delay(hpa_points)
    pred_react = first_scale_up_delay(pred_points)

    if hpa_replica_seconds > 0:
        resource_savings = ((hpa_replica_seconds - pred_replica_seconds) / hpa_replica_seconds) * 100.0
    else:
        resource_savings = 0.0

    lead_time = hpa_react - pred_react if math.isfinite(hpa_react) and math.isfinite(pred_react) else float("nan")
    hpa_up, hpa_down = count_scale_events(hpa_points)
    pred_up, pred_down = count_scale_events(pred_points)

    plot_actual_vs_pred(actual, predicted, out_dir / "actual_vs_predicted_cpu.png")
    plot_replicas(hpa_points, pred_points, out_dir / "hpa_vs_predictive_replicas.png")
    plot_resource_usage(hpa_points, pred_points, out_dir / "resource_usage_comparison.png")

    summary = {
        "prediction_rmse": rmse_v,
        "prediction_mae": mae_v,
        "prediction_mape": mape_v,
        "hpa_replica_seconds": hpa_replica_seconds,
        "predictive_replica_seconds": pred_replica_seconds,
        "resource_savings_percent": resource_savings,
        "hpa_scaleup_reaction_seconds": hpa_react,
        "predictive_scaleup_reaction_seconds": pred_react,
        "scaling_lead_time_seconds_hpa_minus_predictive": lead_time,
        "hpa_scale_up_events": hpa_up,
        "hpa_scale_down_events": hpa_down,
        "predictive_scale_up_events": pred_up,
        "predictive_scale_down_events": pred_down,
        "workload_config": meta.get("workload_config", {}),
        "image": meta.get("image"),
        "image_digest": meta.get("image_digest"),
    }

    scientifically_valid = (
        len(actual) >= 20
        and len(hpa_points) >= 20
        and len(pred_points) >= 20
        and rmse_v > 0.0
        and pred_up >= 2
        and pred_down >= 2
        and not math.isnan(lead_time)
    )
    summary["scientifically_valid"] = scientifically_valid

    with (out_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print("=== Research Summary ===")
    print(f"RMSE: {rmse_v:.4f}")
    print(f"MAE: {mae_v:.4f}")
    print(f"MAPE: {mape_v:.4f}%")
    print(f"HPA replica-seconds: {hpa_replica_seconds:.2f}")
    print(f"Predictive replica-seconds: {pred_replica_seconds:.2f}")
    print(f"Resource savings: {resource_savings:.2f}%")
    print(f"HPA scale-up reaction: {hpa_react:.2f}s")
    print(f"Predictive scale-up reaction: {pred_react:.2f}s")
    print(f"Scaling lead time (HPA - Predictive): {lead_time:.2f}s")
    print(f"HPA events: scale-up={hpa_up}, scale-down={hpa_down}")
    print(f"Predictive events: scale-up={pred_up}, scale-down={pred_down}")
    print(f"Scientifically valid for report: {'YES' if scientifically_valid else 'NO'}")
    if not scientifically_valid:
        print("Reason: insufficient data points and/or missing measurable scale-up events.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
