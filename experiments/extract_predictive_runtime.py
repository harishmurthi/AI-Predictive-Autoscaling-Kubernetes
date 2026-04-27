#!/usr/bin/env python3
import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional


COLLECTED_RE = re.compile(
    r'^(?P<ts>\S+ \S+) .* \[data_collector\] Collected metrics: '
    r'cpu=(?P<cpu>-?[0-9.]+)% .* req_rate=(?P<req>-?[0-9.]+) rps replicas=(?P<rep>[0-9]+)'
)

LOOP_RE = re.compile(
    r'^(?P<ts>\S+ \S+) .* \[__main__\] Loop summary '
    r'actual_cpu=(?P<cpu>-?[0-9.]+) predicted_cpu=(?P<pred>-?[0-9.]+) '
    r'action=(?P<action>\S+) replicas=(?P<rep>[0-9]+)'
)

DECISION_RE = re.compile(
    r'^(?P<ts>\S+ \S+) .* \[__main__\] Decision details: '
    r'actual_cpu=(?P<cpu>-?[0-9.]+) predicted_cpu=(?P<pred>-?[0-9.]+)'
    r'(?: projected_cpu=(?P<proj>-?[0-9.]+))?'
    r'(?: req_rate=(?P<req>-?[0-9.]+))?'
    r'(?: req_delta=(?P<delta>-?[0-9.]+))?'
    r'.* current_replicas=(?P<rep>[0-9]+) decision_reason=(?P<reason>.*)$'
)

RUNTIME_RE = re.compile(r'^(?P<prefix>\S+ \S+) .* \[__main__\] RUNTIME_METRIC (?P<body>.*)$')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract predictive runtime CSV from autoscaler logs.")
    parser.add_argument("--log", required=True, help="Path to predictive autoscaler log")
    parser.add_argument("--out", required=True, help="Path to predictive_runtime.csv output")
    parser.add_argument(
        "--metrics-csv",
        default=None,
        help="Optional raw /app/metrics_log.csv copied from the pod. Used first if available.",
    )
    return parser.parse_args()


def to_isoish(ts: str) -> str:
    return ts.replace(" ", "T", 1)


def parse_runtime_body(body: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for key, value in re.findall(r'([a-zA-Z_]+)=(".*?"|\S+)', body):
        result[key] = value.strip('"')
    return result


def load_metrics_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    normalized: List[Dict[str, str]] = []
    prev_req = 0.0
    for row in rows:
        try:
            req = float(row.get("request_rate", 0.0) or 0.0)
        except ValueError:
            req = 0.0
        try:
            delta = float(row.get("req_delta", "")) if row.get("req_delta", "") not in ("", None) else req - prev_req
        except ValueError:
            delta = req - prev_req
        normalized.append(
            {
                "timestamp": row.get("timestamp", ""),
                "cpu": str(row.get("cpu", row.get("actual_cpu", 0.0) or 0.0)),
                "replicas": str(row.get("replicas", 1)),
                "request_rate": f"{req:.3f}",
                "req_delta": f"{delta:.3f}",
                "projected_cpu": str(row.get("projected_cpu", row.get("predicted_cpu", row.get("cpu_predicted", 0.0) or 0.0))),
                "predicted_cpu": str(row.get("predicted_cpu", row.get("cpu_predicted", row.get("projected_cpu", 0.0) or 0.0))),
                "reason": row.get("reason", ""),
            }
        )
        prev_req = req
    return normalized


def extract_from_logs(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    current_metrics: Dict[str, str] = {}
    prev_req = 0.0
    parsed_lines = 0

    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            if not line:
                continue

            match = COLLECTED_RE.search(line)
            if match:
                parsed_lines += 1
                current_metrics = {
                    "timestamp": to_isoish(match.group("ts")),
                    "cpu": match.group("cpu"),
                    "replicas": match.group("rep"),
                    "request_rate": match.group("req"),
                }
                continue

            match = RUNTIME_RE.search(line)
            if match:
                parsed_lines += 1
                body = parse_runtime_body(match.group("body"))
                try:
                    req = float(body.get("request_rate", 0.0) or 0.0)
                except ValueError:
                    req = 0.0
                try:
                    delta = float(body.get("req_delta", "")) if body.get("req_delta", "") not in ("", None) else req - prev_req
                except ValueError:
                    delta = req - prev_req
                rows.append(
                    {
                        "timestamp": body.get("timestamp", to_isoish(match.group("prefix"))),
                        "cpu": str(body.get("cpu", 0.0)),
                        "replicas": str(body.get("replicas", 1)),
                        "request_rate": f"{req:.3f}",
                        "req_delta": f"{delta:.3f}",
                        "projected_cpu": str(body.get("projected_cpu", body.get("predicted_cpu", 0.0))),
                        "predicted_cpu": str(body.get("predicted_cpu", body.get("projected_cpu", 0.0))),
                        "reason": body.get("reason", ""),
                    }
                )
                prev_req = req
                continue

            match = DECISION_RE.search(line)
            if match:
                parsed_lines += 1
                req_str = match.group("req") or current_metrics.get("request_rate", "0.0")
                try:
                    req = float(req_str)
                except ValueError:
                    req = 0.0
                try:
                    delta = float(match.group("delta")) if match.group("delta") not in (None, "") else req - prev_req
                except ValueError:
                    delta = req - prev_req
                pred = match.group("pred")
                proj = match.group("proj") or pred
                rows.append(
                    {
                        "timestamp": to_isoish(match.group("ts")),
                        "cpu": match.group("cpu") or current_metrics.get("cpu", "0.0"),
                        "replicas": match.group("rep") or current_metrics.get("replicas", "1"),
                        "request_rate": f"{req:.3f}",
                        "req_delta": f"{delta:.3f}",
                        "projected_cpu": proj,
                        "predicted_cpu": pred,
                        "reason": match.group("reason"),
                    }
                )
                prev_req = req
                continue

            match = LOOP_RE.search(line)
            if match:
                parsed_lines += 1
                req = 0.0
                try:
                    req = float(current_metrics.get("request_rate", 0.0) or 0.0)
                except ValueError:
                    req = 0.0
                delta = req - prev_req
                rows.append(
                    {
                        "timestamp": to_isoish(match.group("ts")),
                        "cpu": match.group("cpu"),
                        "replicas": match.group("rep"),
                        "request_rate": f"{req:.3f}",
                        "req_delta": f"{delta:.3f}",
                        "projected_cpu": match.group("pred"),
                        "predicted_cpu": match.group("pred"),
                        "reason": match.group("action"),
                    }
                )
                prev_req = req

    print(f"Parsed log lines: {parsed_lines}")
    return rows


def dedupe_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    by_ts: Dict[str, Dict[str, str]] = {}
    for row in rows:
        by_ts[row["timestamp"]] = row
    return [by_ts[key] for key in sorted(by_ts.keys())]


def write_rows(rows: List[Dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "timestamp",
                "cpu",
                "replicas",
                "request_rate",
                "req_delta",
                "projected_cpu",
                "predicted_cpu",
                "reason",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    out_path = Path(args.out)
    rows: List[Dict[str, str]] = []

    if args.metrics_csv:
        metrics_path = Path(args.metrics_csv)
        if metrics_path.exists() and metrics_path.stat().st_size > 0:
            rows = load_metrics_csv(metrics_path)
            print(f"Loaded pod metrics CSV rows: {len(rows)}")

    if not rows:
        log_path = Path(args.log)
        if not log_path.exists():
            print("Predictive logs missing or invalid: log file not found", file=sys.stderr)
            write_rows([], out_path)
            return 1
        rows = extract_from_logs(log_path)

    rows = dedupe_rows(rows)
    write_rows(rows, out_path)
    print(f"CSV rows written: {len(rows)}")
    if rows:
        print("Sample row preview:")
        print(rows[0])
        return 0

    print("Predictive logs missing or invalid", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
