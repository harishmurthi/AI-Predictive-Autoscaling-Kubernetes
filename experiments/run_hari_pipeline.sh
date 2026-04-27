#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ccbd/AIPREDICTIVE/predictive"
HARI_DIR="$PROJECT_ROOT/hari"
LOGS_DIR="$HARI_DIR/logs"
GRAPHS_DIR="$HARI_DIR/graphs"
METRICS_DIR="$HARI_DIR/metrics"
CONFIGS_DIR="$HARI_DIR/configs"
SUMMARY_DIR="$HARI_DIR/summary"

IMAGE="${IMAGE:-hari053/predictive-autoscaler:latest}"
SKIP_BUILD_PUSH="${SKIP_BUILD_PUSH:-0}"
AUTO_NS="autoscale-demo"
MON_NS="monitoring"
AUTO_DEPLOY="predictive-autoscaler"
TARGET_DEPLOY="cpu-demo"
TARGET_SERVICE="cpu-demo"
LOCUST_USERS="${LOCUST_USERS:-400}"
LOCUST_SPAWN_RATE="${LOCUST_SPAWN_RATE:-120}"
BURST_DURATION_SEC="${BURST_DURATION_SEC:-120}"
IDLE_DURATION_SEC="${IDLE_DURATION_SEC:-120}"
SAMPLE_INTERVAL_SEC="${SAMPLE_INTERVAL_SEC:-5}"
CPU_LIMIT_MILLICORES="${CPU_LIMIT_MILLICORES:-1000}"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
fail() { echo "ERROR: $*" >&2; exit 1; }
run_kubectl() { kubectl "$@"; }

create_output_structure() {
  mkdir -p "$LOGS_DIR" "$GRAPHS_DIR" "$METRICS_DIR" "$CONFIGS_DIR" "$SUMMARY_DIR"
}

save_configs() {
  cp "$PROJECT_ROOT/k8s/autoscaler-deployment.yaml" "$CONFIGS_DIR/autoscaler-deployment.yaml"
  cp "$PROJECT_ROOT/k8s/rbac.yaml" "$CONFIGS_DIR/rbac.yaml"
  cp "$PROJECT_ROOT/locust/locustfile.py" "$CONFIGS_DIR/locustfile.py"
  cat > "$CONFIGS_DIR/autoscaler-params.env" <<EOF
LOOP_INTERVAL_SECONDS=5
COOLDOWN_SECONDS=8
SCALE_UP_MARGIN=18
BURST_RPS_DELTA_THRESHOLD=2
ABSOLUTE_BURST_RPS_THRESHOLD=10
BURST_SCALE_STEP=3
MAX_SCALE_UP_STEP=5
COOLDOWN_BYPASS_MARGIN=4
SCALE_DOWN_STREAK_REQUIRED=3
CPU_RATE_WINDOW=15s
REQUEST_RATE_WINDOW=15s
EOF
  run_kubectl -n monitoring get configmap -o yaml > "$CONFIGS_DIR/monitoring-configmaps.yaml" || true
}

verify_environment() {
  run_kubectl get nodes -o wide > "$SUMMARY_DIR/nodes.txt"
  run_kubectl get ns "$MON_NS" "$AUTO_NS" > "$SUMMARY_DIR/namespaces.txt"
  run_kubectl -n "$MON_NS" get svc prometheus-server > "$SUMMARY_DIR/prometheus-service.txt"
  run_kubectl -n "$AUTO_NS" get deploy "$TARGET_DEPLOY" "$AUTO_DEPLOY" -o wide > "$SUMMARY_DIR/deployments.txt"
}

ensure_target_service() {
  cat <<YAML | run_kubectl apply -f - >/dev/null
apiVersion: v1
kind: Service
metadata:
  name: ${TARGET_SERVICE}
  namespace: ${AUTO_NS}
spec:
  selector:
    app: ${TARGET_DEPLOY}
  ports:
  - port: 80
    targetPort: 80
    protocol: TCP
YAML
}

ensure_http_ready_target() {
  local container_name
  container_name="$(run_kubectl -n "$AUTO_NS" get deployment "$TARGET_DEPLOY" -o jsonpath='{.spec.template.spec.containers[0].name}')"
  local target_image="nginx:1.27-alpine"
  run_kubectl -n "$AUTO_NS" set image "deployment/${TARGET_DEPLOY}" "${container_name}=${target_image}" >/dev/null
  run_kubectl -n "$AUTO_NS" patch "deployment/${TARGET_DEPLOY}" --type='merge' -p '{
    "spec": {
      "replicas": 1,
      "template": {
        "spec": {
          "containers": [{
            "name": "'"${container_name}"'",
            "image": "'"${target_image}"'",
            "ports": [{"containerPort": 80}],
            "resources": {
              "requests": {"cpu": "100m", "memory": "128Mi"},
              "limits": {"cpu": "250m", "memory": "256Mi"}
            }
          }]
        }
      }
    }
  }' >/dev/null
  run_kubectl -n "$AUTO_NS" rollout restart "deployment/${TARGET_DEPLOY}" >/dev/null
  run_kubectl -n "$AUTO_NS" rollout status "deployment/${TARGET_DEPLOY}" --timeout=180s >/dev/null
}

ensure_locust_configmap() {
  run_kubectl -n "$AUTO_NS" delete configmap locustfile-cm --ignore-not-found >/dev/null
  run_kubectl -n "$AUTO_NS" create configmap locustfile-cm --from-file=locustfile.py="$PROJECT_ROOT/locust/locustfile.py" >/dev/null
}

cleanup_old_locust_jobs() {
  local jobs
  jobs="$(run_kubectl -n "$AUTO_NS" get jobs -o name | grep 'job.batch/locust-' || true)"
  if [[ -n "$jobs" ]]; then
    while IFS= read -r job; do
      [[ -n "$job" ]] || continue
      run_kubectl -n "$AUTO_NS" delete "$job" --ignore-not-found >/dev/null
    done <<< "$jobs"
  fi
}

write_hpa_manifest() {
  cat <<YAML
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ${TARGET_DEPLOY}
  namespace: ${AUTO_NS}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ${TARGET_DEPLOY}
  minReplicas: 1
  maxReplicas: 8
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 55
YAML
}

sample_runtime() {
  local outfile="$1"
  local mode="$2"
  echo "timestamp,cpu_millicores,replicas" > "$outfile"
  local start now elapsed ts replicas top_output cpu_m
  start="$(date +%s)"
  while true; do
    now="$(date +%s)"
    elapsed=$((now - start))
    (( elapsed > BURST_DURATION_SEC + IDLE_DURATION_SEC )) && break
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    replicas="$(run_kubectl -n "$AUTO_NS" get deployment "$TARGET_DEPLOY" -o jsonpath='{.spec.replicas}' 2>/dev/null || echo 0)"
    top_output="$(run_kubectl top pods -n "$AUTO_NS" -l app="$TARGET_DEPLOY" --no-headers 2>/dev/null || true)"
    if [[ -z "$top_output" ]]; then
      cpu_m=0
    else
      cpu_m="$(echo "$top_output" | awk '
        function to_m(v){ if (v ~ /m$/) { sub(/m$/, "", v); return v + 0 } return (v + 0) * 1000 }
        {sum += to_m($2)} END {printf("%.0f", sum)}
      ')"
    fi
    echo "$ts,$cpu_m,$replicas" >> "$outfile"
    sleep "$SAMPLE_INTERVAL_SEC"
  done
}

run_locust_job() {
  local name="$1"
  local duration="$2"
  local users="$3"
  local spawn_rate="$4"
  local job_name="locust-${name}-$(date +%s)"
  cat <<YAML | run_kubectl apply -f - >/dev/null
apiVersion: batch/v1
kind: Job
metadata:
  name: ${job_name}
  namespace: ${AUTO_NS}
spec:
  ttlSecondsAfterFinished: 300
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: locust
        image: locustio/locust:2.31.6
        env:
        - name: BURST_WINDOW_SECONDS
          value: "20"
        - name: BURST_MULTIPLIER
          value: "12"
        command: ["/bin/sh", "-c"]
        args:
        - |
          locust -f /mnt/locust/locustfile.py \
            --host=http://${TARGET_SERVICE}.${AUTO_NS}.svc.cluster.local \
            --headless \
            -u ${users} \
            -r ${spawn_rate} \
            -t ${duration}s \
            --exit-code-on-error 0 \
            --only-summary
        volumeMounts:
        - name: locustfile
          mountPath: /mnt/locust
      volumes:
      - name: locustfile
        configMap:
          name: locustfile-cm
YAML
  run_kubectl -n "$AUTO_NS" wait --for=condition=complete "job/${job_name}" --timeout=$((duration + 300))s >/dev/null
  run_kubectl -n "$AUTO_NS" logs "job/${job_name}" > "$LOGS_DIR/${name}_locust.log" || true
  if grep -q "ConnectionRefusedError" "$LOGS_DIR/${name}_locust.log"; then
    fail "Locust workload failed with connection errors for ${name}"
  fi
}

run_workload() {
  local prefix="$1"
  run_locust_job "${prefix}-burst" "$BURST_DURATION_SEC" "$LOCUST_USERS" "$LOCUST_SPAWN_RATE"
  sleep "$IDLE_DURATION_SEC"
}

build_and_push_image() {
  if [[ "$SKIP_BUILD_PUSH" == "1" ]]; then
    log "Skipping Docker build/push and using existing image: $IMAGE"
    echo "SKIPPED build/push; using existing image $IMAGE" > "$LOGS_DIR/docker_build.log"
    echo "SKIPPED push; existing image assumed available: $IMAGE" > "$LOGS_DIR/docker_push.log"
    return 0
  fi
  local docker_cmd="docker"
  if ! docker info >/dev/null 2>&1; then
    docker_cmd="sudo docker"
  fi
  (cd "$PROJECT_ROOT" && $docker_cmd build -t "$IMAGE" . > "$LOGS_DIR/docker_build.log" 2>&1)
  (cd "$PROJECT_ROOT" && $docker_cmd push "$IMAGE" > "$LOGS_DIR/docker_push.log" 2>&1)
}

deploy_predictive() {
  run_kubectl apply -f "$PROJECT_ROOT/k8s/rbac.yaml" >/dev/null
  run_kubectl apply -f "$PROJECT_ROOT/k8s/autoscaler-deployment.yaml" >/dev/null
  local container_name
  container_name="$(run_kubectl -n "$AUTO_NS" get deployment "$AUTO_DEPLOY" -o jsonpath='{.spec.template.spec.containers[0].name}')"
  run_kubectl -n "$AUTO_NS" set image "deployment/${AUTO_DEPLOY}" "${container_name}=${IMAGE}" >/dev/null
  run_kubectl -n "$AUTO_NS" set env "deployment/${AUTO_DEPLOY}" \
    LOOP_INTERVAL_SECONDS=5 \
    COOLDOWN_SECONDS=8 \
    SCALE_UP_MARGIN=18 \
    BURST_RPS_DELTA_THRESHOLD=2 \
    ABSOLUTE_BURST_RPS_THRESHOLD=10 \
    BURST_SCALE_STEP=3 \
    MAX_SCALE_UP_STEP=5 \
    COOLDOWN_BYPASS_MARGIN=4 \
    SCALE_DOWN_STREAK_REQUIRED=3 \
    CPU_RATE_WINDOW=15s \
    REQUEST_RATE_WINDOW=15s >/dev/null
  run_kubectl -n "$AUTO_NS" rollout restart "deployment/${AUTO_DEPLOY}" >/dev/null
  run_kubectl -n "$AUTO_NS" rollout status "deployment/${AUTO_DEPLOY}" --timeout=180s >/dev/null
}

run_hpa_baseline() {
  log "Running HPA baseline"
  write_hpa_manifest | run_kubectl apply -f - >/dev/null
  run_kubectl -n "$AUTO_NS" scale "deployment/${AUTO_DEPLOY}" --replicas=0 >/dev/null
  run_kubectl -n "$AUTO_NS" scale "deployment/${TARGET_DEPLOY}" --replicas=1 >/dev/null
  run_kubectl -n "$AUTO_NS" rollout status "deployment/${TARGET_DEPLOY}" --timeout=180s >/dev/null

  cleanup_old_locust_jobs
  ensure_locust_configmap
  sample_runtime "$LOGS_DIR/hpa_runtime.csv" "hpa" &
  local sampler_pid=$!
  run_workload "hpa"
  wait "$sampler_pid"
}

extract_predictive_runtime() {
  local pod
  pod="$(run_kubectl -n "$AUTO_NS" get pods -l app=$AUTO_DEPLOY -o jsonpath='{.items[0].metadata.name}')"
  local raw_metrics_csv="$LOGS_DIR/predictive_metrics_raw.csv"
  rm -f "$raw_metrics_csv"
  run_kubectl -n "$AUTO_NS" cp "${pod}:/app/metrics_log.csv" "$raw_metrics_csv" >/dev/null 2>&1 || true
  run_kubectl -n "$AUTO_NS" logs "pod/${pod}" --tail=-1 > "$LOGS_DIR/predictive_autoscaler.log"
  python3 "$PROJECT_ROOT/experiments/extract_predictive_runtime.py" \
    --log "$LOGS_DIR/predictive_autoscaler.log" \
    --out "$LOGS_DIR/predictive_runtime.csv" \
    --metrics-csv "$raw_metrics_csv"
}

run_predictive() {
  log "Running predictive experiment"
  run_kubectl -n "$AUTO_NS" delete hpa "$TARGET_DEPLOY" --ignore-not-found >/dev/null
  run_kubectl -n "$AUTO_NS" scale "deployment/${AUTO_DEPLOY}" --replicas=1 >/dev/null
  run_kubectl -n "$AUTO_NS" rollout restart "deployment/${AUTO_DEPLOY}" >/dev/null
  run_kubectl -n "$AUTO_NS" rollout status "deployment/${AUTO_DEPLOY}" --timeout=180s >/dev/null
  run_kubectl -n "$AUTO_NS" scale "deployment/${TARGET_DEPLOY}" --replicas=1 >/dev/null
  run_kubectl -n "$AUTO_NS" rollout status "deployment/${TARGET_DEPLOY}" --timeout=180s >/dev/null

  cleanup_old_locust_jobs
  ensure_locust_configmap
  run_workload "predictive"
  extract_predictive_runtime
}

validate_outputs() {
  set +e
  python3 "$PROJECT_ROOT/experiments/validate_runtime_plots.py" \
    --predictive "$LOGS_DIR/predictive_runtime.csv" \
    --hpa "$LOGS_DIR/hpa_runtime.csv" \
    --outdir "$GRAPHS_DIR" \
    --predictive-cpu-limit-millicores "$CPU_LIMIT_MILLICORES" \
    --hpa-cpu-limit-millicores "$CPU_LIMIT_MILLICORES" | tee "$SUMMARY_DIR/validation_stdout.txt"
  local validator_status=${PIPESTATUS[0]}
  set -e

  if [[ -f "$GRAPHS_DIR/summary.txt" ]]; then
    cp "$GRAPHS_DIR/summary.txt" "$SUMMARY_DIR/summary.txt"
  else
    cat > "$SUMMARY_DIR/summary.txt" <<EOF
validator_exit_code=${validator_status}
claim_supported=UNKNOWN
telemetry_errors=["validator did not produce summary.txt"]
EOF
  fi

  if [[ $validator_status -ne 0 ]]; then
    log "Validator reported a non-success outcome (exit=${validator_status}). Results marked in summary files."
  fi
}

write_metrics_json() {
  python3 - "$LOGS_DIR/predictive_runtime.csv" "$LOGS_DIR/hpa_runtime.csv" "$METRICS_DIR/metrics.json" <<'PY'
import csv, json, sys
from datetime import datetime

pred_path, hpa_path, out_path = sys.argv[1:]

def parse_ts(v):
    if v.endswith("Z"):
        return datetime.fromisoformat(v.replace("Z","+00:00"))
    return datetime.fromisoformat(v.replace(",", "."))

def load(path):
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return rows

pred = load(pred_path)
hpa = load(hpa_path)

def first_scale(rows):
    if not rows:
      return None
    initial = float(rows[0]["replicas"])
    t0 = parse_ts(rows[0]["timestamp"])
    for row in rows:
      if float(row["replicas"]) > initial:
        return (parse_ts(row["timestamp"]) - t0).total_seconds()
    return None

def max_cpu(rows):
    key = "cpu" if rows and "cpu" in rows[0] else "cpu_millicores"
    return max(float(r[key]) for r in rows) if rows else 0.0

def replica_seconds(rows):
    total = 0.0
    for a, b in zip(rows, rows[1:]):
      dt = (parse_ts(b["timestamp"]) - parse_ts(a["timestamp"])).total_seconds()
      total += float(a["replicas"]) * max(dt, 0.0)
    return total

pred_delay = first_scale(pred)
hpa_delay = first_scale(hpa)
data = {
  "predictive_delay": pred_delay,
  "hpa_delay": hpa_delay,
  "delay_difference": None if pred_delay is None or hpa_delay is None else hpa_delay - pred_delay,
  "max_cpu_hpa": max_cpu(hpa),
  "max_cpu_predictive": max_cpu(pred),
  "replica_seconds_hpa": replica_seconds(hpa),
  "replica_seconds_predictive": replica_seconds(pred),
}
with open(out_path, "w", encoding="utf-8") as fh:
  json.dump(data, fh, indent=2)
PY
}

write_experiment_summary() {
  local claim_status telemetry_status
  claim_status="UNKNOWN"
  telemetry_status="UNKNOWN"
  if grep -q "claim_supported=True" "$SUMMARY_DIR/summary.txt"; then
    claim_status="YES"
  elif grep -q "claim_supported=False" "$SUMMARY_DIR/summary.txt"; then
    claim_status="NO"
  fi
  if grep -q "telemetry_errors=\\[\\]" "$SUMMARY_DIR/summary.txt"; then
    telemetry_status="request-rate present and usable"
  else
    telemetry_status="INVALID RUN"
  fi
  cat > "$SUMMARY_DIR/experiment_summary.txt" <<EOF
Predictive < HPA: ${claim_status}
Telemetry quality: ${telemetry_status}
Key observations:
- Baseline and predictive runs used the same burst/idle workload profile.
- Validation results are recorded in ${SUMMARY_DIR}/summary.txt.
- Metrics are recorded in ${METRICS_DIR}/metrics.json.
Notes for paper writing:
- Use the validator verdict directly; do not claim proactive scaling if claim_supported=False.
- Check predictive_autoscaler.log for first burst timing and request-rate source before writing conclusions.
- If docker push was skipped or failed, results reflect the existing cluster image tag, not an unpublished local build.
EOF
}

main() {
  create_output_structure
  save_configs
  verify_environment
  ensure_target_service
  ensure_http_ready_target
  build_and_push_image
  deploy_predictive
  run_hpa_baseline
  run_predictive
  validate_outputs
  write_metrics_json
  write_experiment_summary
  echo "Experiment completed. All results saved in 'hari' folder. Ready for paper generation."
}

main "$@"
