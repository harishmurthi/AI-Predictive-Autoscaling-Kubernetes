#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ccbd/predictive"
IMAGE="hari053/predictive-autoscaler:latest"
AUTO_NS="autoscale-demo"
MON_NS="monitoring"
AUTO_DEPLOY="predictive-autoscaler"
TARGET_DEPLOY="cpu-demo"
TARGET_SERVICE="cpu-demo"
BURST_DURATION_SEC="${BURST_DURATION_SEC:-120}"
IDLE_DURATION_SEC="${IDLE_DURATION_SEC:-120}"
EXPERIMENT_DURATION_SEC="${EXPERIMENT_DURATION_SEC:-$(( (BURST_DURATION_SEC + IDLE_DURATION_SEC) * 2 ))}"
SAMPLE_INTERVAL_SEC="${SAMPLE_INTERVAL_SEC:-5}"
LOCUST_USERS="${LOCUST_USERS:-300}"
LOCUST_SPAWN_RATE="${LOCUST_SPAWN_RATE:-120}"

RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
BASE_OUT_DIR="$PROJECT_ROOT/experiments/output/$RUN_TS"
OUT_DIR="$BASE_OUT_DIR/attempt1"
mkdir -p "$OUT_DIR"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
fail() { echo "ERROR: $*" >&2; exit 1; }

run_kubectl() {
  kubectl "$@"
}

if docker info >/dev/null 2>&1; then
  DOCKER="docker"
else
  DOCKER="sudo docker"
fi

require_file() {
  local path="$1"
  [[ -f "$path" ]] || fail "Missing required file: $path"
}

require_dir() {
  local path="$1"
  [[ -d "$path" ]] || fail "Missing required directory: $path"
}

collect_env_info() {
  {
    echo "=== Environment ==="
    uname -a
    echo
    python3 --version
    docker --version || true
    kubectl version --client
    echo
    echo "=== Cluster Context ==="
    kubectl config current-context
    kubectl get nodes -o wide
    echo
    echo "=== Workload Config ==="
    echo "BURST_DURATION_SEC=$BURST_DURATION_SEC"
    echo "IDLE_DURATION_SEC=$IDLE_DURATION_SEC"
    echo "EXPERIMENT_DURATION_SEC=$EXPERIMENT_DURATION_SEC"
    echo "SAMPLE_INTERVAL_SEC=$SAMPLE_INTERVAL_SEC"
    echo "LOCUST_USERS=$LOCUST_USERS"
    echo "LOCUST_SPAWN_RATE=$LOCUST_SPAWN_RATE"
  } | tee "$OUT_DIR/reproducibility.txt"
}

validate_structure() {
  log "Phase 1: validating project structure"
  require_dir "$PROJECT_ROOT/app"
  require_dir "$PROJECT_ROOT/models"
  require_dir "$PROJECT_ROOT/k8s"
  require_dir "$PROJECT_ROOT/locust"
  require_file "$PROJECT_ROOT/models/offline_lstm_model.h5"
  require_file "$PROJECT_ROOT/requirements.txt"
  require_file "$PROJECT_ROOT/Dockerfile"
  require_file "$PROJECT_ROOT/locust/locustfile.py"
  require_file "$PROJECT_ROOT/k8s/rbac.yaml"
  require_file "$PROJECT_ROOT/k8s/autoscaler-deployment.yaml"
  log "Structure validation passed"
}

build_and_push() {
  log "Phase 2: build and push Docker image"
  cd "$PROJECT_ROOT"
  $DOCKER build -t "$IMAGE" . | tee "$OUT_DIR/docker_build.log"

  local push_log="$OUT_DIR/docker_push.log"
  $DOCKER push "$IMAGE" | tee "$push_log"

  IMAGE_DIGEST="$(grep -oE 'sha256:[a-f0-9]{64}' "$push_log" | tail -1 || true)"
  [[ -n "$IMAGE_DIGEST" ]] || fail "Image digest not found in push output"
  log "Image pushed with digest: $IMAGE_DIGEST"
}

ensure_target_workload() {
  log "Ensuring target workload and service exist"
  run_kubectl -n "$AUTO_NS" get deployment "$TARGET_DEPLOY" >/dev/null || fail "Missing deployment $TARGET_DEPLOY in $AUTO_NS"

  cat <<YAML | run_kubectl apply -f -
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

  run_kubectl -n "$MON_NS" get svc prometheus-server >/dev/null || fail "prometheus-server service missing in $MON_NS"
}

deploy_predictive() {
  log "Phase 3: deploying predictive autoscaler"
  run_kubectl apply -f "$PROJECT_ROOT/k8s/rbac.yaml" >/dev/null
  run_kubectl apply -f "$PROJECT_ROOT/k8s/autoscaler-deployment.yaml" >/dev/null
  local container_name
  container_name="$(run_kubectl -n "$AUTO_NS" get deployment "$AUTO_DEPLOY" -o jsonpath='{.spec.template.spec.containers[0].name}')"
  [[ -n "$container_name" ]] || fail "Failed to detect autoscaler container name"
  run_kubectl -n "$AUTO_NS" set image deployment/$AUTO_DEPLOY "$container_name=$IMAGE" >/dev/null
  run_kubectl -n "$AUTO_NS" patch deployment "$AUTO_DEPLOY" --type='json' -p='[{"op":"replace","path":"/spec/template/spec/containers/0/imagePullPolicy","value":"Always"}]' >/dev/null
  run_kubectl -n "$AUTO_NS" rollout restart deployment/$AUTO_DEPLOY >/dev/null
  run_kubectl -n "$AUTO_NS" rollout status deployment/$AUTO_DEPLOY --timeout=180s >/dev/null

  local pod
  pod="$(run_kubectl -n "$AUTO_NS" get pods -l app=$AUTO_DEPLOY -o jsonpath='{.items[0].metadata.name}')"
  [[ -n "$pod" ]] || fail "No pod found for $AUTO_DEPLOY"

  local ready
  ready="$(run_kubectl -n "$AUTO_NS" get pod "$pod" -o jsonpath='{.status.containerStatuses[0].ready}')"
  [[ "$ready" == "true" ]] || fail "Autoscaler pod not ready"

  local deploy_image
  deploy_image="$(run_kubectl -n "$AUTO_NS" get deployment "$AUTO_DEPLOY" -o jsonpath='{.spec.template.spec.containers[0].image}')"
  [[ "$deploy_image" == "$IMAGE" ]] || fail "Deployment image mismatch: expected $IMAGE got $deploy_image"

  local logs found=0
  for _ in $(seq 1 12); do
    logs="$(run_kubectl -n "$AUTO_NS" logs deployment/$AUTO_DEPLOY --tail=400 || true)"
    echo "$logs" > "$OUT_DIR/predictive_deploy_logs.txt"
    if echo "$logs" | grep -q "Loaded model"; then
      found=1
      break
    fi
    sleep 10
  done
  [[ "$found" -eq 1 ]] || fail "Model load success log missing"
  echo "$logs" | grep -q "Feature mismatch" && fail "Feature mismatch found in logs"
  echo "$logs" | grep -q "Traceback (most recent call last)" && fail "Fatal traceback found in logs"

  log "Predictive deployment validation passed"
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

sample_metrics() {
  local outfile="$1"
  local mode="$2"

  echo "timestamp,unix_ts,mode,replicas,ready_replicas,cpu_millicores,pod_count" > "$outfile"
  local start
  start="$(date +%s)"
  while true; do
    local now elapsed ts replicas ready top_output cpu_m pod_count
    now="$(date +%s)"
    elapsed=$((now - start))
    (( elapsed > EXPERIMENT_DURATION_SEC )) && break

    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    replicas="$(run_kubectl -n "$AUTO_NS" get deployment "$TARGET_DEPLOY" -o jsonpath='{.spec.replicas}' 2>/dev/null || echo 0)"
    ready="$(run_kubectl -n "$AUTO_NS" get deployment "$TARGET_DEPLOY" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo 0)"

    top_output="$(run_kubectl top pods -n "$AUTO_NS" -l app="$TARGET_DEPLOY" --no-headers 2>/dev/null || true)"
    if [[ -z "$top_output" ]]; then
      cpu_m=0
      pod_count=0
    else
      cpu_m="$(echo "$top_output" | awk '
        function to_m(v){
          if (v ~ /m$/) { sub(/m$/, "", v); return v + 0 }
          return (v + 0) * 1000
        }
        {sum += to_m($2); count += 1}
        END {printf("%.0f", sum)}
      ')"
      pod_count="$(echo "$top_output" | wc -l | tr -d ' ')"
    fi

    echo "$ts,$now,$mode,$replicas,$ready,$cpu_m,$pod_count" >> "$outfile"
    sleep "$SAMPLE_INTERVAL_SEC"
  done
}

run_locust_job() {
  local mode="$1"
  local duration_sec="${2}"
  local users="${3}"
  local spawn_rate="${4}"
  local job_name="locust-${mode}-$(date +%s)"

  cat <<YAML | run_kubectl apply -f -
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
        command: ["/bin/sh", "-c"]
        args:
          - |
            locust -f /mnt/locust/locustfile.py \
              --host=http://${TARGET_SERVICE}.${AUTO_NS}.svc.cluster.local \
              --headless \
              -u ${users} \
              -r ${spawn_rate} \
              -t ${duration_sec}s \
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

  run_kubectl -n "$AUTO_NS" wait --for=condition=complete job/$job_name --timeout=$((duration_sec + 300))s >/dev/null
  run_kubectl -n "$AUTO_NS" logs job/$job_name > "$OUT_DIR/${mode}_locust.log" || true
}

run_burst_pattern_workload() {
  local mode="$1"
  local users="$2"
  local spawn_rate="$3"
  local burst_sec="$4"
  local idle_sec="$5"

  log "Running burst workload phase 1 (${burst_sec}s, users=${users}, rate=${spawn_rate})"
  run_locust_job "${mode}-burst1" "$burst_sec" "$users" "$spawn_rate"
  log "Idle cooldown phase 1 (${idle_sec}s)"
  sleep "$idle_sec"

  log "Running burst workload phase 2 (${burst_sec}s, users=${users}, rate=${spawn_rate})"
  run_locust_job "${mode}-burst2" "$burst_sec" "$users" "$spawn_rate"
  log "Idle cooldown phase 2 (${idle_sec}s)"
  sleep "$idle_sec"
}

run_hpa_experiment() {
  log "Phase 4: HPA baseline experiment"
  write_hpa_manifest | run_kubectl apply -f - >/dev/null
  run_kubectl -n "$AUTO_NS" scale deployment "$AUTO_DEPLOY" --replicas=0 >/dev/null
  run_kubectl -n "$AUTO_NS" scale deployment "$TARGET_DEPLOY" --replicas=1 >/dev/null
  run_kubectl -n "$AUTO_NS" rollout status deployment/$TARGET_DEPLOY --timeout=180s >/dev/null

  cleanup_old_locust_jobs
  ensure_locust_configmap
  sample_metrics "$PROJECT_ROOT/hpa_results.csv" "hpa" &
  local sampler_pid=$!
  run_burst_pattern_workload "hpa" "$LOCUST_USERS" "$LOCUST_SPAWN_RATE" "$BURST_DURATION_SEC" "$IDLE_DURATION_SEC"
  wait "$sampler_pid"

  [[ -s "$PROJECT_ROOT/hpa_results.csv" ]] || fail "hpa_results.csv not generated"
  log "HPA experiment completed"
}

run_predictive_experiment() {
  log "Phase 5: Predictive experiment"
  run_kubectl -n "$AUTO_NS" delete hpa "$TARGET_DEPLOY" --ignore-not-found >/dev/null
  run_kubectl -n "$AUTO_NS" set env deployment/$AUTO_DEPLOY \
    LOOP_INTERVAL_SECONDS=10 \
    COOLDOWN_SECONDS=15 \
    UP_THRESHOLD=65 \
    DOWN_THRESHOLD=35 >/dev/null
  run_kubectl -n "$AUTO_NS" scale deployment "$AUTO_DEPLOY" --replicas=1 >/dev/null
  run_kubectl -n "$AUTO_NS" rollout restart deployment/$AUTO_DEPLOY >/dev/null
  run_kubectl -n "$AUTO_NS" rollout status deployment/$AUTO_DEPLOY --timeout=180s >/dev/null
  run_kubectl -n "$AUTO_NS" scale deployment "$TARGET_DEPLOY" --replicas=1 >/dev/null
  run_kubectl -n "$AUTO_NS" rollout status deployment/$TARGET_DEPLOY --timeout=180s >/dev/null

  cleanup_old_locust_jobs
  ensure_locust_configmap
  sample_metrics "$PROJECT_ROOT/predictive_results.csv" "predictive" &
  local sampler_pid=$!
  run_burst_pattern_workload "predictive" "$LOCUST_USERS" "$LOCUST_SPAWN_RATE" "$BURST_DURATION_SEC" "$IDLE_DURATION_SEC"
  wait "$sampler_pid"

  local pod
  pod="$(run_kubectl -n "$AUTO_NS" get pods -l app=$AUTO_DEPLOY -o jsonpath='{.items[0].metadata.name}')"
  run_kubectl cp "$AUTO_NS/$pod:/app/metrics_log.csv" "$PROJECT_ROOT/predictive.csv" >/dev/null
  [[ -s "$PROJECT_ROOT/predictive.csv" ]] || fail "predictive.csv not generated"
  log "Predictive experiment completed"
}

generate_meta() {
  cat > "$OUT_DIR/meta.json" <<EOF
{
  "image": "$IMAGE",
  "image_digest": "$IMAGE_DIGEST",
    "workload_config": {
    "burst_duration_sec": $BURST_DURATION_SEC,
    "idle_duration_sec": $IDLE_DURATION_SEC,
    "experiment_duration_sec": $EXPERIMENT_DURATION_SEC,
    "sample_interval_sec": $SAMPLE_INTERVAL_SEC,
    "locust_users": $LOCUST_USERS,
    "locust_spawn_rate": $LOCUST_SPAWN_RATE,
    "locustfile": "locust/locustfile.py"
  }
}
EOF
}

evaluate_and_graph() {
  log "Phase 6/7/8/9: evaluation, plots, reproducibility, research summary"
  collect_env_info
  generate_meta

python3 - <<'PY'
import importlib
import subprocess
import sys

required = ["numpy", "matplotlib"]
missing = []
for pkg in required:
    try:
        importlib.import_module(pkg)
    except Exception:
        missing.append(pkg)

if missing:
    print(f"Installing missing Python packages for evaluation: {missing}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", *missing])
PY

  python3 "$PROJECT_ROOT/experiments/evaluate.py" \
    "$PROJECT_ROOT/hpa_results.csv" \
    "$PROJECT_ROOT/predictive_results.csv" \
    "$PROJECT_ROOT/predictive.csv" \
    "$OUT_DIR" \
    "$OUT_DIR/meta.json" | tee "$OUT_DIR/research_summary.txt"

  cp "$PROJECT_ROOT/hpa_results.csv" "$OUT_DIR/hpa_results.csv"
  cp "$PROJECT_ROOT/predictive_results.csv" "$OUT_DIR/predictive_results.csv"
  cp "$PROJECT_ROOT/predictive.csv" "$OUT_DIR/predictive.csv"

  log "Pipeline completed successfully"
  log "Outputs: $OUT_DIR"
}

is_scientifically_valid() {
  python3 - "$OUT_DIR/summary.json" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as fp:
    data = json.load(fp)
print("true" if data.get("scientifically_valid") else "false")
PY
}

increase_load_profile() {
  LOCUST_USERS=$((LOCUST_USERS * 2))
  LOCUST_SPAWN_RATE=$((LOCUST_SPAWN_RATE * 2))
  BURST_DURATION_SEC=$((BURST_DURATION_SEC + 60))
  IDLE_DURATION_SEC=$((IDLE_DURATION_SEC + 60))
  EXPERIMENT_DURATION_SEC=$(( (BURST_DURATION_SEC + IDLE_DURATION_SEC) * 2 ))
  log "Load profile increased: users=$LOCUST_USERS spawn_rate=$LOCUST_SPAWN_RATE burst=${BURST_DURATION_SEC}s idle=${IDLE_DURATION_SEC}s"
}

main() {
  validate_structure
  build_and_push
  ensure_target_workload
  deploy_predictive

  local attempt=1
  local max_attempts=3
  while (( attempt <= max_attempts )); do
    OUT_DIR="$BASE_OUT_DIR/attempt${attempt}"
    mkdir -p "$OUT_DIR"
    log "Starting experiment attempt ${attempt}/${max_attempts} (output=$OUT_DIR)"

    run_hpa_experiment
    run_predictive_experiment
    evaluate_and_graph

    if [[ "$(is_scientifically_valid)" == "true" ]]; then
      log "Scientific validity achieved on attempt ${attempt}"
      return 0
    fi

    if (( attempt == max_attempts )); then
      fail "Scientific validity remained NO after ${max_attempts} attempts"
    fi

    log "Scientific validity not achieved; increasing load and retrying"
    increase_load_profile
    attempt=$((attempt + 1))
  done
}

main "$@"
