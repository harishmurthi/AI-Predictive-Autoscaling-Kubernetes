#!/usr/bin/env bash
set -euo pipefail

IMAGE="hari053/predictive-autoscaler:latest"
NAMESPACE="autoscale-demo"
APP_LABEL="predictive-autoscaler"

echo "[1/5] Building Docker image: ${IMAGE}"
docker build -t "${IMAGE}" .

echo "[2/5] Pushing Docker image to Docker Hub: ${IMAGE}"
docker push "${IMAGE}"

echo "[3/5] Applying Kubernetes manifests"
kubectl apply -f k8s/rbac.yaml
kubectl apply -f k8s/autoscaler-deployment.yaml

echo "[4/5] Waiting for rollout"
kubectl -n "${NAMESPACE}" rollout status deployment/predictive-autoscaler --timeout=180s

echo "[5/5] Current pod status"
kubectl -n "${NAMESPACE}" get pods -l app="${APP_LABEL}" -o wide

echo "Streaming logs from deployment/predictive-autoscaler"
kubectl -n "${NAMESPACE}" logs -f deployment/predictive-autoscaler --tail=200
