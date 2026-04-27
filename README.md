# AI-Based Predictive Kubernetes Autoscaler (LSTM)

This project implements a production-grade predictive autoscaler for Kubernetes using a pre-trained LSTM model (`offline_lstm_model.h5`) to proactively scale before CPU spikes.

## Project Structure

```text
predictive/
├── app/
│   ├── data_collector.py
│   ├── model_loader.py
│   ├── predictor.py
│   ├── scaler.py
│   ├── evaluator.py
│   └── main.py
├── models/
│   └── offline_lstm_model.h5
├── k8s/
│   ├── rbac.yaml
│   └── autoscaler-deployment.yaml
├── locust/
│   └── locustfile.py
├── requirements.txt
└── README.md
```

## Research Features Implemented

- Metrics collection from Prometheus:
  - CPU usage (%)
  - Memory usage (%)
  - Request rate (RPS)
- LSTM forecasting using existing offline-trained model (`models/offline_lstm_model.h5`)
- Predictive autoscaling policy:
  - Scale up if predicted CPU > 70%
  - Scale down if predicted CPU < 30%
  - Cooldown = 60 seconds
- Kubernetes scaling via `client-python` with in-cluster config
- HPA comparison support (reads HPA desired replicas if present)
- IEEE-style evaluation outputs:
  - RMSE, MAE, MAPE
  - Resource Savings %
  - Latency Reduction %
  - Prediction vs Actual graph
  - CPU usage comparison graph
  - Replica count comparison graph

## Prerequisites

- Kubernetes cluster running
- Prometheus accessible from cluster
- Target deployment exists:
  - Namespace: `autoscale-demo`
  - Deployment: `cpu-demo`
- Python 3.10+ (for local test) or containerized runtime

## 1) Install Dependencies (Local)

```bash
pip install -r requirements.txt
```

## 2) Run Locally (Optional)

```bash
export PROMETHEUS_URL="http://<prometheus-host>:9090"
export NAMESPACE="autoscale-demo"
export DEPLOYMENT_NAME="cpu-demo"
export MODEL_PATH="models/offline_lstm_model.h5"
python app/main.py
```

## 3) Build and Push Container

From this folder:

```bash
cat > Dockerfile <<'EOF'
FROM python:3.11-slim
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY models ./models
CMD ["python", "app/main.py"]
EOF

docker build -t ccbd/predictive-autoscaler:latest .
docker push ccbd/predictive-autoscaler:latest
```

## 4) Deploy to Kubernetes

```bash
kubectl apply -f k8s/rbac.yaml
kubectl apply -f k8s/autoscaler-deployment.yaml
```

## 5) Generate Burst Load Using Locust

```bash
locust -f locust/locustfile.py --host=http://<target-service-url>
```

Suggested experiment setup:
- Run test with HPA-only baseline.
- Run test with predictive autoscaler enabled.
- Compare generated artifacts.

## 6) Evaluation Outputs

On termination (SIGINT/SIGTERM), artifacts are written to `artifacts/`:

- `evaluation_log.csv`
- `metrics.txt`
- `prediction_vs_actual.png`
- `cpu_usage_comparison.png`
- `replica_count_comparison.png`

## Key Environment Variables

- `PROMETHEUS_URL` (required)
- `NAMESPACE` (default `autoscale-demo`)
- `DEPLOYMENT_NAME` (default `cpu-demo`)
- `MODEL_PATH` (default `models/offline_lstm_model.h5`)
- `LOOP_INTERVAL_SECONDS` (default `30`)
- `COOLDOWN_SECONDS` (default `60`)
- `UP_THRESHOLD` (default `70`)
- `DOWN_THRESHOLD` (default `30`)
- `MIN_REPLICAS` (default `1`)
- `MAX_REPLICAS` (default `20`)
- `SCALE_STEP` (default `1`)
- `REQUEST_MAX` (default `2000`, normalization upper bound)
- `AI_LATENCY_MS`, `HPA_LATENCY_MS` (optional for latency reduction metric)

## Notes for Reproducible IEEE-Style Experiments

- Keep same load profile across baseline and predictive runs.
- Capture Prometheus scrape interval and cluster resource limits in your paper.
- Report RMSE/MAE/MAPE with confidence intervals across multiple runs.
- Use identical min/max replica bounds for fair AI vs HPA comparison.
