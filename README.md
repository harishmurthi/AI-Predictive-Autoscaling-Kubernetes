# AI-Based Predictive Autoscaling for Kubernetes Workloads

## 📌 Overview
This project implements an intelligent autoscaling system using LSTM-based time series prediction and Prometheus metrics.

Traditional Kubernetes Horizontal Pod Autoscaler (HPA) reacts to current load, whereas this system predicts future workload and scales proactively, improving performance and resource utilization.

---

## 🧠 System Architecture

![Architecture](architecture_clean.png)

---

## ⚙️ Technologies Used
- Python  
- Kubernetes  
- Prometheus  
- LSTM (Deep Learning)  
- Docker  

---

## 🚀 Key Features
- Predictive autoscaling using LSTM  
- Burst-aware scaling mechanism  
- Real-time monitoring using Prometheus  
- Automated pipeline execution  
- Comparison with traditional HPA  

---

## 🔄 Workflow
1. Prometheus collects real-time system metrics  
2. Metrics are preprocessed and fed into the LSTM model  
3. LSTM predicts future CPU usage  
4. Autoscaler adjusts the number of pods proactively  
5. System performance is evaluated and visualized  

---

## 📊 Results

### HPA vs Predictive Scaling
![Scaling](outputs/graphs/replica_comparison.png)

### CPU Prediction vs Actual
![CPU](outputs/graphs/prediction_vs_actual.png)

### Resource Usage Comparison
![Resource](outputs/graphs/cpu_comparison.png)

---

## ▶️ How to Run

```bash
bash experiments/run_hari_pipeline.sh
