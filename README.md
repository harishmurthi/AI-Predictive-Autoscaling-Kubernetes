\# AI-Based Predictive Autoscaling for Kubernetes Workloads



\## 📌 Overview

This project implements an intelligent autoscaling system using LSTM-based time series prediction and Prometheus metrics.



Traditional Kubernetes Horizontal Pod Autoscaler (HPA) reacts to current load, whereas this system predicts future workload and scales proactively.



\---



\## 🧠 System Architecture



!\[Architecture](architecture\_clean.png)



\---



\## ⚙️ Technologies Used

\- Python

\- Kubernetes

\- Prometheus

\- LSTM (Deep Learning)

\- Docker



\---



\## 🚀 Key Features

\- Predictive autoscaling using LSTM

\- Burst-aware scaling

\- Real-time monitoring using Prometheus

\- Automated pipeline execution



\---



\## 🔄 Workflow

1\. Prometheus collects real-time metrics  

2\. Metrics are processed and fed to LSTM model  

3\. LSTM predicts future CPU usage  

4\. Autoscaler adjusts replicas proactively  



\---



\## 📊 Results



\### CPU Prediction vs Actual

!\[CPU](outputs/graphs/prediction\_vs\_actual.png)



\### HPA vs Predictive Scaling

!\[Scaling](outputs/graphs/replica\_comparison.png)



\### Resource Usage Comparison

!\[Resource](outputs/graphs/cpu\_comparison.png)



\---



\## ▶️ How to Run

bash experiments/run\_hari\_pipeline.sh



\---



\## 👨‍💻 Author

HARISH (PES1PG24CS025)

