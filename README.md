\# AI-Based Predictive Autoscaling for Kubernetes Workloads



\## 📌 Overview

This project implements an intelligent autoscaling system using LSTM-based time series prediction and Prometheus metrics.



Traditional Kubernetes Horizontal Pod Autoscaler (HPA) reacts to current load, whereas this system predicts future workload and scales proactively, improving performance and resource utilization.



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

\- Burst-aware scaling mechanism  

\- Real-time monitoring using Prometheus  

\- Automated pipeline execution  

\- Comparison with traditional HPA  



\---



\## 🔄 Workflow

1\. Prometheus collects real-time system metrics  

2\. Metrics are preprocessed and fed into the LSTM model  

3\. LSTM predicts future CPU usage  

4\. Autoscaler adjusts the number of pods proactively  

5\. System performance is evaluated and visualized  



\---



\## 📊 Results



\### HPA vs Predictive Scaling

!\[Scaling](./outputs/graphs/cpu_comparison.png)



\### CPU Prediction vs Actual

!\[CPU](https://github.com/harishmurthi/AI-Predictive-Autoscaling-Kubernetes/blob/main/prediction\_vs\_actual.png?raw=true)



\### Resource Usage Comparison

!\[Resource](https://lh3.googleusercontent.com/rd-d/ALs6j_FYRRC964rbxD2rzadYNgOfuqKYvWP9BL_bTcQzcElu9xVPP-Mpo05kiai7NBLjsiW5fSKtyLkRa_QLwJV4bk9JVl_4tdk6hJa7O_FWWynqlvM8fs2RIKuPG9LJXNVr83_Wu0HdXaY1suMD5Soys4Uj-zEOBq4NwA7aOWUAR4uTnyyge4pfovnLG-Sv74lrlqB7ZoqyJyBf-MHr8qm1R3DxaVUQJB16uGoUSNI2fX3V3yjJUkHMU4lYumLJBTjI58PvucChQDU7KVhnVY-411W1nZP6QWGVNESEPypfM3fcJvy76_hXIIkORgktG19smyT40MQXfmOgBGVJ89TZYAOoNPjKZrtEOtWa6Wyz8gOEHiePJG6uGGPRSnELco6wox5kHF6r6-UDHoqDEmmoBKkRSHFVilRUaw-P3q3mk5flb7OEht1QJz8NKjghDtbyucFBtnl9XGBS-k_uozHVGnZH8FtYt6cXO-Xx72g1GPd5BWg7QMTpqeC9lkUD6RSRcikWhlvLTHF7O2L6wxOtu4sSqTjMLlITUD2x9a_MgdHg8wZKi3WX_W4CT1OkzcDYU15pCNcHUmjIQdfW4prBy_OopyqLxgygoF2dKbSk0dPhGW_rOIbCl-ZoP6cUBWFnyKIIQtUKrS53D4LzvZeSnYQPBBneHdz33kdtFObmTSMxOWu0Y5No62ePt0K4uSQwsZNgS6eUdw_4bmToZlKX5pxGAncyDpWj6zgzz762ku56OWiTKTHWzVUeJ9txfCN9--SCZjXypl4Hi5f1fkffHUF7a3mIEYxHQE5YDwFr6nHILqTe9UxTQ2jz_Lw0s06-0SA-6xDyaPkY2w-cIcKopZNSOWDYYGnzRpuWDbI_pDg0cqVlC-JnCnljcSTW5wm0NmqJcLJsz2ZG42OqjBhj2wrPcRtvGO6ZcO8LoLXxwwrN32XMM4W4_lrWL9Q7DU--D-gFrbC9y4gsCjj34Bt1_MTmHGJWY6FHVNGMuPT1DgulXBUhaUKEhDN07U5NZ3xdLmA7w48uNAItFEpkpLiyYg4Lc6Nlq1yp5-O9_UmPaWUc3eqnG3bf9l0W6AXnocFzyNHkx_u0eBUfmW94zd4VM42_INll0lk=w1920-h868?auditContext=prefetch)



\---



\## ▶️ How to Run



```bash

bash experiments/run\_hari\_pipeline.sh

