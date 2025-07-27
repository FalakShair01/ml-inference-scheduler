
---

# Fine-Grained ML Inference Scheduler

![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

## 🌟 Project Overview

This project presents a distributed, fine-grained resource scheduler designed to optimize the serving of Machine Learning inference requests on heterogeneous hardware environments (e.g., a mix of GPU-like and CPU-like compute resources). In real-world ML deployments, efficiently utilizing diverse hardware to minimize latency and maximize throughput under dynamic loads is a significant challenge.

This system addresses these challenges by intelligently routing incoming inference requests to the most suitable and least-loaded worker in real-time. It showcases a robust architecture leveraging modern microservices patterns, centralized state management, and comprehensive observability.

## 💡 Problem Statement

Inference serving for large-scale ML models often faces bottlenecks due to:
* **Heterogeneous Hardware:** Balancing workloads across machines with varying processing capabilities (e.g., high-performance GPUs vs. general-purpose CPUs).
* **Dynamic Workloads:** Handling unpredictable spikes and troughs in request volume.
* **Latency & Throughput Requirements:** Ensuring low response times and high request processing rates.
* **Resource Underutilization:** Static load balancing often leaves expensive, high-capacity resources idle while queues build up elsewhere.

## ✨ Key Features & Solution Highlights

* **Intelligent Scheduling:** Implements a greedy-first algorithm prioritizing faster (simulated GPU) workers and dynamically routing requests based on real-time load.
* **Distributed Architecture:** Built as a set of FastAPI microservices (Scheduler, Workers) for scalability and modularity.
* **Real-time State Management:** Leverages Redis for centralizing worker registration, load tracking, and heartbeat monitoring, ensuring up-to-date cluster insights.
* **Heterogeneous Worker Simulation:** Workers are configured to simulate different inference latencies, mimicking real-world hardware diversity.
* **Comprehensive Observability:** Integrated with Prometheus for metric collection and Grafana for real-time visualization of key performance indicators like latency, throughput, and worker utilization.
* **Containerized Deployment:** All services are containerized with Docker and orchestrated using Docker Compose for easy setup and reproducibility.

## 🧱 Architecture Overview

```
[Client] → [Scheduler] → [GPU Worker] / [CPU Worker]
                        ↕
                     [Redis]
                       ↕
             [Prometheus & Grafana]
```

## ⚙️ Running Locally

### Prerequisites

* Docker & Docker Compose
* Python 3.9+ (for the test client)

### Setup

```bash
git clone https://github.com/your-username/ml-inference-scheduler.git
cd ml-inference-scheduler
docker-compose up --build -d
```

Verify services:

```bash
docker-compose ps
```

Access:

* Grafana: [http://localhost:3000](http://localhost:3000) (`admin`/`admin`)
* Prometheus: [http://localhost:9090](http://localhost:9090)

### Run Test Client

Install deps and run:

```bash
python client.py
```

The `client.py` script bursts inference requests to observe scheduling behavior under load.

## 📈 Metrics & Results

Performance metrics (e.g., **request latency**, **worker utilization**) are tracked and visualized in Grafana.

For detailed graphs and analysis, check out the blog post:
👉 [Read the Medium article](https://medium.com/p/9d5575895c10)

## 🧪 Future Improvements

This project serves as a robust foundation. Potential areas for future exploration include:

- More Sophisticated Scheduling Algorithms: Implementing Reinforcement Learning (RL) or other advanced heuristics for dynamic, adaptive scheduling policies.
- Dynamic Worker Scaling: Integrating with cloud providers (e.g., AWS ECS/EKS, Kubernetes) for automated horizontal scaling of worker instances based on load.
- Request Prioritization: Adding support for high-priority requests to bypass queues or be routed to dedicated resources.
- Fault Tolerance & Resilience: Enhancing error handling and recovery mechanisms for worker failures.
- Cost Optimization: Incorporating cost models into scheduling decisions to minimize operational expenses.

## 👤 Author

**Falak Shair**
[GitHub](https://github.com/falakshair01) | [LinkedIn](https://www.linkedin.com/in/falak-shair-software-engineer/)

---
