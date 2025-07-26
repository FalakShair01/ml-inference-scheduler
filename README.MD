Here’s a **refined and concise version** of your `README.md` with emphasis on clarity, professionalism, and your points about this being a **prototype** and **latency is simulated** via worker scripts:

---

# Fine-Grained ML Inference Scheduler

![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

## 🚀 Overview

This project is a **prototype** for a distributed ML inference scheduler designed to route requests intelligently across **heterogeneous resources** (simulated GPU and CPU workers). It demonstrates how to optimize inference performance using real-time load tracking and greedy-first scheduling.

The worker services **simulate latency** for GPU-like and CPU-like hardware to mimic real-world behavior — making this ideal for testing smart scheduling logic in constrained environments.

## 🔍 Problem Statement

ML inference in production often suffers from:

* Underutilized fast resources (e.g., GPUs)
* High tail latency due to poor request distribution
* Static load balancing that doesn’t adapt to real-time load

This scheduler addresses these issues by dynamically selecting the **least-loaded, fastest available worker**.

## ✨ Key Features

* **Greedy-first scheduling**: Prioritizes GPU-like workers until capacity is reached, then routes to CPU-like ones.
* **Simulated latency**: Workers emulate GPU (low latency) and CPU (higher latency) response times.
* **Real-time load awareness** via Redis.
* **Prometheus + Grafana** for metrics and dashboards.
* **FastAPI microservices** for modular design.
* **Docker Compose** for easy setup.

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
👉 [Read the Medium article](https://medium.com/@falakshair563/optimizing-ml-inference-building-a-smart-resource-scheduler-for-heterogeneous-workloads-9d5575895c10)

## 🧪 Future Improvements

* Reinforcement Learning-based scheduling
* Priority-aware request handling
* Dynamic worker scaling (e.g., with Kubernetes)
* Resilience and fault tolerance
* Cost-aware scheduling logic

## 👤 Author

**Falak Shair**
[GitHub](https://github.com/falakshair01) | [LinkedIn](https://www.linkedin.com/in/falak-shair-software-engineer/)

---