from fastapi import FastAPI, HTTPException
import uvicorn
import time
import random
import httpx
from pydantic import BaseModel
import redis # Add this
import os
import json
import asyncio
from contextlib import asynccontextmanager
from prometheus_client import start_http_server, Counter, Histogram, Gauge 


# Initialize Redis client
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
redis_client = redis.Redis(host=REDIS_HOST, port=6379, db=0, decode_responses=True)

# PROMETHEUS
REQUEST_COUNT = Counter('scheduler_requests_total', 'Total number of requests received by the scheduler.')
REQUEST_LATENCY = Histogram('scheduler_request_latency_seconds', 'Latency of requests through the scheduler.', buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0))
WORKER_LOAD_GAUGE = Gauge('worker_current_load', 'Current load on a worker.', ['worker_id', 'worker_type'])
ACTIVE_WORKERS_GAUGE = Gauge('active_workers_count', 'Number of active workers by type.', ['worker_type'])
SCHEDULER_QUEUE_DEPTH = Gauge('scheduler_queue_depth', 'Number of requests currently in scheduler queue.')

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Scheduler starting up...")
    start_http_server(9001)
    yield

app = FastAPI(lifespan=lifespan,title="ML Inference Scheduler")

@app.get("/")
async def root():
    return {"message": "Scheduler service running"}

@app.post("/infer")
async def schedule_inference(request_data: dict):
    REQUEST_COUNT.inc() # Increment total requests
    SCHEDULER_QUEUE_DEPTH.inc() # Simulate request entering queue
    start_time = time.time()

    chosen_worker_name = None

    # Retrieve all registered workers
    registered_workers = redis_client.hgetall("worker_registry") # Returns dict of {name: json_string}

    active_gpu_workers = []
    active_cpu_workers = []

    for name, data_json in registered_workers.items():
        try:
            info = json.loads(data_json)
            current_load = int(redis_client.get(f"worker_load:{name}") or 0) # Get current load from Redis
            WORKER_LOAD_GAUGE.labels(worker_id=name, worker_type=info["type"]).set(current_load)
            if info["type"] == "GPU":
                active_gpu_workers.append({"name": name, "load": current_load, "sim_latency": info["sim_latency"]})
            elif info["type"] == "CPU":
                active_cpu_workers.append({"name": name, "load": current_load, "sim_latency": info["sim_latency"]})
        except Exception as e:
            print(f"Error parsing worker info from Redis for {name}: {e}")

    # Simple Greedy Scheduling: Prioritize GPU, then CPU, least loaded first
    active_gpu_workers.sort(key=lambda x: x["load"])
    active_cpu_workers.sort(key=lambda x: x["load"])

    if active_gpu_workers and active_gpu_workers[0]["load"] < 2: # Assuming capacity 2 for now
        chosen_worker_name = active_gpu_workers[0]["name"]
    elif active_cpu_workers and active_cpu_workers[0]["load"] < 2:
        chosen_worker_name = active_cpu_workers[0]["name"]

    if not chosen_worker_name:
        raise HTTPException(status_code=503, detail="No workers available to process the request.")

    # Increment load in Redis immediately
    redis_client.incr(f"worker_load:{chosen_worker_name}")


    worker_address = f"http://{chosen_worker_name}:8001" # Using Docker service name

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(worker_address + "/process", json=request_data, timeout=10.0)
            response.raise_for_status()
            worker_response = response.json()
            return {
                "status": "scheduled and processed",
                "worker": chosen_worker_name,
                "request_id": request_data.get("id"),
                "worker_response": worker_response
            }
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Worker communication error: {e}")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Worker responded with error: {e.response.text}")
    finally:
        # Decrement load in Redis after response (or error)
        redis_client.decr(f"worker_load:{chosen_worker_name}")
        SCHEDULER_QUEUE_DEPTH.dec() # Simulate request leaving queue
        latency = time.time() - start_time
        REQUEST_LATENCY.observe(latency) 

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
