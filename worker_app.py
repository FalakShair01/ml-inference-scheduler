from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import time
import os
import redis
import os
import json
import asyncio
from contextlib import asynccontextmanager
from prometheus_client import start_http_server, Counter, Histogram, Gauge

worker_type = os.getenv("WORKER_TYPE", "UNKNOWN")
sim_latency = float(os.getenv("SIM_LATENCY", "0.1"))
worker_id = os.getenv("SERVER", f"{worker_type}_default") 

WORKER_REQUESTS_TOTAL = Counter('worker_requests_total', 'Total requests processed by worker.', ['worker_id', 'worker_type'])
WORKER_PROCESSING_TIME = Histogram('worker_processing_time_seconds', 'Time taken by worker to process a request.', ['worker_id', 'worker_type'], buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0))
WORKER_CURRENT_LOAD = Gauge('worker_current_load', 'Current concurrent requests being processed by worker.', ['worker_id', 'worker_type'])


async def register_worker():
    worker_data = {
        "type": worker_type,
        "sim_latency": sim_latency
    }
    redis_client.hset("worker_registry", worker_id, json.dumps(worker_data))
    redis_client.setnx(f"worker_load:{worker_id}", 0)
    print(f"Worker {worker_id} registered in Redis.")

async def heartbeat():
    while True:
        redis_client.set(f"worker_heartbeat:{worker_id}", int(time.time()), ex=10) # Expires in 10 seconds
        await asyncio.sleep(5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_http_server(9000)
    print(f"Prometheus metrics server started on port 8001 for {worker_id}")
    await register_worker()
    yield

app = FastAPI(lifespan=lifespan)

# Initialize Redis client
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
redis_client = redis.Redis(host=REDIS_HOST, port=6379, db=0, decode_responses=True)

print(f"Starting {worker_type} worker: {worker_id} with simulated latency: {sim_latency*1000}ms")

class RequestData(BaseModel):
    id: str
    data: str

@app.get("/")
async def root():
    return {"message": f"Worker {worker_id} ({worker_type}) running"}


@app.post("/process")
async def process_inference(request_data: RequestData):
    WORKER_CURRENT_LOAD.labels(worker_id=worker_id, worker_type=worker_type).inc() # Increment concurrent load
    start_time = time.time()
    print(f"Worker {worker_id} ({worker_type}) processing request {request_data.id}")

    # Simulate work
    time.sleep(sim_latency)

    end_time = time.time()
    processing_time = end_time - start_time # In seconds

    WORKER_CURRENT_LOAD.labels(worker_id=worker_id, worker_type=worker_type).dec() # Decrement concurrent load
    WORKER_REQUESTS_TOTAL.labels(worker_id=worker_id, worker_type=worker_type).inc() # Increment total requests
    WORKER_PROCESSING_TIME.labels(worker_id=worker_id, worker_type=worker_type).observe(processing_time) # Observe processing time

    return {
        "status": "processed",
        "worker_id": worker_id,
        "worker_type": worker_type,
        "request_id": request_data.id,
        "processing_time_ms": processing_time * 1000
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)