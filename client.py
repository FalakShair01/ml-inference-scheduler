import httpx
import asyncio
import time
import uuid
import json

SCHEDULER_URL = "http://localhost:8000/infer"
NUM_REQUESTS = 500
CONCURRENCY = 30 # Number of concurrent requests
REQUEST_INTERVAL = 0.08 # seconds between starting each request (for sequential generation)

async def send_request(request_id: str, client: httpx.AsyncClient):
    start_time = time.time()
    payload = {"id": request_id, "data": "sample_inference_input"}
    try:
        response = await client.post(SCHEDULER_URL, json=payload)
        response.raise_for_status()
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        print(f"Request {request_id}: Status {response.status_code}, Latency {latency_ms:.2f}ms, Worker {response.json().get('worker')}")
        return {"id": request_id, "latency_ms": latency_ms, "worker": response.json().get('worker'), "status": "success"}
    except httpx.RequestError as e:
        print(f"Request {request_id}: Failed to connect: {e}")
        return {"id": request_id, "latency_ms": -1, "worker": "N/A", "status": "connection_error"}
    except httpx.HTTPStatusError as e:
        print(f"Request {request_id}: Error {e.response.status_code}: {e.response.text}")
        return {"id": request_id, "latency_ms": -1, "worker": "N/A", "status": "http_error"}
    except Exception as e:
        print(f"Request {request_id}: An unexpected error occurred: {e}")
        return {"id": request_id, "latency_ms": -1, "worker": "N/A", "status": "unknown_error"}

async def main():
    all_results = []
    async with httpx.AsyncClient() as client:
        tasks = []
        for i in range(NUM_REQUESTS):
            request_id = str(uuid.uuid4())[:8]
            tasks.append(asyncio.create_task(send_request(request_id, client)))  # Wrap coroutine as Task

            # Optional: Add a small delay to simulate more realistic request patterns
            await asyncio.sleep(REQUEST_INTERVAL) 

            if len(tasks) >= CONCURRENCY:
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for res in done:
                    all_results.append(await res)
                tasks = list(pending)

        # Wait for any remaining tasks
        if tasks:
            done, _ = await asyncio.wait(tasks)
            for res in done:
                all_results.append(res.result())

    successful_requests = [r for r in all_results if r["status"] == "success"]
    if successful_requests:
        avg_latency = sum(r["latency_ms"] for r in successful_requests) / len(successful_requests)
        print(f"\n--- Summary ---")
        print(f"Total requests sent: {NUM_REQUESTS}")
        print(f"Successful requests: {len(successful_requests)}")
        print(f"Average latency for successful requests: {avg_latency:.2f}ms")
    else:
        print("\n--- Summary ---")
        print("No successful requests made.")

    # Save results to a file for later analysis
    with open("request_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Detailed results saved to request_results.json")

if __name__ == "__main__":

    asyncio.run(main())


