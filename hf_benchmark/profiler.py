import time
import torch
from hf_benchmark.device import get_device

def profile_inference(fn, warmup=3, runs=5):
    device = get_device()
    
    print(f"🔥 Warming up...")
    for _ in range(warmup):
        fn()

    # Sync only if on GPU
    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies = []
    print(f"🚀 Benchmarking ({runs} runs)...")
    
    for _ in range(runs):
        start = time.time()
        fn()
        
        # Sync only if on GPU (Crucial for accurate timing)
        if device.type == "cuda":
            torch.cuda.synchronize()
            
        latencies.append((time.time() - start) * 1000)

    latencies.sort()
    return {
        "p50": latencies[len(latencies)//2],
        "p95": latencies[int(len(latencies)*0.95)],
        "mean": sum(latencies) / len(latencies)
    }