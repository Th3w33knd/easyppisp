import torch
import time
from easyppisp.functional import apply_pipeline
from easyppisp.params import PipelineParams

def benchmark():
    # Use a realistic image size (H, W, 3)
    img_size = (1, 2048, 2048, 3)
    img = torch.randn(img_size)
    params = PipelineParams.from_constrained() # Identity params
    
    print(f"Benchmarking with image size: {img_size}")

    # 1. CPU Benchmark
    print("\n[CPU] Running benchmark...")
    img_cpu = img.to('cpu')
    
    # Warmup
    _ = apply_pipeline(img_cpu, params)
    
    start = time.time()
    for _ in range(5):
        _ = apply_pipeline(img_cpu, params)
    end = time.time()
    cpu_time = (end - start) / 5
    print(f"CPU Time: {cpu_time:.4f}s per image")

    # 2. GPU Benchmark (PyTorch Fallback)
    if torch.cuda.is_available():
        print(f"\n[GPU: {torch.cuda.get_device_name(0)}] Running benchmark...")
        img_gpu = img.to('cuda')
        
        # Warmup
        _ = apply_pipeline(img_gpu, params)
        torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(20):
            _ = apply_pipeline(img_gpu, params)
        torch.cuda.synchronize()
        end = time.time()
        gpu_time = (end - start) / 20
        print(f"GPU Time: {gpu_time:.4f}s per image")
        print(f"Speedup: {cpu_time / gpu_time:.2f}x")
        
        from easyppisp._internal.cuda_layer import is_cuda_available
        print(f"Optimized CUDA available: {is_cuda_available()}")
    else:
        print("\nCUDA not available for GPU benchmark.")

if __name__ == "__main__":
    benchmark()
