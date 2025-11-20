import torch
import grouped_gemm as gg

def benchmark(name, func, x, w, batch_sizes, iterations=50):
    print(f"Preparing {name}...")

    for _ in range(10):
        out = func(x, w, batch_sizes, trans_b=True)
        out.sum().backward()
        x.grad = None
        w.grad = None
    torch.cuda.synchronize()

    print(f"Profiling {name}...")
    
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(iterations):
            out = func(x, w, batch_sizes, trans_b=True)
            out.sum().backward()
            x.grad = None
            w.grad = None

    torch.cuda.synchronize()

    total_ms = prof.key_averages().total_average().device_time_total / 1000
    avg_ms = total_ms / iterations
    
    print(f"  -> Total GPU time: {total_ms:.2f} ms")
    print(f"  -> Time per step:  {avg_ms:.3f} ms")
    prof.export_chrome_trace(f'traces/{name}_trace.json')
    return avg_ms

if __name__ == '__main__':
    # --- Configuration ---
    model_config_dict = {
        "Qwen/Qwen3-30B-A3B":{
            "num_experts_per_tok": 8,
            "hidden_size": 2048,
            "num_experts": 128,
            "moe_intermediate_size": 768,
        },
    }
    
    model_name = "Qwen/Qwen3-30B-A3B"
    model_config = model_config_dict[model_name]
    seqlen = 4096
    test_case = "up_proj"

    M = seqlen * model_config["num_experts_per_tok"]
    E = model_config["num_experts"]

    if test_case == "up_proj":
        K = model_config["hidden_size"]
        N = 2 * model_config["moe_intermediate_size"]
    elif test_case == "down_proj":
        K = model_config["moe_intermediate_size"]
        N = model_config["hidden_size"]
    
    print(f"Config: {test_case} | Tokens: {M} | Experts: {E} | Shape: K={K}, N={N}")

    torch.manual_seed(42)
    
    x = torch.rand(M, K, dtype=torch.bfloat16, device='cuda', requires_grad=True)
    w = torch.rand(E, N, K, dtype=torch.bfloat16, device='cuda', requires_grad=True)
    batch_sizes = torch.tensor([M//E]*E, device='cpu')

    time_base = benchmark(
        "Base (Loop)", 
        gg.ops.gmm_base, 
        x, w, batch_sizes
    )
    
    print("-" * 30)

    # 2. cuBLAS (Batched)
    time_cublas = benchmark(
        "cuBLAS (Batched)", 
        gg.ops.gmm_cuBLAS, 
        x, w, batch_sizes
    )

    print("=" * 30)
    print(f"Speedup: {time_base / time_cublas:.2f}x")
    print("=" * 30)