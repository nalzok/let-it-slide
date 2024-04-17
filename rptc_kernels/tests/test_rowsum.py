import torch
import rptc_kernels


def benchmark():
    torch.manual_seed(42)

    m = 4096
    n = 4096
    memory_consumption = m * n * 2
    print(f"{m = }, {n = }, {memory_consumption = }")

    decompressed = torch.randn((m, n), dtype=torch.float16, device="cuda")

    # warm up
    throwaway = torch.empty((m,), dtype=torch.float16, device="cuda")
    _ = rptc_kernels.rowsum(decompressed, throwaway)

    out = torch.zeros((m,), dtype=torch.float16, device="cuda")
    elapsed_time = rptc_kernels.rowsum(decompressed, out) / 1000
    bandwidth = memory_consumption / elapsed_time / 1024**3
    print(f"Nanoseconds Elapsed (rowsum): {elapsed_time * 1e9}")
    print(f"Memory Bandwidth (rowsum): {bandwidth:.4f} GiB/s")

    ground_truth = torch.sum(decompressed, dim=-1)
    error_abs = torch.max(torch.abs(out-ground_truth))
    error_rol = torch.max(torch.abs(out-ground_truth)/torch.abs(ground_truth))
    print("Absolute Error:", error_abs.item())
    print("Relative Error:", error_rol.item())
    print("Correct?", torch.allclose(out, ground_truth, rtol=1e-3))
    print("out", out)
    print("ground_truth", ground_truth)



if __name__ == "__main__":
    benchmark()
