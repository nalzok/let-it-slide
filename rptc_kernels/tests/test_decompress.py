import torch
import rptc_kernels


def benchmark():
    torch.manual_seed(42)

    compressed_m = 4096
    compressed_n = 256
    m, n = compressed_m, compressed_n * 16
    memory_consumption = compressed_m * compressed_n * 4
    print(f"{m = }, {n = }, {memory_consumption = }")

    compressed = torch.randint(torch.iinfo(torch.int32).max+1, (compressed_m, compressed_n),
                      dtype=torch.int32, device="cuda")
    codebook = torch.randn(1<<16, dtype=torch.float16, device="cuda")
    out = torch.full((m, n), -1, dtype=torch.float16, device="cuda")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    rptc_kernels.decompress(compressed, codebook, out)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event) / 1000
    print(f"Seconds elapsed: {elapsed_time_ms:.4f}")

    x = torch.randn((n,), dtype=torch.float16, device="cuda")
    out = torch.full((m,), -1, dtype=torch.float16, device="cuda")

    matvec_list = [
        (16, rptc_kernels.decompress_matvec_16),
        (14, rptc_kernels.decompress_matvec_14),
        (12, rptc_kernels.decompress_matvec_12),
        (10, rptc_kernels.decompress_matvec_10),
        (8, rptc_kernels.decompress_matvec_8),
    ]

    for L, decompress_matvec in matvec_list:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        decompress_matvec(compressed, codebook[:1<<L], x, out)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event) / 1000
        print(f"Seconds elapsed ({L = }): {elapsed_time_ms:.4f}")


if __name__ == "__main__":
    benchmark()
