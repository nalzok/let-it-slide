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
    decompressed = torch.empty((m, n), dtype=torch.float16, device="cuda")

    elapsed_time = rptc_kernels.decompress(compressed, codebook, decompressed) / 1000
    bandwidth = (memory_consumption + m * n * 2) / elapsed_time / 1024**3
    print(f"Memory Bandwidth (decompress): {bandwidth:.4f} GiB/s")

    x = torch.randn((n,), dtype=torch.float16, device="cuda")

    matvec_list = [
        (16, rptc_kernels.decompress_matvec_16),
        (14, rptc_kernels.decompress_matvec_14),
        (12, rptc_kernels.decompress_matvec_12),
        (10, rptc_kernels.decompress_matvec_10),
        (8, rptc_kernels.decompress_matvec_8),
    ]

    for L, decompress_matvec in matvec_list:
        out = torch.zeros((m,), dtype=torch.float32, device="cuda")
        elapsed_time = decompress_matvec(compressed, codebook[:1<<L], x, out) / 1000
        bandwidth = memory_consumption / elapsed_time / 1024**3
        print(f"Memory Bandwidth (decompress_matvec<{L=}>): {bandwidth:.4f} GiB/s")



if __name__ == "__main__":
    benchmark()
