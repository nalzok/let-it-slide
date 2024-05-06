import torch
import rptc_kernels


def prepare_arguments(L, m, n, compressed_m, compressed_n):
    compressed = torch.randint(torch.iinfo(torch.int32).max+1,
                               (compressed_n // 32 // 4, m, 32 * 4),
                               dtype=torch.int32,
                               device="cuda").cuda()
    assert compressed.numel() == compressed_m * compressed_n
    codebook = torch.ones(1<<L, dtype=torch.float16, device="cpu").cuda()
    x = torch.randn((n,), dtype=torch.float16, device="cpu").cuda()
    out = torch.zeros((m,), dtype=torch.float16, device="cuda")     # some kernels require zero-initialization
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return compressed, codebook, x, out


def benchmark():
    torch.manual_seed(42)

    compressed_m = 4096
    compressed_n = 256
    m, n = compressed_m, compressed_n * 16
    memory_consumption = compressed_m * compressed_n * 4
    print(f"{m = }, {n = }, {memory_consumption = }")

    matvec_list = [
        ("decompress_matvec_16", 16, rptc_kernels.decompress_matvec_16),
        ("decompress_matvec_14", 14, rptc_kernels.decompress_matvec_14),
        ("decompress_matvec_12", 12, rptc_kernels.decompress_matvec_12),
        ("decompress_matvec_10", 10, rptc_kernels.decompress_matvec_10),
        ("decompress_matvec_8", 8, rptc_kernels.decompress_matvec_8),
        ("decompress_matvec_6", 6, rptc_kernels.decompress_matvec_6),
        ("decompress_matvec_4", 4, rptc_kernels.decompress_matvec_4),
        ("decompress_matvec_2", 2, rptc_kernels.decompress_matvec_2),
        ("decompress_matvec_t_16", 16, rptc_kernels.decompress_matvec_t_16),
        ("decompress_matvec_t_14", 14, rptc_kernels.decompress_matvec_t_14),
    ]

    for name, L, decompress_matvec in matvec_list:
        compressed, codebook, x, out = prepare_arguments(L, m, n, compressed_m, compressed_n)
        _ = decompress_matvec(compressed, codebook, x, out)
        print(name, out.min().item(), "=", out.max().item(), "=", x.sum().item())

        # elapsed_time_list = []
        # for _ in range(1):
        #     compressed, codebook, x, out = prepare_arguments(L, m, n, compressed_m, compressed_n)
        #     elapsed_time = decompress_matvec(compressed, codebook, x, out) / 1000
        #     elapsed_time_list.append(elapsed_time)
        #
        # avg_elapsed_time = sum(elapsed_time_list)/len(elapsed_time_list)
        # bandwidth = memory_consumption / avg_elapsed_time / 1024**3
        # print(name, f"Memory Bandwidth (decompress_matvec<{L=}>): {bandwidth:.4f} GiB/s, {avg_elapsed_time = }")

    # PyTorch
    decompressed = torch.ones((m, n), dtype=torch.float16, device="cpu").cuda()
    x = torch.randn((n,), dtype=torch.float16, device="cpu").cuda()
    out = decompressed @ x
    print("torch", out.min().item(), "=", out.max().item(), "=", x.sum().item())


if __name__ == "__main__":
    benchmark()
