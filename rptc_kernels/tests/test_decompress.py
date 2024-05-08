import torch
import rptc_kernels


def prepare_arguments(L, S, m, n, compressed_m, compressed_n):
    compressed = torch.randint(torch.iinfo(torch.int32).max+1,
                               (compressed_n // 32 // 4, m, 32 * 4),
                               dtype=torch.int32,
                               device="cuda").cuda()
    assert compressed.numel() == compressed_m * compressed_n
    codebook = torch.ones((S,), dtype=torch.float16, device="cpu").cuda()
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
        ("decompress_matvec_16_128", 16, 128, rptc_kernels.decompress_matvec_16_128),
        ("decompress_matvec_16_64", 16, 64, rptc_kernels.decompress_matvec_16_64),
        ("decompress_matvec_14_128", 14, 128, rptc_kernels.decompress_matvec_14_128),
        ("decompress_matvec_14_64", 14, 64, rptc_kernels.decompress_matvec_14_64),
    ]

    for name, L, S, decompress_matvec in matvec_list:
        compressed, codebook, x, out = prepare_arguments(L, S, m, n, compressed_m, compressed_n)
        _ = decompress_matvec(compressed, codebook, x, out)
        print(name, out.min().item(), "=", out.max().item(), "=", x.sum().item())

    # PyTorch
    decompressed = torch.ones((m, n), dtype=torch.float16, device="cpu").cuda()
    x = torch.randn((n,), dtype=torch.float16, device="cpu").cuda()
    out = decompressed @ x
    print("torch", out.min().item(), "=", out.max().item(), "=", x.sum().item())


if __name__ == "__main__":
    benchmark()
