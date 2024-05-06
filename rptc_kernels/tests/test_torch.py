import torch


def benchmark():
    torch.manual_seed(42)

    m, n = 4096, 4096
    memory_consumption = m * n * 2
    print(f"{m = }, {n = }, {memory_consumption = }")

    decompressed = torch.randn((m, n), dtype=torch.float16, device="cpu").cuda()
    x = torch.randn((n,), dtype=torch.float16, device="cpu").cuda()

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    result = decompressed @ x
    dummy = torch.sum(result)
    print(dummy.item())


if __name__ == "__main__":
    benchmark()
