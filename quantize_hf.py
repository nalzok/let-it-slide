import re
from pathlib import Path

import torch
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import trange

from rptc import quantize, dequantize


def sinkhorn_knopp_factorize(M):
    out_features, in_features = M.shape
    c = np.ones((1, in_features))
    r = np.ones((out_features, 1))

    # TODO: switch to torch
    # TODO: make n_iterations smaller
    MM = M**2
    n_iterations = 64
    for _ in range(n_iterations):
        c *= np.sum((1/r) * MM / c, axis=0, keepdims=True)
        r *= np.sum((1/r) * MM / c, axis=1, keepdims=True)

    out_scales = np.sqrt(r)
    in_scales = np.sqrt(c/in_features)
    normalized = (1/out_scales) * M / in_scales

    return out_scales, normalized, in_scales


hf_to_hessian = {
    "q": "qkv",
    "k": "qkv",
    "v": "qkv",
    "o": "o",
    "gate": "up",
    "up": "up",
    "down": "down",
}


model_to_hessian = {
    "meta-llama/Llama-2-7b-hf": "/share/desa/nfs01/qs234/huggingface/hub/models--relaxml--Hessians-Llama-2-7b-6144/snapshots/cafc59c036c6416ec2a9d5790752bec51297c197/",
}


def flat_to_sym(V, N):
    A = torch.zeros(N, N, dtype=V.dtype, device=V.device)
    idxs = torch.tril_indices(N, N, device=V.device)
    A[idxs.unbind()] = V
    A[idxs[1, :], idxs[0, :]] = V
    return A


def load_hessian(hessian_root, layer, hf_name):
    hessian_name = hf_to_hessian[hf_name]
    path = hessian_root / f"{layer}_{hessian_name}.pt"
    hessian_data = torch.load(path)
    hessian = flat_to_sym(hessian_data["flatH"], hessian_data["n"])
    return hessian


batch_quantize = jax.jit(jax.vmap(quantize, (None, None, 0)))
batch_dequantize = jax.jit(jax.vmap(dequantize, (None, 0)))


def round_weights(weights, corrections, permuted_alphabet):
    rounded_weights = np.empty_like(weights)

    batch_size = 16
    for i in (pbar := trange((weights.shape[0]+batch_size-1) // batch_size, leave=False)):
        batch_index = slice(batch_size*i, batch_size*i+batch_size)
        rows = jnp.array(weights[batch_index])
        quantized, expected_rho = batch_quantize(permuted_alphabet, corrections, rows)
        rounded_weights[batch_index] = batch_dequantize(permuted_alphabet, quantized)
        pbar.set_description(str(jnp.mean(expected_rho)))

    return rounded_weights


def quantize_model(model, hessian_root, permuted_alphabet):
    pattern = re.compile(r"model\.layers\.(\d+)\.(self_attn\.(\w+)_proj|mlp\.(\w+)_proj)")
    for name, module in model.named_modules():
        match = re.fullmatch(pattern, name)
        if match is not None:
            layer = int(match.group(1))
            hf_name = match.group(3) or match.group(4)

            weights = np.copy(module.weight.numpy())
            out_scales, normalized, in_scales = sinkhorn_knopp_factorize(weights)

            H = load_hessian(hessian_root, layer, hf_name)
            U = torch.linalg.cholesky(H, upper=True)
            corrections = jnp.array(U.T * in_scales)

            # rng = np.random.default_rng(42)
            # permuted_alphabet = rng.choice(normalized.reshape(-1), size=1<<16, replace=False)
            normalized_rounded = round_weights(normalized, corrections, permuted_alphabet)
            recon = out_scales * normalized_rounded * in_scales
            module.weight.copy_(torch.from_numpy(recon))

            W = torch.from_numpy(weights).float()
            E = W - torch.from_numpy(recon).float()
            raw = torch.from_numpy(normalized - normalized_rounded).square().mean()
            frob_abs = E.square().mean()
            frob_rel = frob_abs / W.square().mean()
            proxy_abs = ((E @ H) * E).mean()
            proxy_rel = proxy_abs / ((W @ H) * W).mean()
            print(f"{layer=}, {hf_name=}, {raw=:.6g}, {frob_abs=:.6g}, {frob_rel=:.6g}, {proxy_abs=:.6g}, {proxy_rel=:.6f}")


def main(model_name, shift_register_size, permuted_alphabet_path, quantized_root):
    permuted_alphabet_tag = "none" if permuted_alphabet_path is None else permuted_alphabet_path.stem
    quantized_name = f"{model_name}-{permuted_alphabet_tag}"
    quantized_path = quantized_root / quantized_name
    print(f"{quantized_path = }")

    M = 1<<shift_register_size
    if permuted_alphabet_path is None:
        permutation = jax.random.permutation(jax.random.PRNGKey(42), M)
        alphabet = jsp.stats.norm.ppf((2*jnp.arange(M)+1)/2/M)
        permuted_alphabet = alphabet[permutation]
    else:
        permuted_alphabet = jnp.load(permuted_alphabet_path)

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    with torch.inference_mode():
        hessian_root = Path(model_to_hessian[model_name])
        quantize_model(model, hessian_root, permuted_alphabet)

    model.save_pretrained(quantized_path, safe_serialization=True)

    del model

    quantized_model = AutoModelForCausalLM.from_pretrained(quantized_path, device_map="auto")
    input_text = "It is a truth universally acknowledged that "
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    output = quantized_model.generate(**input_ids, max_new_tokens=128)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(output_text)


if __name__ == "__main__":
    model_name = "meta-llama/Llama-2-7b-hf"
    shift_register_size = 16
    permuted_alphabet_path = None
    quantized_root = Path("/share/desa/nfs01/qs234/checkpoints/")
    main(model_name, shift_register_size, permuted_alphabet_path, quantized_root)
