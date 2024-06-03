import re
from pathlib import Path
from functools import partial

import torch
import numpy as np
import jax
import jax.numpy as jnp
from transformers import AutoModelForCausalLM, AutoTokenizer

from rptc import quantize, dequantize


hf_to_fused = {
    "q": "qkv",
    "k": "qkv",
    "v": "qkv",
    "o": "o",
    "up": "up",
    "gate": "up",
    "down": "down",
}


model_to_hessian = {
    "meta-llama/Llama-2-7b-hf": "/mnt/desa_data/qingyao/.cache/huggingface/hub/models--relaxml--Hessians-Llama-2-7b-6144/snapshots/cafc59c036c6416ec2a9d5790752bec51297c197/",
}


model_to_quip = {
    "meta-llama/Llama-2-7b-hf": "/mnt/desa_data/qingyao/research/quip-sharp/ckpt/2_7b_2bit/",
}


def flat_to_sym(V, N):
    A = torch.zeros(N, N, dtype=V.dtype, device=V.device)
    idxs = torch.tril_indices(N, N, device=V.device)
    A[idxs.unbind()] = V
    A[idxs[1, :], idxs[0, :]] = V
    return A


def load_hessian(hessian_root, layer, hf_name):
    fused_name = hf_to_fused[hf_name]
    path = hessian_root / f"{layer}_{fused_name}.pt"
    hessian_data = torch.load(path)
    hessian = flat_to_sym(hessian_data["flatH"], hessian_data["n"])
    mu = hessian_data["mu"]
    hessian += torch.outer(mu, mu)
    hessian.div_(torch.diag(hessian).mean())
    n = hessian_data["n"]
    idx = torch.arange(n)
    hessian[idx, idx] += 1e-2
    return hessian


def load_quip(quip_root, layer, hf_name):
    fused_name = hf_to_fused[hf_name]
    hatW_path = quip_root / f"hatW-{layer}_{fused_name}.pt"
    quip_hatW = torch.load(hatW_path)

    out_features, _ = quip_hatW.shape
    if hf_name == "q":
        quip_hatW = quip_hatW[:out_features//3]
    elif hf_name == "k":
        quip_hatW = quip_hatW[out_features//3:out_features*2//3]
    elif hf_name == "v":
        quip_hatW = quip_hatW[out_features*2//3:]
    elif hf_name == "up":
        quip_hatW = quip_hatW[:out_features//2]
    elif hf_name == "gate":
        quip_hatW = quip_hatW[out_features//2:]

    return quip_hatW


def factorize(W):
    # c.r. Sinkhorn-Knopp algorithm
    # c.r. iterative proportional fitting
    # c.r. iterative Bregman projection

    def cond_fun(val):
        _, _, _, _, i = val
        return i < 64

    def body_fun(val):
        _, biases_in, scales_out, scales_in, i = val

        next_biases_out = jnp.mean((W - biases_in) / scales_in, axis=1, keepdims=True) / jnp.mean(1/scales_in)
        next_biases_in = jnp.mean((W - next_biases_out) / scales_out, axis=0, keepdims=True) / jnp.mean(1/scales_out)
        bias_balancer = (jnp.mean(next_biases_in) - jnp.mean(next_biases_out)) / 2
        next_biases_out += bias_balancer
        next_biases_in -= bias_balancer

        next_scales_out = jnp.sqrt(jnp.mean(((W - next_biases_out - next_biases_in) / scales_in)**2, axis=1, keepdims=True))
        next_scales_in = jnp.sqrt(jnp.mean(((W - next_biases_out - next_biases_in) / next_scales_out)**2, axis=0, keepdims=True))
        scale_balancer = jnp.exp((jnp.mean(jnp.log(next_scales_in)) - jnp.mean(jnp.log(next_scales_out))) / 2)
        next_scales_out *= scale_balancer
        next_scales_in /= scale_balancer

        next_i = i + 1

        next_val = next_biases_out, next_biases_in, next_scales_out, next_scales_in, next_i
        return next_val

    out_features, in_features = W.shape
    biases_out = jnp.zeros((out_features, 1))
    biases_in = jnp.zeros((1, in_features))
    scales_out = jnp.ones((out_features, 1))
    scales_in = jnp.ones((1, in_features))
    init_val = biases_out, biases_in, scales_out, scales_in, 0

    biases_out, biases_in, scales_out, scales_in, _ = jax.lax.while_loop(cond_fun, body_fun, init_val)

    normalized = (W - biases_out - biases_in) / scales_out / scales_in

    return biases_out, biases_in, scales_out, scales_in, normalized


def round_weights(key, weights, importance, shift_register_size):
    out_features, in_features = weights.shape
    num_banks = 32
    batch_size = 8
    assert out_features % (num_banks * batch_size) == 0, (out_features, num_banks, batch_size)
    n = out_features // batch_size // num_banks
    weights_reshaped = jnp.reshape(weights, (num_banks, n, batch_size, in_features))

    # TODO: consider H when choosing alphabet
    # TODO: Lloyd-Max Quantization
    # TODO: https://ieeexplore.ieee.org/document/340468
    # TODO: include more "outliers" and fewer small values
    # TODO: sort rows by kurtosis
    # TODO: block diagonal importance
    # TODO: jax.random.fold_in
    def round_megabatch(megabatch):
        M = 1<<shift_register_size
        x = (2*jnp.arange(M)+1)/2/M
        xp = (2*jnp.arange(megabatch.size)+1)/2/megabatch.size
        fp = jnp.sort(megabatch, axis=None)
        alphabet = jnp.interp(x, xp, fp)
        permuted_alphabet = jax.random.permutation(key, alphabet)

        def round_batch(batch):
            batch_quantize = jax.vmap(quantize, (None, None, 0))
            batch_dequantize = jax.vmap(dequantize, (None, 0))
            quantized, _ = batch_quantize(permuted_alphabet, importance, batch)
            rounded = batch_dequantize(permuted_alphabet, quantized)
            return rounded

        batch_rounded_reshaped = jax.lax.map(round_batch, megabatch)
        return batch_rounded_reshaped

    rounded_reshaped = jax.lax.map(round_megabatch, weights_reshaped)
    rounded = jnp.reshape(rounded_reshaped, (out_features, in_features))

    return rounded


@partial(jax.jit, static_argnames=("shift_register_size",))
def quantize_layer(key, H, W, shift_register_size):
    biases_out, biases_in, scales_out, scales_in, normalized = factorize(W)

    # TODO: jnp.sum(H, axis=-1) instead of jnp.diag(H)
    importance = jnp.diag(jnp.diag(H) * jnp.ravel(scales_in))

    normalized_rounded = round_weights(key, normalized, importance, shift_register_size)
    ftcq_recon = biases_out + biases_in + scales_out * scales_in * normalized_rounded

    return ftcq_recon


@jax.jit
def evaluate_layer(H, W, recon):
    frob_reference = jnp.sum(W**2)**0.5
    proxy_reference = jnp.sum((W @ H) * W)
    proxy_diag_reference = jnp.sum((W * jnp.diag(H)) * W)
    proxy_sum_reference = jnp.sum((W * jnp.sum(H, axis=-1)) * W)

    E = W - recon
    frob = jnp.sum(E**2) / frob_reference
    proxy = jnp.sum((E @ H) * E) / proxy_reference
    proxy_diag = jnp.sum((E * jnp.diag(H)) * E) / proxy_diag_reference
    proxy_sum = jnp.sum((E * jnp.sum(H, axis=-1)) * E) / proxy_sum_reference

    return frob, proxy, proxy_diag, proxy_sum


def quantize_model(key, full_model, hessian_root, quip_root, shift_register_size):
    pattern = re.compile(r"model\.layers\.(\d+)\.(self_attn\.(\w+)_proj|mlp\.(\w+)_proj)")
    for i, (name, module) in enumerate(full_model.named_modules()):
        match = re.fullmatch(pattern, name)
        if match is not None:
            layer = int(match.group(1))
            hf_name = match.group(3) or match.group(4)
            key_layer = jax.random.fold_in(key, i)

            H = jnp.array(load_hessian(hessian_root, layer, hf_name))
            W = jnp.array(module.weight.numpy())

            ftcq_recon = quantize_layer(key_layer, H, W, shift_register_size)
            ftcq_frob, ftcq_proxy, ftcq_proxy_diag, ftcq_proxy_sum = evaluate_layer(H, W, ftcq_recon)

            scale = jnp.mean(W**2)**0.5
            quip_hatW = jnp.array(load_quip(quip_root, layer, hf_name))
            quip_recon = quip_hatW * scale
            quip_frob, quip_proxy, quip_proxy_diag, quip_proxy_sum = evaluate_layer(H, W, quip_recon)

            # TODO: is Hessian diagonal informative enough?
            print(f"{layer=}, {hf_name=}, "
                  f"frob={ftcq_frob:g}({ftcq_frob/quip_frob:g}x), "
                  f"proxy={ftcq_proxy:g}({ftcq_proxy/quip_proxy:g}x), "
                  f"proxy_diag={ftcq_proxy_diag:g}({ftcq_proxy_diag/quip_proxy_diag:g}x), "
                  f"proxy_sum={ftcq_proxy_sum:g}({ftcq_proxy_sum/quip_proxy_sum:g}x)")

            module.weight.copy_(torch.from_numpy(np.array(ftcq_recon)))


def main(key, model_name, shift_register_size, ftcq_root):
    with torch.inference_mode():
        full_model = AutoModelForCausalLM.from_pretrained(model_name)
        hessian_root = Path(model_to_hessian[model_name])
        quip_root = Path(model_to_quip[model_name])
        quantize_model(key, full_model, hessian_root, quip_root, shift_register_size)

    ftcq_path = ftcq_root / model_name
    full_model.save_pretrained(ftcq_path, safe_serialization=True)

    del full_model

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ftcq_model = AutoModelForCausalLM.from_pretrained(ftcq_path, device_map="auto")
    input_text = "It is a truth universally acknowledged that "
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    output = ftcq_model.generate(**input_ids, max_new_tokens=128)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(output_text)


if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    model_name = "meta-llama/Llama-2-7b-hf"
    shift_register_size = 14
    ftcq_root = Path("/mnt/desa_data/qingyao/checkpoints/ftcq")
    main(key, model_name, shift_register_size, ftcq_root)
