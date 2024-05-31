import re
from pathlib import Path
from functools import partial

import torch
import numpy as np
import jax
import jax.numpy as jnp
from transformers import AutoModelForCausalLM, AutoTokenizer

from rptc import quantize, dequantize


# TODO: debias
def sinkhorn_knopp_factorize(M):
    MM = M**2

    def cond_fun(val):
        _, _, residual, i = val
        # return jnp.logical_and(residual > 1e-13, i < 32)
        return i < 32

    def body_fun(val):
        c, r, _, i = val
        next_c = c * jnp.sum(MM / r / c, axis=0, keepdims=True)
        next_r = r * jnp.sum(MM / r / next_c, axis=1, keepdims=True)
        maybe_doubly_stochastic = MM / next_r / next_c
        row_sums = jnp.sum(maybe_doubly_stochastic, axis=0)
        col_sums = jnp.sum(maybe_doubly_stochastic, axis=1)
        next_residual = jnp.maximum(jnp.var(row_sums), jnp.var(col_sums))
        next_i = i + 1
        return next_c, next_r, next_residual, next_i

    out_features, in_features = M.shape
    c = jnp.ones((1, in_features))
    r = jnp.ones((out_features, 1))
    init_val = c, r, jnp.inf, 0
    c, r, _, _ = jax.lax.while_loop(cond_fun, body_fun, init_val)

    out_scales = jnp.sqrt(r)
    in_scales = jnp.sqrt(c/in_features)
    normalized = M / out_scales / in_scales

    return out_scales, normalized, in_scales


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


def round_weights(weights, importance, permuted_alphabet):
    out_features, in_features = weights.shape
    batch_size = 32
    assert out_features % batch_size == 0
    n = out_features // batch_size
    weights_reshaped = jnp.reshape(weights, (n, batch_size, in_features))

    def round_(batch):
        batch_quantize = jax.vmap(quantize, (None, None, 0))
        batch_dequantize = jax.vmap(dequantize, (None, 0))
        quantized, _ = batch_quantize(permuted_alphabet, importance, batch)
        rounded = batch_dequantize(permuted_alphabet, quantized)
        return rounded

    rounded_reshaped = jax.lax.map(round_, weights_reshaped)
    rounded = jnp.reshape(rounded_reshaped, (out_features, in_features))

    return rounded


@partial(jax.jit, static_argnames=("shift_register_size",), donate_argnames=("W",))
def quantize_layer(key, H, W, quip_hatW, shift_register_size):
    scale = jnp.mean(W**2)**0.5
    frob_reference = jnp.sum(W**2)**0.5
    proxy_reference = jnp.sum((W @ H) * W)

    out_scales, normalized, in_scales = sinkhorn_knopp_factorize(W)

    importance = jnp.diag(jnp.diag(H) * jnp.ravel(in_scales))

    M = 1<<shift_register_size
    x = (2*jnp.arange(M)+1)/2/M
    xp = (2*jnp.arange(jnp.size(W))+1)/2/jnp.size(W)
    fp = jnp.sort(normalized, axis=None)
    alphabet = jnp.interp(x, xp, fp)
    permuted_alphabet = jax.random.permutation(key, alphabet)

    normalized_rounded = round_weights(normalized, importance, permuted_alphabet)
    recon = out_scales * normalized_rounded * in_scales

    E = W - recon
    raw = jnp.mean((normalized-normalized_rounded)**2)
    frob = jnp.sum(E**2) / frob_reference
    proxy = jnp.sum((E @ H) * E) / proxy_reference

    quip_recon = quip_hatW * scale
    quip_E = W - quip_recon
    quip_frob = jnp.sum(quip_E**2) / frob_reference
    quip_proxy = jnp.sum((quip_E @ H) * quip_E) / proxy_reference

    return recon, raw, frob, proxy, quip_frob, quip_proxy


def quantize_model(key, model, hessian_root, quip_root, shift_register_size):
    pattern = re.compile(r"model\.layers\.(\d+)\.(self_attn\.(\w+)_proj|mlp\.(\w+)_proj)")
    for i, (name, module) in enumerate(model.named_modules()):
        match = re.fullmatch(pattern, name)
        if match is not None:
            layer = int(match.group(1))
            hf_name = match.group(3) or match.group(4)
            key_layer = jax.random.fold_in(key, i)

            H = jnp.array(load_hessian(hessian_root, layer, hf_name))
            W = jnp.array(module.weight.numpy())
            quip_hatW = jnp.array(load_quip(quip_root, layer, hf_name))

            recon, raw, frob, proxy, quip_frob, quip_proxy = quantize_layer(key_layer, H, W, quip_hatW, shift_register_size)
            module.weight.copy_(torch.from_numpy(np.array(recon)))

            print(f"{layer=}, {hf_name=}, {raw=:g}, frob={frob:g}({quip_frob:g}), proxy={proxy:g}({quip_proxy:g})")


def main(key, model_name, quantized_root, shift_register_size):
    quantized_path = quantized_root / model_name
    print(f"{quantized_path = }")

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    with torch.inference_mode():
        hessian_root = Path(model_to_hessian[model_name])
        quip_root = Path(model_to_quip[model_name])
        quantize_model(key, model, hessian_root, quip_root, shift_register_size)

    model.save_pretrained(quantized_path, safe_serialization=True)

    del model

    quantized_model = AutoModelForCausalLM.from_pretrained(quantized_path, device_map="auto")
    input_text = "It is a truth universally acknowledged that "
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    output = quantized_model.generate(**input_ids, max_new_tokens=128)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(output_text)


if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    model_name = "meta-llama/Llama-2-7b-hf"
    shift_register_size = 16
    quantized_root = Path("/mnt/desa_data/qingyao/checkpoints/")
    main(key, model_name, quantized_root, shift_register_size)
