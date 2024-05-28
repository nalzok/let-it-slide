# Implements the random permutation trellis encoder
# https://ee.stanford.edu/~gray/trellis.pdf

from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import scipy
from scipy.optimize import quadratic_assignment
import optax
from tqdm import trange
import matplotlib.pyplot as plt


def index_fn(x, L, S):
    x = (x >> 8) ^ (x >> 4) ^ x
    # x = x >> (L-S)
    x = x & ((1<<S)-1)
    return x


def manifest_alphabet(absolute, angle):
    dummy = absolute * jnp.exp(2j*jnp.pi * angle)
    alphabet = jnp.column_stack((jnp.real(dummy), jnp.imag(dummy)))
    return alphabet


def quantize(alphabet, L, S, R, samples):
    length, V = samples.shape

    assert jnp.iinfo(jnp.int32).bits % R == 0
    pack_size = jnp.iinfo(jnp.int32).bits // R

    permutation = jnp.arange(1<<L)
    idx = index_fn(permutation, L, S)
    permuted_alphabet = alphabet[idx]

    def viterbi_step(carry, index):
        rho, prev_msbs = carry

        reshaped_rho = jnp.reshape(rho, (1<<R, 1<<(L-R)))
        prev_msb = jnp.argmin(reshaped_rho, axis=-2)
        pos, offset = divmod(index, pack_size)
        updated_prev_msbs = prev_msbs.at[pos].add(prev_msb<<(offset*R))

        additive_loss = jnp.sum((samples[index]-permuted_alphabet)**2, axis=-1)
        updated_rho = jnp.repeat(reshaped_rho[prev_msb, jnp.arange(1<<(L-R))], 1<<R) + additive_loss

        return (updated_rho, updated_prev_msbs), None

    rho = jnp.full((1<<L,), jnp.inf)
    rho = rho.at[0].set(0)  # initial state is zero
    prev_msbs = jnp.zeros(((length+pack_size-1)//pack_size, 1<<(L-R)), dtype=jnp.int32)
    (rho, prev_msbs), _ = jax.lax.scan(viterbi_step, (rho, prev_msbs), jnp.arange(length))

    def backtrack_input(state, index):
        pos, offset = divmod(index, pack_size)
        msb = (prev_msbs[pos, state>>R]>>(offset*R)) & ((1<<R)-1)
        return msb<<(L-R) | state>>R, state & ((1<<R)-1)

    last_state = jnp.argmin(rho)
    zero, quantized_rev = jax.lax.scan(backtrack_input, last_state, jnp.flipud(jnp.arange(length)))
    # checkify.check(zero == 0, "zero should be 0, got {}", zero)
    quantized = jnp.flipud(quantized_rev)
    expected_loss = jnp.min(rho)

    return quantized, expected_loss


def dequantize(alphabet, L, S, R, quantized):
    def f(state, input_):
        next_state = (state<<R) | input_
        idx = index_fn(next_state, L, S)
        output = alphabet[idx]
        return next_state, output

    init_state = 0
    _, dequantized = jax.lax.scan(f, init_state, quantized)

    return dequantized
    

def evaluate(alphabet, L, S, R, samples):
    quantized, _ = quantize(alphabet, L, S, R, samples)
    dequantized = dequantize(alphabet, L, S, R, quantized)
    residual = samples - dequantized
    mse = jnp.mean(residual**2)

    bincount = jnp.bincount(quantized, length=1<<R)
    dist = bincount / jnp.sum(bincount)
    entropy = -jnp.sum(dist * jnp.log2(dist))

    return mse, entropy


def pretrain(absolute, angle, L, S, R, pre_learning_rate, pre_n_steps):
    permutation = jnp.arange(1<<L)
    idx = index_fn(permutation, L, S)
    groups = jnp.reshape(idx & ((1<<S)-1), (1<<(L-R), 1<<R))

    @jax.value_and_grad
    def eval_angle_grad(ang):
        alphabet = manifest_alphabet(absolute, ang)
        points = alphabet[groups]
        distances = jnp.sum((points[:, jnp.newaxis, :, :] - points[:, :, jnp.newaxis, :])**2, axis=-1)
        ungerboeck = jnp.sum(jnp.min(distances + jnp.eye(1<<R) * jnp.inf, axis=(-2, -1)), axis=-1)
        return -ungerboeck

    @jax.jit
    def train_step(ang, opt_state):
        dispersion, grads = eval_angle_grad(ang)
        updates, opt_state = gradient_transform.update(grads, opt_state, ang)
        ang = optax.apply_updates(ang, updates)
        return dispersion, ang, opt_state

    scheduler = optax.warmup_cosine_decay_schedule(
            init_value=0,
            peak_value=pre_learning_rate,
            warmup_steps=pre_n_steps // 256,
            decay_steps=pre_n_steps)
    gradient_transform = optax.chain(
            optax.scale_by_adam(),
            optax.scale_by_schedule(scheduler),
            optax.scale(-1.0),
    )
    opt_state = gradient_transform.init(angle)

    for step in (pbar := trange(n_steps)):
        ungerboeck, angle, opt_state = train_step(angle, opt_state)
        pbar.set_description(f"{ungerboeck.item() = :.4f}")

    return angle


def train(key, parameters, L, S, R, V, T, learning_rate, n_steps):
    @partial(jax.value_and_grad, has_aux=True)
    def eval_params_grad(params, samples):
        alphabet = manifest_alphabet(*params)
        mse, entropy = evaluate(alphabet, L, S, R, samples)
        return mse, entropy

    @jax.jit
    def train_step(key_step, params, opt_state):
        samples = jax.random.normal(key_step, (T, V))
        (mse, entropy), grads = eval_params_grad(params, samples)
        updates, opt_state = gradient_transform.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return mse, entropy, params, opt_state

    scheduler = optax.warmup_cosine_decay_schedule(
            init_value=0,
            peak_value=learning_rate,
            warmup_steps=n_steps // 256,
            decay_steps=n_steps)
    gradient_transform = optax.chain(
            optax.scale_by_adam(),
            optax.scale_by_schedule(scheduler),
            optax.scale(-1.0),
    )
    opt_state = gradient_transform.init(parameters)

    for step in (pbar := trange(n_steps, leave=False)):
        key_step = jax.random.fold_in(key, step)
        mse, entropy, parameters, opt_state = train_step(key_step, parameters, opt_state)
        pbar.set_description(f"{mse.item() = :.4f}, {entropy.item() = :.4f}")

    return parameters


def main(L, S, R, V, T, learning_rate, n_steps):
    key = jax.random.PRNGKey(42)
    key_alphabet, key_train, key_test = jax.random.split(key, num=3)

    assert V == 2
    gaussian = jax.random.normal(key_alphabet, (1<<S, V))
    dummy = jax.lax.complex(gaussian[:, 0], gaussian[:, 1])
    absolute, angle = jnp.absolute(dummy), jnp.angle(dummy)
    alphabet = manifest_alphabet(absolute, angle)
    assert alphabet.shape == (1<<S, V), (alphabet.shape, (1<<S, V))

    # Speed
    # from time import perf_counter
    # dummy_replication = 2
    # dummy_batch_size = 1024
    # dummy_T = 4096
    # dummy = jnp.zeros((dummy_replication, dummy_batch_size, dummy_T, V), dtype=jnp.float16)
    # print("dummy.shape", dummy.shape)
    # batch_quantize = jax.vmap(quantize, (None, None, None, None, 0))
    # _ = jax.lax.map(partial(batch_quantize, alphabet, L, S, R), dummy)    # warmup
    # begin = perf_counter()
    # _ = jax.lax.map(partial(batch_quantize, alphabet, L, S, R), dummy)
    # end = perf_counter()
    # print("Throughput", dummy.size/(end-begin))

    # Pretraining
    # angle = pretrain(absolute, angle, L, S, R, learning_rate, n_steps)
    # permutation = np.arange(1<<L)
    # idx = index_fn(permutation, L, S)
    # groups = np.reshape(idx & ((1<<S)-1), (1<<(L-R), 1<<R))
    # A = np.zeros((1<<S, 1<<S))
    # for group in groups:
    #     for i in group:
    #         for j in group:
    #             A[i, j] += i != j
    # B = np.sum(np.abs(alphabet[jnp.newaxis, :, :] - alphabet[:, jnp.newaxis, :]), axis=-1)
    # res = quadratic_assignment(A, B, options={"maximize": True})
    # absolute, angle = absolute[res.col_ind], angle[res.col_ind]
    # alphabet_permuted = manifest_alphabet(absolute, angle)

    # Visualization
    # fig, ax = plt.subplots()
    # for i, group in enumerate(groups):
    #     ab = alphabet[group]
    #     ax.scatter(ab[:, 0], ab[:, 1], marker=".")
    # ax.axis("equal")
    # ax.set_xlim((-5, 5))
    # ax.set_ylim((-5, 5))
    # fig.tight_layout()
    # fig.set_size_inches(10, 10)
    # fig.set_dpi(200)
    # fig.savefig("alphabet-before.png")
    # plt.close()
    #
    # fig, ax = plt.subplots()
    # for i, group in enumerate(groups):
    #     ab = alphabet_permuted[group]
    #     ax.scatter(ab[:, 0], ab[:, 1], marker=".")
    # ax.axis("equal")
    # ax.set_xlim((-5, 5))
    # ax.set_ylim((-5, 5))
    # fig.tight_layout()
    # fig.set_size_inches(10, 10)
    # fig.set_dpi(200)
    # fig.savefig("alphabet-after.png")
    # plt.close()

    # Fine-tuning
    absolute, angle = train(key_train, (absolute, angle), L, S, R, V, T, learning_rate, n_steps)
    alphabet_tuned = manifest_alphabet(absolute, angle)

    # A = np.zeros((1<<S, 1<<S))
    # for group in groups:
    #     for i in group:
    #         for j in group:
    #             A[i, j] += i != j
    # B = np.sum(np.abs(alphabet_tuned[jnp.newaxis, :, :] - alphabet_tuned[:, jnp.newaxis, :]), axis=-1)
    # res = quadratic_assignment(A, B, options={"maximize": True})
    # absolute, angle = absolute[res.col_ind], angle[res.col_ind]
    # alphabet_tuned_permuted = manifest_alphabet(absolute, angle)
    # angle = pretrain(absolute, angle, L, S, R, learning_rate, n_steps)
    # alphabet_tuned_permuted = manifest_alphabet(absolute, angle)

    # MSE Evaluation
    samples = jax.random.normal(key_test, (2**20//T, T, V))

    # mse_all, entropy_all = jax.lax.map(partial(evaluate, alphabet, L, S, R), samples)
    # mse_mean, mse_std = jnp.mean(mse_all).item(), jnp.std(mse_all).item()
    # entropy_mean, entropy_std = jnp.mean(entropy_all).item(), jnp.std(entropy_all).item()
    # print(f"Original: {mse_mean = :.4f} ({mse_std:.3f}), {entropy_mean = :.4f} ({entropy_std:.3f})")

    # mse_all, entropy_all = jax.lax.map(partial(evaluate, alphabet_permuted, L, S, R), samples)
    # mse_mean, mse_std = jnp.mean(mse_all).item(), jnp.std(mse_all).item()
    # entropy_mean, entropy_std = jnp.mean(entropy_all).item(), jnp.std(entropy_all).item()
    # print(f"Permuted: {mse_mean = :.4f} ({mse_std:.3f}), {entropy_mean = :.4f} ({entropy_std:.3f})")

    mse_all, entropy_all = jax.lax.map(partial(evaluate, alphabet_tuned, L, S, R), samples)
    mse_mean, mse_std = jnp.mean(mse_all).item(), jnp.std(mse_all).item()
    entropy_mean, entropy_std = jnp.mean(entropy_all).item(), jnp.std(entropy_all).item()
    print(f"Finetuned: {mse_mean = :.4f} ({mse_std:.3f}), {entropy_mean = :.4f} ({entropy_std:.3f})")

    # mse_all, entropy_all = jax.lax.map(partial(evaluate, alphabet_tuned_permuted, L, S, R), samples)
    # mse_mean, mse_std = jnp.mean(mse_all).item(), jnp.std(mse_all).item()
    # entropy_mean, entropy_std = jnp.mean(entropy_all).item(), jnp.std(entropy_all).item()
    # print(f"Finetuned+Permuted: {mse_mean = :.4f} ({mse_std:.3f}), {entropy_mean = :.4f} ({entropy_std:.3f})")


if __name__ == "__main__":
    L = 16
    S = 9
    R = 4
    V = 2
    T = 2**10
    learning_rate = 1e-3
    n_steps = 2**10
    main(L, S, R, V, T, learning_rate, n_steps)
