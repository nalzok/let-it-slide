# Implements the random permutation trellis encoder
# https://ee.stanford.edu/~gray/trellis.pdf

from functools import partial

import jax
import jax.numpy as jnp
import optax
from tqdm import trange


def index_fn(x, L, S):
    x = x * (2 * x + 1)
    x ^= x >> (L-S)
    x = x & ((1<<S)-1)
    return x


def quantize(alphabet, L, S, R, importance, samples):
    length, V = samples.shape
    maybe_length, maybe_V = importance.shape
    assert maybe_length == length and maybe_V == V, (importance.shape, samples.shape)

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

        additive_loss = jnp.sum(importance[index] * (samples[index]-permuted_alphabet)**2, axis=-1)
        updated_rho = jnp.repeat(reshaped_rho[prev_msb, jnp.arange(1<<(L-R))], 1<<R) + additive_loss

        return (updated_rho, updated_prev_msbs), None

    rho = jnp.full((1<<L,), jnp.inf)
    rho = rho.at[0].set(0)  # initial state is zero (TODO: Tail-Biting)
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
    

def evaluate(alphabet, L, S, R, importance, samples):
    quantized, _ = quantize(alphabet, L, S, R, importance, samples)
    dequantized = dequantize(alphabet, L, S, R, quantized)
    residual = samples - dequantized
    mse = jnp.mean(importance * residual**2)

    bincount = jnp.bincount(quantized, length=1<<R)
    dist = bincount / jnp.sum(bincount)
    entropy = -jnp.sum(dist * jnp.log2(dist))

    return mse, entropy


def train(key, alphabet, L, S, R, V, T, importance, learning_rate, n_steps):
    @jax.jit
    def train_step(key_step, ab, opt_state):
        samples = jax.random.normal(key_step, (T, V))
        grad_fn = jax.value_and_grad(evaluate, has_aux=True)
        (mse, entropy), grads = grad_fn(ab, L, S, R, importance, samples)
        updates, opt_state = gradient_transform.update(grads, opt_state, ab)
        ab = optax.apply_updates(ab, updates)
        return mse, entropy, ab, opt_state

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
    opt_state = gradient_transform.init(alphabet)

    for step in (pbar := trange(n_steps, leave=False)):
        key_step = jax.random.fold_in(key, step)
        mse, entropy, alphabet, opt_state = train_step(key_step, alphabet, opt_state)
        pbar.set_description(f"{mse.item() = :.4f}, {entropy.item() = :.4f}")

    return alphabet


def main(L, S, R, V, T, learning_rate, n_steps):
    key = jax.random.PRNGKey(42)
    key_alphabet, key_train, key_test = jax.random.split(key, num=3)

    assert V == 2
    alphabet = jax.random.normal(key_alphabet, (1<<S, V))
    importance = jnp.ones((T, V))

    alphabet_tuned = train(key_train, alphabet, L, S, R, V, T, importance, learning_rate, n_steps)

    samples = jax.random.normal(key_test, (2**20//T, T, V))

    mse_all, entropy_all = jax.lax.map(partial(evaluate, alphabet, L, S, R, importance), samples)
    mse_mean, mse_std = jnp.mean(mse_all).item(), jnp.std(mse_all).item()
    entropy_mean, entropy_std = jnp.mean(entropy_all).item(), jnp.std(entropy_all).item()
    print(f"Original: {mse_mean = :.4f} ({mse_std:.3f}), {entropy_mean = :.4f} ({entropy_std:.3f})")

    mse_all, entropy_all = jax.lax.map(partial(evaluate, alphabet_tuned, L, S, R, importance), samples)
    mse_mean, mse_std = jnp.mean(mse_all).item(), jnp.std(mse_all).item()
    entropy_mean, entropy_std = jnp.mean(entropy_all).item(), jnp.std(entropy_all).item()
    print(f"Finetuned: {mse_mean = :.4f} ({mse_std:.3f}), {entropy_mean = :.4f} ({entropy_std:.3f})")


if __name__ == "__main__":
    L = 16
    S = 9
    R = 4
    V = 2
    T = 2**10
    learning_rate = 1e-3
    n_steps = 2**10
    main(L, S, R, V, T, learning_rate, n_steps)
