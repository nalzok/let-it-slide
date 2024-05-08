# Implements the random permutation trellis encoder
# https://ee.stanford.edu/~gray/trellis.pdf

from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import optax
from tqdm import trange


def index_fn(x, L, S):
    x = x * (2 * x + 1)
    x ^= x >> (L-S)
    x = x & ((1<<S)-1)
    return x


def quantize(alphabet, L, S, corrections, samples, diag_only=True):
    length = samples.shape[0]
    permutation = jnp.arange(1<<L)
    idx = index_fn(permutation, L, S)
    permuted_alphabet = alphabet[idx]

    assert len(samples.shape) == 1, samples.shape
    assert len(corrections.shape) == 2, corrections.shape
    assert length == corrections.shape[0] == corrections.shape[1], (length, corrections.shape)

    # checkify.check(jnp.allclose(jnp.tril(corrections), corrections), "corrections must be lower triangular")

    def viterbi_step(carry, index):
        rho, prev_states = carry
        prefixes = jnp.arange(1<<(L-2))
        discarded = jnp.arange(1<<2)
        last_state = (discarded<<(L-2)) | prefixes[..., jnp.newaxis]

        optimal_discarded = jnp.argmin(rho[last_state], axis=-1)
        prev_state = (optimal_discarded<<(L-2)) | prefixes
        updated_prev_states = prev_states.at[index].set(prev_state)
        
        if diag_only:
            additive_loss = jnp.sum(corrections[index]**2) * (samples[index] - permuted_alphabet)**2

        else:
            def end(val):
                idx, state, error = val
                return idx >= 0

            def make_correction(val):
                idx, state, error = val
                updated_state = prev_states[idx, state>>2]
                updated_error = error + corrections[index, idx] * (samples[idx] - permuted_alphabet[state])
                return idx-1, updated_state, updated_error

            init_val = (index, jnp.arange(1<<L), jnp.zeros((1<<L,)))
            negative_one, zero, error = jax.lax.while_loop(end, make_correction, init_val)
            # checkify.check(negative_one == -1, "negative_one should be -1, got {}", negative_one)
            # checkify.check(zero == 0, "zero should be 0, got {}", zero)
            additive_loss = error**2

        updated_rho = jnp.repeat(rho[prev_state], 1<<2) + additive_loss

        return (updated_rho, updated_prev_states), None

    rho = jnp.full((1<<L,), jnp.inf)
    rho = rho.at[0].set(0)  # initial state is zero (TODO: Tail-Biting)
    prev_states = jnp.empty((length, 1<<(L-2)), dtype=int)
    (rho, prev_states), _ = jax.lax.scan(viterbi_step, (rho, prev_states), jnp.arange(length))

    def backtrack_input(state, prev_state):
        return prev_state[state>>2], state & 0b11

    last_state = jnp.argmin(rho)
    zero, quantized_rev = jax.lax.scan(backtrack_input, last_state, jnp.flipud(prev_states))
    # checkify.check(zero == 0, "zero should be 0, got {}", zero)
    quantized = jnp.flipud(quantized_rev)
    expected_loss = jnp.min(rho)

    return quantized, expected_loss


def dequantize(alphabet, L, S, quantized):
    def f(state, input_):
        next_state = (state<<2) | input_
        idx = index_fn(next_state, L, S)
        output = alphabet[idx]
        return next_state, output

    init_state = 0
    _, dequantized = jax.lax.scan(f, init_state, quantized)

    return dequantized
    

def evaluate(alphabet, L, S, corrections, samples):
    quantized, _ = quantize(alphabet, L, S, corrections, samples)
    dequantized = dequantize(alphabet, L, S, quantized)
    residual = samples - dequantized
    mse = jnp.mean(residual**2)

    bincount = jnp.bincount(quantized, length=4)
    dist = bincount / jnp.sum(bincount)
    entropy = -jnp.sum(dist * jnp.log2(dist))

    return mse, entropy


def train(key, alphabet, L, S, block_size, learning_rate, n_steps):

    @jax.jit
    def train_step(key_step, ab, opt_state):
        grad_fn = jax.value_and_grad(evaluate, has_aux=True)
        samples = jax.random.normal(key_step, (block_size,))
        corrections = jnp.eye(block_size)
        (mse, entropy), grads = grad_fn(ab, L, S, corrections, samples)
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

    for step in (pbar := trange(n_steps)):
        key_step = jax.random.fold_in(key, step)
        mse, entropy, alphabet, opt_state = train_step(key_step, alphabet, opt_state)
        pbar.set_description(f"{mse.item() = :.4f}, {entropy.item() = :.4f}")

    return alphabet


def main(L, S, block_size, learning_rate, n_steps):
    key = jax.random.PRNGKey(42)
    key_train, key_test = jax.random.split(key)
    alphabet_half = jsp.stats.norm.ppf(1/2 + 1/(1<<(S+1)) + jnp.arange(1<<(S-1))/(1<<S))
    alphabet = jnp.concatenate([-jnp.flipud(alphabet_half), alphabet_half])
    print("Before:", alphabet)

    samples = jax.random.normal(key_test, (2**20//block_size, block_size))
    corrections = jnp.eye(block_size)

    mse_all, entropy_all = jax.lax.map(partial(evaluate, alphabet, L, S, corrections), samples)
    mse_mean = jnp.mean(mse_all).item()
    mse_std = jnp.std(mse_all).item()
    entropy_mean = jnp.mean(entropy_all).item()
    entropy_std = jnp.std(entropy_all).item()
    print(f"Before: {mse_mean = :.4f} ({mse_std:.3f}), {entropy_mean = :.4f} ({entropy_std:.3f})")

    # fine-tine the alphabet
    # alphabet = train(key_train, alphabet, L, S, block_size, learning_rate, n_steps)
    # print("After:", alphabet)
    #
    # mse_all, entropy_all = jax.lax.map(partial(evaluate, alphabet, L, S, corrections), samples)
    # mse_mean = jnp.mean(mse_all).item()
    # mse_std = jnp.std(mse_all).item()
    # entropy_mean = jnp.mean(entropy_all).item()
    # entropy_std = jnp.std(entropy_all).item()
    # print(f"After: {mse_mean = :.4f} ({mse_std:.3f}), {entropy_mean = :.4f} ({entropy_std:.3f})")


if __name__ == "__main__":
    L = 16
    S = 6
    block_size = 2**10
    learning_rate = 1e-2
    n_steps = 2**10
    main(L, S, block_size, learning_rate, n_steps)
