# Implements the random permutation trellis encoder
# https://ee.stanford.edu/~gray/trellis.pdf

from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import optax
from tqdm import trange


def quantize(permuted_alphabet, corrections, samples, diag_only=True):
    M, = permuted_alphabet.shape
    L = M.bit_length() - 1
    length = samples.shape[0]

    assert len(samples.shape) == 1, samples.shape
    assert len(corrections.shape) == 2, corrections.shape
    assert length == corrections.shape[0] == corrections.shape[1], (length, corrections.shape)

    # checkify.check(jnp.allclose(jnp.tril(corrections), corrections), "corrections must be lower triangular")

    def viterbi_step(carry, index):
        rho, prev_states = carry
        prefixes = jnp.arange(M>>2)
        discarded = jnp.arange(1<<2)
        last_state = (discarded << (L-2)) | prefixes[..., jnp.newaxis]

        optimal_discarded = jnp.argmin(rho[last_state], axis=-1)
        prev_state = (optimal_discarded << (L-2)) | prefixes
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

            negative_one, zero, error = jax.lax.while_loop(end, make_correction, (index, jnp.arange(M), jnp.zeros((M,))))
            # checkify.check(negative_one == -1, "negative_one should be -1, got {}", negative_one)
            # checkify.check(zero == 0, "zero should be 0, got {}", zero)
            additive_loss = error**2

        updated_rho = jnp.repeat(rho[prev_state], 1<<2) + additive_loss

        return (updated_rho, updated_prev_states), None

    rho = jnp.full((M,), jnp.inf)
    rho = rho.at[0].set(0)  # initial state is zero (TODO: Tail-Biting)
    prev_states = jnp.empty((length, M>>2), dtype=int)
    (rho, prev_states), _ = jax.lax.scan(viterbi_step, (rho, prev_states), jnp.arange(length))

    def backtrack_input(state, prev_state):
        return prev_state[state>>2], state & 0b11

    last_state = jnp.argmin(rho)
    zero, quantized_rev = jax.lax.scan(backtrack_input, last_state, jnp.flipud(prev_states))
    # checkify.check(zero == 0, "zero should be 0, got {}", zero)
    quantized = jnp.flipud(quantized_rev)
    expected_loss = jnp.min(rho)

    return quantized, expected_loss


def dequantize(permuted_alphabet, quantized):
    M, = permuted_alphabet.shape

    def f(state, input_):
        next_state = (M-1) & ((state<<2) | input_)
        output = permuted_alphabet[next_state]
        return next_state, output

    init_state = 0
    _, dequantized = jax.lax.scan(f, init_state, quantized)

    return dequantized
    

def evaluate(permuted_alphabet, corrections, samples):
    quantized, _ = quantize(permuted_alphabet, corrections, samples)
    dequantized = dequantize(permuted_alphabet, quantized)
    residual = samples - dequantized
    mse = jnp.mean(residual**2)

    bincount = jnp.bincount(quantized, length=4)
    dist = bincount / jnp.sum(bincount)
    entropy = -jnp.sum(dist * jnp.log2(dist))

    return mse, entropy


def train(key, permuted_alphabet, block_size, learning_rate, n_steps):

    @jax.jit
    def train_step(key_step, pab, opt_state):
        grad_fn = jax.value_and_grad(evaluate, has_aux=True)
        samples = jax.random.normal(key_step, (block_size,))
        corrections = jnp.eye(block_size)
        (mse, entropy), grads = grad_fn(pab, corrections, samples)
        updates, opt_state = gradient_transform.update(grads, opt_state, pab)
        pab = optax.apply_updates(pab, updates)

        return mse, entropy, pab, opt_state

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
    opt_state = gradient_transform.init(permuted_alphabet)

    for step in (pbar := trange(n_steps)):
        key_step = jax.random.fold_in(key, step)
        mse, entropy, permuted_alphabet, opt_state = train_step(key_step, permuted_alphabet, opt_state)
        pbar.set_description(f"{mse.item() = :.4f}, {entropy.item() = :.4f}")

    return permuted_alphabet


def main(block_size, learning_rate, n_steps):
    L = 16
    M = 1<<L

    import numpy as np

    key = jax.random.PRNGKey(42)
    key_perm, key_train, key_test = jax.random.split(key, num=3)
    permutation = jax.random.permutation(key_perm, M)
    permutation = jnp.arange(M)

    # A New Class of Invertible Mappings, Theorem 3
    # permutation = (permutation + ((permutation*permutation) | 0b10101)) & (M - 1)
    # print(M, len(set(np.array(permutation))))

    # https://github.com/skeeto/hash-prospector?tab=readme-ov-file#16-bit-hashes
    # permutation ^= permutation >> 7
    # permutation &= M - 1
    # permutation *= 0x2993
    # permutation &= M - 1
    # permutation ^= permutation >> 5
    # permutation &= M - 1
    # permutation *= 0xe877
    # permutation &= M - 1
    # permutation ^= permutation >> 9
    # permutation &= M - 1
    # permutation *= 0x0235
    # permutation &= M - 1
    # permutation ^= permutation >> 10
    # permutation &= M - 1

    # RC6 function
    permutation = permutation * (2*permutation+1)
    permutation = permutation * 1664525 + 1013904223
    permutation &= M - 1

    print(permutation)
    print(M, len(set(np.array(permutation))))

    invperm = jnp.argsort(permutation)
    alphabet_half = jnp.interp(jnp.linspace(0, 1, M//2),
                               jnp.linspace(0, 1, 64),
                               jsp.stats.norm.ppf(129/256 + jnp.arange(64)/128))
    alphabet = jnp.concatenate([jnp.flipud(alphabet_half), -alphabet_half])

    permuted_alphabet = alphabet[permutation]
    print("Before:", permuted_alphabet[invperm])

    samples = jax.random.normal(key_test, (2**20//block_size, block_size))
    corrections = jnp.eye(block_size)

    mse_all, entropy_all = jax.lax.map(partial(evaluate, permuted_alphabet, corrections), samples)
    mse_mean = jnp.mean(mse_all).item()
    mse_std = jnp.std(mse_all).item()
    entropy_mean = jnp.mean(entropy_all).item()
    entropy_std = jnp.std(entropy_all).item()
    print(f"Before: {mse_mean = :.4f} ({mse_std:.3f}), {entropy_mean = :.4f} ({entropy_std:.3f})")

    # fine-tine the alphabet; needs a lot of iterations to hit all symbols
    permuted_alphabet = train(key_train, permuted_alphabet, block_size, learning_rate, n_steps)
    print("After", permuted_alphabet[invperm])

    mse_all, entropy_all = jax.lax.map(partial(evaluate, permuted_alphabet, corrections), samples)
    mse_mean = jnp.mean(mse_all).item()
    mse_std = jnp.std(mse_all).item()
    entropy_mean = jnp.mean(entropy_all).item()
    entropy_std = jnp.std(entropy_all).item()
    print(f"After: {mse_mean = :.4f} ({mse_std:.3f}), {entropy_mean = :.4f} ({entropy_std:.3f})")


if __name__ == "__main__":
    block_size = 2**10
    learning_rate = 1e-2
    n_steps = 2**22
    main(block_size, learning_rate, n_steps)
