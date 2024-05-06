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
    rho_sum = jnp.min(rho)

    return quantized, rho_sum


def dequantize(permuted_alphabet, quantized):
    M, = permuted_alphabet.shape

    def f(state, input_):
        next_state = (M-1) & ((state<<2) | input_)
        output = permuted_alphabet[next_state]
        return next_state, output

    init_state = 0
    _, dequantized = jax.lax.scan(f, init_state, quantized)

    return dequantized
    

def evaluate(permuted_alphabet, A, B, corrections, samples):
    batch_size = 256
    m, n = samples.shape
    assert m % batch_size == 0
    reshaped = jnp.reshape(samples - A @ B, (m // batch_size, batch_size, n))
    vquantize = jax.vmap(quantize, in_axes=(None, None, 0))
    quantized, rho_sums = jax.lax.map(lambda s: vquantize(permuted_alphabet, corrections, s), reshaped)
    vdequantize = jax.vmap(dequantize, in_axes=(None, 0))
    dequantized = jax.lax.map(lambda q: vdequantize(permuted_alphabet, q), quantized).reshape(m, n) + A @ B
    residual = samples - dequantized

    mse = jnp.mean(residual**2)
    rho_sum = jnp.mean(rho_sums)
    loss = mse + rho_sums

    bincount = jnp.bincount(jnp.reshape(quantized, -1), length=4)
    dist = bincount / jnp.sum(bincount)
    entropy = -jnp.sum(dist * jnp.log2(dist))

    return mse, (entropy, quantized.reshape(m, n))


def train(permuted_alphabet, A, B, samples, learning_rate, n_steps):
    block_size = samples.shape[-1]

    @jax.jit
    def train_step(pab, a, b, opt_state):
        grad_fn = jax.value_and_grad(evaluate, argnums=(0, 1, 2), has_aux=True)
        corrections = jnp.eye(block_size)
        (mse, (entropy, quantized)), (grads_pab, grads_a, grads_b) = grad_fn(pab, a, b, corrections, samples)
        grads_pab = jnp.zeros_like(grads_pab)
        grads = grads_pab, grads_a, grads_b
        updates, opt_state = gradient_transform.update(grads, opt_state, (pab, a, b))
        pab, a, b = optax.apply_updates((pab, a, b), updates)

        return mse, entropy, pab, a, b, opt_state, quantized

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
    opt_state = gradient_transform.init((permuted_alphabet, A, B))

    last_quantized = jnp.zeros_like(samples, dtype=int)
    for step in (pbar := trange(n_steps)):
        mse, entropy, permuted_alphabet, A, B, opt_state, quantized = train_step(permuted_alphabet, A, B, opt_state)
        changed = jnp.any(last_quantized != quantized)
        pbar.set_description(f"{mse.item() = :.4f}, {entropy.item() = :.4f}, {changed.item() = }")
        last_quantized = quantized

    return permuted_alphabet, A, B


def main(m, n, rank, learning_rate, n_steps):
    L = 16
    M = 1<<L

    key = jax.random.PRNGKey(42)
    key_perm, key_sample, key_A = jax.random.split(key, num=3)
    permutation = jax.random.permutation(key_perm, M)
    invperm = jnp.argsort(permutation)
    alphabet = jsp.stats.norm.ppf((2*jnp.arange(M)+1)/2/M)
    permuted_alphabet = alphabet[permutation]
    print("Before:", permuted_alphabet[invperm])

    samples = jax.random.normal(key_sample, (m, n))
    A = jax.random.normal(key_A, (m, rank))
    B = jnp.zeros((rank, n))

    corrections = jnp.eye(n)

    mse, (entropy, _) = evaluate(permuted_alphabet, A, B, corrections, samples)
    print(f"Before: {mse = :.4f}, {entropy = :.4f}")

    # fine-tine the alphabet; needs a lot of iterations to hit all symbols
    permuted_alphabet, A, B = train(permuted_alphabet, A, B, samples, learning_rate, n_steps)
    print("Trained alphabet", permuted_alphabet[invperm])

    mse, (entropy, _) = evaluate(permuted_alphabet, A, B, corrections, samples)
    print(f"After: {mse = :.4f}, {entropy = :.4f}")


if __name__ == "__main__":
    m, n, rank = 4096, 4096, 1
    learning_rate = 1e-2
    n_steps = 2**10
    main(m, n, rank, learning_rate, n_steps)
