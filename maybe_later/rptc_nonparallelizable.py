# Implements the random permutation trellis encoder
# https://ee.stanford.edu/~gray/trellis.pdf

from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import optax
from tqdm import trange


def ror(x, l, L):
    l %= L
    mask = (1<<L)-1
    rotated = (x<<(L-l)) | (x>>l)
    result = mask & rotated
    return result


def rol(x, l, L):
    l %= L
    mask = (1<<L)-1
    rotated = (x<<l) | (x>>(L-l))
    result = mask & rotated
    return result


def transit(state, input_, M):
    L = M.bit_length() - 1
    # next_state = (M-1) & ((rol(state, input_*(L//4), L)<<2) | input_)
    # next_state = (M-1) & ((state<<2) | input_)
    next_state = (M-1) & (rol(state, 5, L) ^ input_)
    return next_state


def quantize(permuted_alphabet, corrections, samples, diag_only=True):
    M, = permuted_alphabet.shape
    length = samples.shape[0]

    assert len(samples.shape) == 1, samples.shape
    assert len(corrections.shape) == 2, corrections.shape
    assert length == corrections.shape[0] == corrections.shape[1], (length, corrections.shape)

    # checkify.check(jnp.allclose(jnp.tril(corrections), corrections), "corrections must be lower triangular")

    full_state = jnp.arange(M)
    full_state_next = jnp.stack([transit(full_state, input_, M) for input_ in range(1<<2)], axis=-1)

    # histogram = jax.numpy.bincount(full_state_next.reshape(-1), length=M)
    # jax.debug.print("histogram: {} =?= {}", jnp.min(histogram), jnp.max(histogram))

    # divide by 4 gives us the row index, because each row in full_state_next contains 4 elements
    last_state = jnp.argsort(full_state_next, axis=None, kind="stable").reshape(M, 1<<2) >> 2

    # jax.debug.print("full_state_next: {}", full_state_next)
    # jax.debug.print("last_state: {}", last_state)

    def viterbi_step(carry, index):
        rho, prev_states = carry

        prev_state_idx = jnp.argmin(rho[last_state], axis=-1)
        prev_state = jnp.take_along_axis(last_state, prev_state_idx[:, jnp.newaxis], axis=-1).squeeze(-1)
        updated_prev_states = prev_states.at[index].set(prev_state)
        
        if diag_only:
            additive_loss = jnp.sum(corrections[index]**2) * (samples[index] - permuted_alphabet)**2

        else:
            def end(val):
                idx, state, error = val
                return idx >= 0

            def make_correction(val):
                idx, state, error = val
                updated_state = prev_states[idx, state]
                updated_error = error + corrections[index, idx] * (samples[idx] - permuted_alphabet[state])
                return idx-1, updated_state, updated_error

            negative_one, zero, error = jax.lax.while_loop(end, make_correction, (index, jnp.arange(M), jnp.zeros((M,))))
            # checkify.check(negative_one == -1, "negative_one should be -1, got {}", negative_one)
            # checkify.check(zero == 0, "zero should be 0, got {}", zero)
            additive_loss = error**2

        updated_rho = rho[prev_state] + additive_loss

        return (updated_rho, updated_prev_states), None

    rho = jnp.full((M,), jnp.inf)
    # TODO: FIXME
    init = M // 3
    rho = rho.at[init].set(0)    # init state is zero (TODO: Tail-Biting)
    prev_states = jnp.empty((length, M), dtype=int)
    (rho, prev_states), _ = jax.lax.scan(viterbi_step, (rho, prev_states), jnp.arange(length))

    def backtrack_input(state, prev_state):
        prev = prev_state[state]
        mask = transit(prev, jnp.arange(1<<2), M) == state
        input_ = jnp.flatnonzero(mask, size=1)  # single-element array
        return prev, input_[0]

    last_state = jnp.argmin(rho)
    maybe_init, quantized_rev = jax.lax.scan(backtrack_input, last_state, jnp.flipud(prev_states))
    # checkify.check(maybe_init == init, "maybe_init should be {}, got {}", init, maybe_init)
    quantized = jnp.flipud(quantized_rev)
    expected_loss = jnp.min(rho)

    return quantized, expected_loss


def dequantize(permuted_alphabet, quantized):
    M, = permuted_alphabet.shape
    L = M.bit_length() - 1

    def f(state, input_):
        next_state = transit(state, input_, M)
        output = permuted_alphabet[next_state]
        return next_state, output

    init = M // 3
    _, dequantized = jax.lax.scan(f, init, quantized)

    return dequantized
    

def evaluate(permuted_alphabet, corrections, samples):
    quantized, _ = quantize(permuted_alphabet, corrections, samples)
    dequantized = dequantize(permuted_alphabet, quantized)
    # jax.debug.print("samples: {}", samples)
    # jax.debug.print("quantized: {}", quantized)
    # jax.debug.print("dequantized: {}", dequantized)
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
    L = 8
    M = 1<<L

    key = jax.random.PRNGKey(42)
    key_perm, key_train, key_test = jax.random.split(key, num=3)
    permutation = jax.random.permutation(key_perm, M)
    invperm = jnp.argsort(permutation)
    alphabet = jsp.stats.norm.ppf((2*jnp.arange(M)+1)/2/M)
    permuted_alphabet = alphabet[permutation]
    # print("Before:", permuted_alphabet[invperm])

    samples = jax.random.normal(key_test, (2**20//block_size, block_size))
    corrections = jnp.eye(block_size)

    mse_all, entropy_all = jax.lax.map(partial(evaluate, permuted_alphabet, corrections), samples)
    mse_mean = jnp.mean(mse_all).item()
    mse_std = jnp.std(mse_all).item()
    entropy_mean = jnp.mean(entropy_all).item()
    entropy_std = jnp.std(entropy_all).item()
    print(f"Before: {mse_mean = :.4f} ({mse_std:.3f}), {entropy_mean = :.4f} ({entropy_std:.3f})")

    # fine-tine the alphabet; needs a lot of iterations to hit all symbols
    # permuted_alphabet = train(key_train, permuted_alphabet, block_size, learning_rate, n_steps)
    # print("After", permuted_alphabet[invperm])
    #
    # mse_all, entropy_all = jax.lax.map(partial(evaluate, permuted_alphabet, corrections), samples)
    # mse_mean = jnp.mean(mse_all).item()
    # mse_std = jnp.std(mse_all).item()
    # entropy_mean = jnp.mean(entropy_all).item()
    # entropy_std = jnp.std(entropy_all).item()
    # print(f"After: {mse_mean = :.4f} ({mse_std:.3f}), {entropy_mean = :.4f} ({entropy_std:.3f})")


if __name__ == "__main__":
    block_size = 2**10
    learning_rate = 1e-2
    n_steps = 2**12
    main(block_size, learning_rate, n_steps)
