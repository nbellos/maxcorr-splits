import numpy as np

from objective import precompute_P, rho_exact_from_D, rho_exact


def label_permutation_null(X, s, n_samples, rng):
    n = X.shape[1]
    pstats = precompute_P(X)
    out = np.empty(n_samples)
    for i in range(n_samples):
        perm = rng.permutation(n)
        D = X[:, perm] @ s
        out[i] = rho_exact_from_D(D, pstats)
    return out


def block_bootstrap_null(X, s, n_samples, block_len, rng):
    T = X.shape[0]
    n_blocks = int(np.ceil(T / block_len))
    starts_max = T - block_len
    out = np.empty(n_samples)
    for i in range(n_samples):
        starts = rng.integers(0, starts_max + 1, size=n_blocks)
        idx = np.concatenate([np.arange(st, st + block_len) for st in starts])[:T]
        Xb = X[idx]
        out[i] = rho_exact(Xb, s)
    return out


def extra_nulls(X, s, n_samples, block_len, rng):
    return {
        "label_permutation": label_permutation_null(X, s, n_samples, rng),
        "block_bootstrap": block_bootstrap_null(X, s, n_samples, block_len, rng),
    }
