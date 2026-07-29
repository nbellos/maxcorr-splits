import numpy as np

from solvers import kl_refine


def _repair_balance(s, C):
    s = s.copy().astype(np.int8)
    g = C @ s
    while int(s.sum()) != 0:
        majority = 1 if s.sum() > 0 else -1
        idx = np.where(s == majority)[0]
        delta = 4.0 * (np.diag(C)[idx] - s[idx] * g[idx])
        i = idx[np.argmin(delta)]
        g = g - 2 * s[i] * C[:, i]
        s[i] = -s[i]
    return s


def _spins_from_column(col):
    s = np.sign(col).astype(np.int8)
    s[s == 0] = 1
    return s


def _sb_agents_finals(C, X, lam, agents, mode, heated, seed=None):
    import torch
    import simulated_bifurcation as sb

    n = C.shape[0]
    M = C + lam * np.ones((n, n))
    if seed is not None:
        torch.manual_seed(seed)
    matrix = torch.tensor(M, dtype=torch.float32)
    result, _ = sb.minimize(
        matrix, domain="spin", agents=agents, mode=mode, heated=heated,
        best_only=False, verbose=False,
    )
    arr = result.numpy()
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[0] != n and arr.shape[1] == n:
        arr = arr.T

    finals = []
    for a in range(arr.shape[1]):
        s = _spins_from_column(arr[:, a])
        s = _repair_balance(s, C)
        s = kl_refine(C, s, X)
        finals.append(s)
    return finals


def sb_split(C, X=None, lam=None, agents=128, mode="discrete", heated=False, seed=None):
    n = C.shape[0]
    if lam is None:
        lam = float(np.mean(np.abs(C)))
    finals = _sb_agents_finals(C, X, lam, agents, mode, heated, seed)

    if X is not None:
        from objective import precompute_P, rho_exact
        pstats = precompute_P(X)
        return max(finals, key=lambda s: rho_exact(X, s, pstats))

    from objective import surrogate
    return max(finals, key=lambda s: -surrogate(C, s))


def sb_library(C, X, cfg, rng):
    n = C.shape[0]
    sb_cfg = cfg.get("sb", {})
    lam_grid = sb_cfg.get("lam_grid", [0.25, 0.5, 1.0, 2.0])
    agents = sb_cfg.get("agents", 128)
    mode = sb_cfg.get("mode", "discrete")
    heated = sb_cfg.get("heated", False)
    mean_abs = float(np.mean(np.abs(C)))

    seen = set()
    candidates = []
    for lam_mult in lam_grid:
        lam = lam_mult * mean_abs
        seed = int(rng.integers(0, 2**31 - 1))
        for s in _sb_agents_finals(C, X, lam, agents, mode, heated, seed):
            key = tuple(s.tolist())
            mirror = tuple((-s).tolist())
            if key in seen or mirror in seen:
                continue
            seen.add(key)
            candidates.append(s)
    return candidates
