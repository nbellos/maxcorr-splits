import time

import numpy as np

from solvers import spectral_split, kl_refine


def spectral_lower_bound(C):
    n = C.shape[0]
    ones = np.ones((n, 1)) / np.sqrt(n)
    P = np.eye(n) - ones @ ones.T
    lam_min = np.linalg.eigvalsh(P @ C @ P)[1]
    return n * float(lam_min)


def start_convergence(candidates, s_best):
    n = len(s_best)
    hits = sum(1 for c in candidates if abs(int(c["s"] @ s_best)) == n)
    rhos = np.array([c["rho_train"] for c in candidates])
    return {
        "n_starts": len(candidates),
        "frac_at_best": hits / len(candidates) if candidates else float("nan"),
        "rho_train_mean": float(rhos.mean()) if len(rhos) else float("nan"),
        "rho_train_std": float(rhos.std(ddof=1)) if len(rhos) > 1 else 0.0,
        "rho_train_min": float(rhos.min()) if len(rhos) else float("nan"),
        "rho_train_max": float(rhos.max()) if len(rhos) else float("nan"),
    }


def runtime_scaling(C, sizes, rng=None):
    rng = rng or np.random.default_rng()
    n_full = C.shape[0]
    out = []
    for size in sizes:
        size = min(size, n_full)
        size -= size % 2
        idx = rng.choice(n_full, size=size, replace=False)
        Csub = C[np.ix_(idx, idx)]
        t0 = time.perf_counter()
        kl_refine(Csub, spectral_split(Csub))
        out.append({"n": len(idx), "seconds": time.perf_counter() - t0})
    return out
