import numpy as np

from objective import precompute_P, rho_exact, surrogate
from solvers import generate_starts, refine


def build_library(X_train, X_valid, C, cfg, rng):
    n = C.shape[0]
    p_tr = precompute_P(X_train)
    p_va = precompute_P(X_valid)

    candidates = []
    for name, s0 in generate_starts(C, cfg, rng):
        s = refine(C, s0, X_train, cfg)
        candidates.append({
            "name": name,
            "s": s,
            "rho_train": rho_exact(X_train, s, p_tr),
            "rho_valid": rho_exact(X_valid, s, p_va),
            "v": surrogate(C, s),
        })

    candidates.sort(key=lambda c: c["rho_valid"], reverse=True)

    retained = []
    max_ov = cfg.get("max_overlap", 0.6)
    target = cfg.get("library_target", 10)
    for c in candidates:
        if all(abs(int(c["s"] @ r["s"])) / n <= max_ov for r in retained):
            retained.append(c)
        if len(retained) >= target:
            break

    return retained, candidates