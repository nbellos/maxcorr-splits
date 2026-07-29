import numpy as np

from clustering import validate_split
from solvers import random_balanced


def cluster_null_scores(X, K_range, n_samples, method, n_pcs, crossfit_iters, rng):
    out = {K: {"Rep_res": [], "Sep_row_res": []} for K in K_range}
    n = X.shape[1]
    for _ in range(n_samples):
        s = random_balanced(n, rng)
        res, _ = validate_split(X, s, K_range, method=method, n_pcs=n_pcs, crossfit_iters=crossfit_iters)
        for K in K_range:
            out[K]["Rep_res"].append(res[K]["residualized"]["Rep"])
            out[K]["Sep_row_res"].append(res[K]["residualized"]["Sep_row"])
    return {K: {m: np.array(v) for m, v in d.items()} for K, d in out.items()}
