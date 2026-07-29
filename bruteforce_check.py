import sys
from itertools import combinations

import numpy as np
import yaml

sys.path.insert(0, "scripts")

from preprocessing import load_returns, chrono_split, vol_normalize
from objective import precompute_P, rho_exact, rho_exact_batch
from solvers import spectral_split, kl_refine


def brute_force_optimum(X, pstats):
    n = X.shape[1]
    combos = [c for c in combinations(range(n), n // 2) if 0 in c]
    best_rho, best_s = -2.0, None
    for start in range(0, len(combos), 5000):
        block = combos[start:start + 5000]
        S = -np.ones((n, len(block)))
        for j, c in enumerate(block):
            S[list(c), j] = 1.0
        r = rho_exact_batch(X, S, pstats)
        j = int(np.argmax(r))
        if r[j] > best_rho:
            best_rho, best_s = float(r[j]), S[:, j].copy()
    return best_rho, best_s, len(combos)


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    df = load_returns(cfg["data_path"], cfg["data_type"], cfg["min_obs"])
    X = df.to_numpy()
    X_tr, X_va = chrono_split(X, cfg["train_fraction"])
    X_tr, X_va = vol_normalize(X_tr, X_va)
    pst = precompute_P(X_tr)

    best_rho, best_s, n_combos = brute_force_optimum(X_tr, pst)

    C = np.corrcoef(X_tr, rowvar=False)
    s_kl = kl_refine(C, spectral_split(C), X_tr)
    rho_kl = rho_exact(X_tr, s_kl, pst)

    names = list(df.columns)
    green = [names[i] for i in np.where(best_s == 1)[0]]
    red = [names[i] for i in np.where(best_s == -1)[0]]

    print(f"enumerated {n_combos} balanced splits")
    print(f"brute force optimum rho_train = {best_rho:.6f}")
    print(f"solver (spectral+KL)          = {rho_kl:.6f}")
    print(f"gap = {best_rho - rho_kl:.2e}")
    print("solver matches brute force:", bool(abs(best_rho - rho_kl) < 1e-9))
    print("bf green:", ", ".join(green))
    print("bf red  :", ", ".join(red))
    p_va = precompute_P(X_va)
    print(f"bf split rho_valid = {rho_exact(X_va, best_s, p_va):.6f}")


if __name__ == "__main__":
    main()
