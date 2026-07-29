import sys
import time

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")

from preprocessing import load_returns, chrono_split, vol_normalize
from objective import precompute_P, rho_exact
from solvers import spectral_split, kl_refine
from bruteforce_check import brute_force_optimum
from sb_solver import _sb_agents_finals

TOLERANCE = 1e-6


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    df = load_returns(cfg["data_path"], cfg["data_type"], cfg["min_obs"])
    X = df.to_numpy()
    X_tr, X_va = chrono_split(X, cfg["train_fraction"])
    X_tr, _ = vol_normalize(X_tr, X_va)
    C = np.corrcoef(X_tr, rowvar=False)
    pst = precompute_P(X_tr)

    optimum, _, n_combos = brute_force_optimum(X_tr, pst)
    rho_kl = rho_exact(X_tr, kl_refine(C, spectral_split(C), X_tr), pst)
    print(f"exact brute-force optimum over {n_combos} balanced splits = {optimum:.6f}")
    print(f"single spectral+KL run                                   = {rho_kl:.6f}")

    sb_cfg = cfg.get("sb", {})
    lam_grid = sb_cfg.get("lam_grid", [0.25, 0.5, 1.0, 2.0])
    agents = sb_cfg.get("agents", 128)
    mode = sb_cfg.get("mode", "discrete")
    heated = sb_cfg.get("heated", False)
    repeats = sb_cfg.get("repeats", 5)
    mean_abs = float(np.mean(np.abs(C)))

    rows = []
    for lam_mult in lam_grid:
        lam = lam_mult * mean_abs
        for rep in range(repeats):
            t0 = time.perf_counter()
            finals = _sb_agents_finals(C, X_tr, lam, agents, mode, heated, seed=rep)
            elapsed = time.perf_counter() - t0
            rhos = np.array([rho_exact(X_tr, s, pst) for s in finals])
            best_rho = float(rhos.max())
            n_at_optimum = int((np.abs(rhos - optimum) < TOLERANCE).sum())
            rows.append({
                "lam_mult": lam_mult,
                "repeat": rep,
                "n_agents": len(finals),
                "best_rho_train": best_rho,
                "hits_optimum": bool(abs(best_rho - optimum) < TOLERANCE),
                "n_agents_at_optimum": n_at_optimum,
                "seconds": elapsed,
            })

    tab = pd.DataFrame(rows)
    print(tab.round(6).to_string(index=False))

    summary = tab.groupby("lam_mult").agg(
        best_rho_train=("best_rho_train", "max"),
        hit_rate=("hits_optimum", "mean"),
        mean_agents_at_optimum=("n_agents_at_optimum", "mean"),
        mean_seconds=("seconds", "mean"),
    )
    print("\nper-lam summary:")
    print(summary.round(4).to_string())

    print(f"\nreference: spectral+KL single run = {rho_kl:.6f}")
    print(f"reference: brute force optimum    = {optimum:.6f}")

    tab.to_csv("results/tables/sb_benchmark.csv", index=False)


if __name__ == "__main__":
    main()
