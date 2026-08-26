import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, "scripts")

from preprocessing import load_returns, chrono_split, vol_normalize, estimate_C
from split_library import build_library


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    df = load_returns(cfg["data_path"], cfg["data_type"], cfg["min_obs"])
    X = df.to_numpy()
    X_tr, X_va = chrono_split(X, cfg["train_fraction"])
    X_tr, X_va = vol_normalize(X_tr, X_va)

    T, n = X_tr.shape
    mp_bound = (1.0 + np.sqrt(n / T)) ** 2
    C_sample = np.corrcoef(X_tr, rowvar=False)
    eigvals = np.linalg.eigvalsh(C_sample)
    n_below = int((eigvals < mp_bound).sum())
    print(f"Marchenko-Pastur bound (1+sqrt(n/T))^2 = {mp_bound:.4f}")
    print(f"eigenvalues below bound: {n_below}/{n}")

    s_sample = None
    rows = []
    for method in ("sample", "mp_clip", "ledoit_wolf"):
        C = estimate_C(X_tr, method)
        library, _ = build_library(X_tr, X_va, C, cfg, np.random.default_rng(cfg["seed"]))
        best = library[0]
        s = best["s"]
        overlap = float(abs(int(s @ s_sample))) / n if s_sample is not None else 1.0
        if method == "sample":
            s_sample = s
        rows.append({
            "corr_method": method,
            "rho_train": best["rho_train"],
            "rho_valid": best["rho_valid"],
            "v": best["v"],
            "differs_from_sample": method != "sample" and overlap < 1.0,
            "overlap_with_sample": overlap,
        })

    tab = pd.DataFrame(rows)
    print("\n" + tab.round(4).to_string(index=False))
    tab.to_csv("results/tables/denoise_comparison.csv", index=False)


if __name__ == "__main__":
    main()
