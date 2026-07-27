import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, "scripts")

from preprocessing import load_returns, chrono_split, vol_normalize, residualize_market, estimate_C
from split_library import build_library
from clustering import validate_split
from consensus import consensus_over_library
from walkforward import run_walkforward


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    rng = np.random.default_rng(cfg["seed"])

    df = load_returns(cfg["data_path"], cfg["data_type"], cfg["min_obs"])
    names = list(df.columns)
    print(f"loaded {len(names)} assets x {len(df)} observations, {df.index[0].date()} -> {df.index[-1].date()}")

    print("\n=== part A: walk-forward persistence ===")
    rows, zs, summary, last = run_walkforward(df, cfg, rng)
    wf = pd.DataFrame(rows)
    print(wf.round(4).to_string(index=False))
    print(f"\nyears={summary['n_years']}  mean_z={summary['mean_z']:+.2f}  "
          f"z>2: {summary['years_above_2']}/{summary['n_years']}  "
          f"stouffer_Z={summary['stouffer_Z']:+.2f}")
    wf.to_csv("results/tables/walkforward.csv", index=False)
    np.save("results/tables/last_year_null.npy", last[1])

    print("\n=== part B: split library and clustering validation ===")
    X = df.to_numpy()
    X_tr, X_va = chrono_split(X, cfg["train_fraction"])
    X_tr, X_va = vol_normalize(X_tr, X_va)
    if cfg.get("residualize_market", False):
        X_tr, X_va = residualize_market(X_tr), residualize_market(X_va)
    C = estimate_C(X_tr, cfg["corr_method"])

    library, _ = build_library(X_tr, X_va, C, cfg, rng)
    lib = pd.DataFrame([{k: e[k] for k in ("name", "rho_train", "rho_valid", "v")} for e in library])
    print(lib.round(4).to_string(index=False))
    lib.to_csv("results/tables/library.csv", index=False)

    best = library[0]
    green = [names[i] for i in np.where(best["s"] == 1)[0]]
    red = [names[i] for i in np.where(best["s"] == -1)[0]]
    print(f"\nbest split  rho_train={best['rho_train']:.4f}  rho_valid={best['rho_valid']:.4f}")
    print("green:", ", ".join(green))
    print("red  :", ", ".join(red))
    np.save("results/tables/best_split.npy", best["s"])

    results, _ = validate_split(
        X_tr, best["s"], cfg["K_range"], method=cfg["cluster_method"],
        n_pcs=cfg["n_global_pcs"], crossfit_iters=cfg["crossfit_iters"],
    )
    ksel = pd.DataFrame([{
        "K": K,
        "Rep_raw": r["raw"]["Rep"],
        "Sep_row_raw": r["raw"]["Sep_row"],
        "Rep_res": r["residualized"]["Rep"],
        "Sep_row_res": r["residualized"]["Sep_row"],
        "sizesA": "/".join(map(str, r["sizesA"])),
        "sizesB": "/".join(map(str, r["sizesB"])),
    } for K, r in results.items()]).set_index("K")
    print("\nK-selection:")
    print(ksel.round(3).to_string())
    ksel.to_csv("results/tables/k_selection.csv")

    K_best = max(results, key=lambda K: results[K]["residualized"]["Rep"])
    np.save("results/tables/R_matched.npy", results[K_best]["R_matched"])
    np.save("results/tables/R_res_matched.npy", results[K_best]["R_res_matched"])

    cons = consensus_over_library(X_tr, library, K_best,
                                  method=cfg["cluster_method"], n_pcs=cfg["n_global_pcs"])
    stab = pd.DataFrame({"asset": names, "cluster": cons["labels"], "confidence": cons["confidence"]})
    stab = stab.sort_values("confidence")
    print(f"\nconsensus K={K_best}; least stable names:")
    print(stab.head(8).round(3).to_string(index=False))
    stab.to_csv("results/tables/consensus.csv", index=False)


if __name__ == "__main__":
    main()