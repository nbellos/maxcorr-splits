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
from objective import precompute_P, rho_exact, surrogate, check_balanced
from plotting import plot_walkforward_persistence, plot_correlation_heatmap
from cluster_null import cluster_null_scores

WF_BASE_COLS = ["train_year", "test_year", "rho_train", "rho_next", "null_mean", "null_sd", "z", "percentile"]

ASSET_CLASSES = {
    "EURUSD": "fx", "GBPUSD": "fx", "USDJPY": "fx", "USDCHF": "fx", "AUDUSD": "fx", "USDCAD": "fx",
    "SPX": "equity", "NDX": "equity", "DAX": "equity", "UKX": "equity", "IBEX": "equity",
    "CAC": "equity", "FTSEMIB": "equity", "NKY": "equity", "HSI": "equity",
    "GOLD": "commodity", "SILVER": "commodity", "BRENT": "commodity", "WTI": "commodity", "NATGAS": "commodity",
}


def class_balance_table(X, s, names, taxonomy):
    classes = sorted(set(taxonomy.get(nm, "unclassified") for nm in names))
    rows = []
    for cls in classes:
        idx = [i for i, nm in enumerate(names) if taxonomy.get(nm, "unclassified") == cls]
        gi = [i for i in idx if s[i] == 1]
        ri = [i for i in idx if s[i] == -1]
        if gi and ri:
            a = X[:, gi].sum(axis=1)
            b = X[:, ri].sum(axis=1)
            pearson = float(np.corrcoef(a, b)[0, 1])
        else:
            pearson = float("nan")
        rows.append({"class": cls, "n_green": len(gi), "n_red": len(ri), "pearson": pearson})

    a_total = X[:, s == 1].sum(axis=1)
    b_total = X[:, s == -1].sum(axis=1)
    rows.insert(0, {
        "class": "total",
        "n_green": int((s == 1).sum()),
        "n_red": int((s == -1).sum()),
        "pearson": float(np.corrcoef(a_total, b_total)[0, 1]),
    })
    return pd.DataFrame(rows)


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
    print(f"\nyears={summary['n_years']}  "
          f"above 95th pct: {summary['years_above_95']}/{summary['n_years']}  "
          f"worst percentile={summary['min_percentile']:.3f}  "
          f"(mean_z={summary['mean_z']:+.2f}, stouffer_Z={summary['stouffer_Z']:+.2f})")
    wf[WF_BASE_COLS].to_csv("results/tables/walkforward.csv", index=False)
    if cfg.get("extra_nulls", False):
        wf.to_csv("results/tables/walkforward_extra_nulls.csv", index=False)
    np.save("results/tables/last_year_null.npy", last[1])
    plot_walkforward_persistence(rows, summary, "results/figures/walkforward_persistence.png")

    print("\n=== part B: split library and clustering validation ===")
    X = df.to_numpy()
    X_tr, X_va = chrono_split(X, cfg["train_fraction"])
    X_tr, X_va = vol_normalize(X_tr, X_va)
    if cfg.get("residualize_market", False):
        X_tr, X_va = residualize_market(X_tr, X_va)
    C = estimate_C(X_tr, cfg["corr_method"])

    if cfg.get("solver_method", "kl") == "sb" and cfg.get("sb", {}).get("enabled", False):
        from sb_solver import sb_split, sb_library
        sb_cfg = cfg["sb"]
        p_tr, p_va = precompute_P(X_tr), precompute_P(X_va)
        if sb_cfg.get("use_as_library", False):
            cand_s = sb_library(C, X_tr, cfg, rng)
        else:
            cand_s = [sb_split(C, X_tr, agents=sb_cfg.get("agents", 128),
                                mode=sb_cfg.get("mode", "discrete"),
                                heated=sb_cfg.get("heated", False))]
        candidates = [{
            "name": f"sb_{i}", "s": s,
            "rho_train": rho_exact(X_tr, s, p_tr),
            "rho_valid": rho_exact(X_va, s, p_va),
            "v": surrogate(C, s),
        } for i, s in enumerate(cand_s)]
        candidates.sort(key=lambda c: c["rho_valid"], reverse=True)
        library = candidates[:cfg.get("library_target", 10)]
    else:
        library, candidates = build_library(X_tr, X_va, C, cfg, rng)
    lib = pd.DataFrame([{k: e[k] for k in ("name", "rho_train", "rho_valid", "v")} for e in library])
    print(lib.round(4).to_string(index=False))
    lib.to_csv("results/tables/library.csv", index=False)

    best = library[0]
    check_balanced(best["s"])
    green = [names[i] for i in np.where(best["s"] == 1)[0]]
    red = [names[i] for i in np.where(best["s"] == -1)[0]]
    print(f"\nbest split  rho_train={best['rho_train']:.4f}  rho_valid={best['rho_valid']:.4f}")
    print("green:", ", ".join(green))
    print("red  :", ", ".join(red))
    np.save("results/tables/best_split.npy", best["s"])
    plot_correlation_heatmap(C, best["s"], names, best["rho_train"], best["rho_valid"],
                              "results/figures/correlation_heatmap.png")

    classbal = class_balance_table(X_tr, best["s"], names, ASSET_CLASSES)
    print("\nclass balance (best split):")
    print(classbal.round(4).to_string(index=False))
    classbal.to_csv("results/tables/class_balance.csv", index=False)

    if cfg.get("diagnostics", False):
        from diagnostics import spectral_lower_bound, start_convergence, runtime_scaling
        lb = spectral_lower_bound(C)
        print(f"\ndiagnostics: spectral lower bound (n*lambda_min) = {lb:.4f}  "
              f"best surrogate v = {best['v']:.4f}  gap = {best['v'] - lb:.4f}")
        conv = start_convergence(candidates, best["s"])
        print(f"diagnostics: {conv['n_starts']} starts, "
              f"{conv['frac_at_best']:.1%} landed on best split, "
              f"rho_train range [{conv['rho_train_min']:.4f}, {conv['rho_train_max']:.4f}] "
              f"mean={conv['rho_train_mean']:.4f} std={conv['rho_train_std']:.4f}")
        sizes = sorted({min(len(names), s) for s in (20, 40, 80, 120, len(names))})
        scaling = runtime_scaling(C, sizes, rng)
        print("diagnostics: runtime scaling", scaling)

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

    null_scores = cluster_null_scores(X_tr, cfg["K_range"], cfg.get("cluster_null_samples", 50),
                                       cfg["cluster_method"], cfg["n_global_pcs"], cfg["crossfit_iters"], rng)
    null_rows = []
    for K in cfg["K_range"]:
        rep_null = null_scores[K]["Rep_res"]
        rep_obs = results[K]["residualized"]["Rep"]
        mu, sd = float(rep_null.mean()), float(rep_null.std(ddof=1))
        z = (rep_obs - mu) / sd if sd > 0 else float("nan")
        pct = float((rep_null < rep_obs).mean())
        null_rows.append({"K": K, "Rep_res_obs": rep_obs, "null_mean": mu, "null_sd": sd, "z": z, "percentile": pct})
    cluster_null_tab = pd.DataFrame(null_rows).set_index("K")
    print("\nPart B null check (best split vs random splits through the same clustering pipeline):")
    print(cluster_null_tab.round(3).to_string())
    cluster_null_tab.to_csv("results/tables/cluster_null.csv")

    kw = cfg.get("k_selection", {})
    alpha, beta, gamma = kw.get("alpha", 1.0), kw.get("beta", 0.5), kw.get("gamma", 1.0)
    cons_by_K, scores = {}, {}
    for K, r in results.items():
        cons_by_K[K] = consensus_over_library(X_tr, library, K, method=cfg["cluster_method"], n_pcs=cfg["n_global_pcs"])
        sizes = np.concatenate([r["sizesA"], r["sizesB"]]).astype(float)
        size_penalty = float(sizes.std() / sizes.mean()) if sizes.mean() > 0 else 0.0
        stability = float(cons_by_K[K]["confidence"].mean())
        scores[K] = (r["residualized"]["Rep"] + alpha * r["residualized"]["Sep_row"]
                     + gamma * stability - beta * size_penalty)

    kscore = pd.DataFrame({"K": list(scores.keys()), "score": list(scores.values())}).set_index("K")
    print("\nK-selection composite score (Rep_res + a*Sep_res + g*Stability - b*SizePenalty):")
    print(kscore.round(3).to_string())
    kscore.to_csv("results/tables/k_selection_score.csv")

    K_best = max(scores, key=scores.get)
    np.save("results/tables/R_matched.npy", results[K_best]["R_matched"])
    np.save("results/tables/R_res_matched.npy", results[K_best]["R_res_matched"])

    cons = cons_by_K[K_best]
    stab = pd.DataFrame({"asset": names, "cluster": cons["labels"], "confidence": cons["confidence"]})
    stab = stab.sort_values("confidence")
    print(f"\nconsensus K={K_best}; least stable names:")
    print(stab.head(8).round(3).to_string(index=False))
    stab.to_csv("results/tables/consensus.csv", index=False)


if __name__ == "__main__":
    main()