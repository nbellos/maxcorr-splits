import numpy as np

from objective import precompute_P, rho_exact, rho_exact_batch
from solvers import kl_refine, random_balanced, spectral_split
from extra_nulls import extra_nulls as compute_extra_nulls


def yearly_blocks(index):
    years = index.year
    return [(str(y), np.where(years == y)[0]) for y in np.unique(years) if (years == y).sum() >= 100]


def optimize_year(X_year, n_restarts, rng):
    sd = X_year.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    Xn = X_year / sd
    C = np.corrcoef(Xn, rowvar=False)
    best_s, best_rho = None, -np.inf
    starts = [spectral_split(C)] + [random_balanced(C.shape[0], rng) for _ in range(n_restarts)]
    for s0 in starts:
        s = kl_refine(C, s0, Xn)
        r = rho_exact(Xn, s)
        if r > best_rho:
            best_s, best_rho = s, r
    return best_s, best_rho


def run_walkforward(df, cfg, rng):
    X_all = df.to_numpy()
    blocks = yearly_blocks(df.index)
    n_mc = cfg.get("mc_samples", 1000)
    n_restarts = cfg.get("n_random_starts", 20)

    rows = []
    zs = []
    last = None
    for (lab_prev, idx_prev), (lab_next, idx_next) in zip(blocks[:-1], blocks[1:]):
        s_opt, rho_in = optimize_year(X_all[idx_prev], n_restarts, rng)

        Xn = X_all[idx_next]
        sd = Xn.std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        Xn = Xn / sd
        pst = precompute_P(Xn)
        rho_next = rho_exact(Xn, s_opt, pst)

        S = np.stack([random_balanced(Xn.shape[1], rng) for _ in range(n_mc)], axis=1)
        null = rho_exact_batch(Xn, S, pst)
        mu, sd_null = float(null.mean()), float(null.std(ddof=1))
        z = (rho_next - mu) / sd_null if sd_null > 0 else np.nan
        pct = float((null < rho_next).mean())

        row = {
            "train_year": lab_prev,
            "test_year": lab_next,
            "rho_train": rho_in,
            "rho_next": rho_next,
            "null_mean": mu,
            "null_sd": sd_null,
            "z": z,
            "percentile": pct,
        }
        if cfg.get("extra_nulls", False):
            block_len = cfg.get("block_len", 21)
            for null_name, null_vals in compute_extra_nulls(Xn, s_opt, n_mc, block_len, rng).items():
                nm, nsd = float(null_vals.mean()), float(null_vals.std(ddof=1))
                row[f"{null_name}_z"] = (rho_next - nm) / nsd if nsd > 0 else np.nan
                row[f"{null_name}_percentile"] = float((null_vals < rho_next).mean())
        rows.append(row)
        zs.append(z)
        last = (lab_next, null, rho_next)

    zs = np.array(zs)
    pcts = np.array([r["percentile"] for r in rows])
    summary = {
        "n_years": len(zs),
        "years_above_95": int((pcts >= 0.95).sum()),
        "min_percentile": float(pcts.min()),
        "mean_z": float(zs.mean()),
        "years_above_2": int((zs > 2).sum()),
        "stouffer_Z": float(zs.sum() / np.sqrt(len(zs))),
    }
    return rows, zs, summary, last