import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_walkforward_persistence(rows, summary, path):
    labels = [f"{r['train_year']}\n->{r['test_year']}" for r in rows]
    z = np.array([r["z"] for r in rows])
    colors = ["#2a9d5b" if zi > 2 else ("#e07b39" if zi > 0 else "#c0392b") for zi in z]

    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(rows)), 4.5))
    ax.bar(range(len(z)), z, color=colors)
    ax.axhline(2.0, color="black", linestyle="--", linewidth=1, label="z = 2 (significance)")
    ax.axhline(0.0, color="gray", linestyle="-", linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("z-score (realized vs. random-split null)")
    ax.set_title(
        f"Walk-forward correlation persistence  "
        f"(mean z={summary['mean_z']:+.2f}, Stouffer Z={summary['stouffer_Z']:+.2f}, "
        f"{summary['years_above_2']}/{summary['n_years']} years z>2)"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(C, s, names, rho_train, rho_valid, path):
    order = np.concatenate([np.where(s == 1)[0], np.where(s == -1)[0]])
    n_green = int((s == 1).sum())
    C_ord = C[np.ix_(order, order)]
    labels = [names[i] for i in order]

    fig, ax = plt.subplots(figsize=(max(6, 0.35 * len(names)), max(6, 0.35 * len(names))))
    im = ax.imshow(C_ord, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.axhline(n_green - 0.5, color="black", linewidth=2)
    ax.axvline(n_green - 0.5, color="black", linewidth=2)
    ax.set_title(f"Best split correlation structure\nrho_train={rho_train:.4f}  rho_valid={rho_valid:.4f}", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8, label="correlation")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
