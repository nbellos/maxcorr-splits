import numpy as np

from objective import precompute_P, rho_exact_from_D


def random_balanced(n, rng):
    s = np.ones(n, dtype=np.int8)
    s[rng.choice(n, size=n // 2, replace=False)] = -1
    return s


def _median_balance(v):
    s = np.where(v >= np.median(v), 1, -1).astype(np.int8)
    excess = int(s.sum()) // 2
    if excess != 0:
        side = 1 if excess > 0 else -1
        idx = np.where(s == side)[0]
        order = np.argsort(np.abs(v[idx]))
        s[idx[order[:abs(excess)]]] *= -1
    return s


def _projected(C):
    n = C.shape[0]
    ones = np.ones((n, 1)) / np.sqrt(n)
    P = np.eye(n) - ones @ ones.T
    return P @ C @ P


def spectral_split(C):
    _, vecs = np.linalg.eigh(_projected(C))
    return _median_balance(vecs[:, 0])


def bottom_eigenspace_starts(C, n_starts, k, rng):
    _, vecs = np.linalg.eigh(_projected(C))
    basis = vecs[:, :k]
    return [_median_balance(basis @ rng.standard_normal(k)) for _ in range(n_starts)]


def pca_balanced_start(C):
    _, vecs = np.linalg.eigh(C)
    load = vecs[:, -1]
    order = np.argsort(-np.abs(load))
    n = len(load)
    s = np.zeros(n, dtype=np.int8)
    cap = n // 2
    counts = {1: 0, -1: 0}
    exposure = 0.0
    for i in order:
        pick = 1 if exposure * load[i] <= 0 else -1
        if counts[pick] >= cap:
            pick = -pick
        s[i] = pick
        counts[pick] += 1
        exposure += pick * load[i]
    return s


def _swap_delta(C, s, g, greens, reds):
    sg = s * g
    base = -4.0 * (sg[greens][:, None] + sg[reds][None, :])
    diag = 4.0 * (np.diag(C)[greens][:, None] + np.diag(C)[reds][None, :])
    cross = -8.0 * C[np.ix_(greens, reds)]
    return base + diag + cross


def kl_pass(C, s, X=None, pstats=None):
    s = s.copy()
    n = len(s)
    g = C @ s
    D = X @ s if X is not None else None
    locked = np.zeros(n, dtype=bool)
    seq = []

    def score():
        if X is not None:
            return rho_exact_from_D(D, pstats)
        return -float(s @ g)

    best_score, best_t = score(), 0
    for t in range(n // 2):
        greens = np.where((s == 1) & ~locked)[0]
        reds = np.where((s == -1) & ~locked)[0]
        if len(greens) == 0 or len(reds) == 0:
            break
        delta = _swap_delta(C, s, g, greens, reds)
        bi = np.unravel_index(np.argmin(delta), delta.shape)
        i, j = greens[bi[0]], reds[bi[1]]
        s[i], s[j] = -1, 1
        g = g + 2 * (C[:, j] - C[:, i])
        if D is not None:
            D = D + 2 * (X[:, j] - X[:, i])
        locked[i] = locked[j] = True
        seq.append((i, j))
        sc = score()
        if sc > best_score + 1e-12:
            best_score, best_t = sc, t + 1
    for (i, j) in reversed(seq[best_t:]):
        s[i], s[j] = 1, -1
    return s, best_score, best_t


def kl_refine(C, s, X=None, max_passes=60):
    pstats = precompute_P(X) if X is not None else None
    for _ in range(max_passes):
        s, _, moved = kl_pass(C, s, X, pstats)
        if moved == 0:
            break
    return s


def beam_kl(C, s, X=None, H=5, K=2, B=3, max_rounds=40):
    pstats = precompute_P(X) if X is not None else None

    def make_node(sv):
        return {
            "s": sv.copy(),
            "g": C @ sv,
            "D": X @ sv if X is not None else None,
            "locked": np.zeros(len(sv), dtype=bool),
        }

    def node_score(nd):
        if X is not None:
            return rho_exact_from_D(nd["D"], pstats)
        return -float(nd["s"] @ nd["g"])

    current = s.copy()
    best_overall = node_score(make_node(current))
    for _ in range(max_rounds):
        beam = [make_node(current)]
        best_state, best_score = None, best_overall
        for _ in range(H):
            children = []
            for nd in beam:
                greens = np.where((nd["s"] == 1) & ~nd["locked"])[0]
                reds = np.where((nd["s"] == -1) & ~nd["locked"])[0]
                if len(greens) == 0 or len(reds) == 0:
                    continue
                delta = _swap_delta(C, nd["s"], nd["g"], greens, reds)
                flat = np.argsort(delta, axis=None)[:K]
                for f in flat:
                    bi = np.unravel_index(f, delta.shape)
                    i, j = greens[bi[0]], reds[bi[1]]
                    child = {
                        "s": nd["s"].copy(),
                        "g": nd["g"] + 2 * (C[:, j] - C[:, i]),
                        "D": None if nd["D"] is None else nd["D"] + 2 * (X[:, j] - X[:, i]),
                        "locked": nd["locked"].copy(),
                    }
                    child["s"][i], child["s"][j] = -1, 1
                    child["locked"][i] = child["locked"][j] = True
                    children.append(child)
            if not children:
                break
            children.sort(key=node_score, reverse=True)
            beam = children[:B]
            if node_score(beam[0]) > best_score + 1e-12:
                best_score, best_state = node_score(beam[0]), beam[0]["s"].copy()
        if best_state is None:
            break
        current, best_overall = best_state, best_score
    return kl_refine(C, current, X)


def generate_starts(C, cfg, rng):
    starts = [("spectral", spectral_split(C)), ("pca_balanced", pca_balanced_start(C))]
    starts += [
        (f"eigenspace_{i}", s)
        for i, s in enumerate(
            bottom_eigenspace_starts(C, cfg.get("n_eigenspace_starts", 10), k=5, rng=rng)
        )
    ]
    starts += [
        (f"random_{i}", random_balanced(C.shape[0], rng))
        for i in range(cfg.get("n_random_starts", 20))
    ]
    return starts


def refine(C, s, X=None, cfg=None):
    cfg = cfg or {}
    if cfg.get("use_beam", True):
        b = cfg.get("beam", {})
        return beam_kl(C, s, X, H=b.get("H", 5), K=b.get("K", 2), B=b.get("B", 3))
    return kl_refine(C, s, X, max_passes=cfg.get("kl_max_passes", 60))