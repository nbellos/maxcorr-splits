import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, SpectralClustering


def corr_distance(Xcols):
    C = np.corrcoef(Xcols, rowvar=False)
    return np.sqrt(np.maximum(2.0 * (1.0 - C), 0.0))


def cluster_half(Xcols, K, method="hierarchical"):
    if method == "hierarchical":
        D = corr_distance(Xcols)
        model = AgglomerativeClustering(n_clusters=K, metric="precomputed", linkage="average")
        return model.fit_predict(D)
    if method == "spectral":
        C = np.corrcoef(Xcols, rowvar=False)
        A = np.abs(C)
        np.fill_diagonal(A, 0.0)
        model = SpectralClustering(n_clusters=K, affinity="precomputed", random_state=0)
        return model.fit_predict(A)
    raise ValueError(f"unknown cluster method: {method}")


def prototypes(Xcols, labels, K):
    protos = np.zeros((Xcols.shape[0], K))
    for k in range(K):
        members = labels == k
        p = Xcols[:, members].mean(axis=1) if members.any() else np.zeros(Xcols.shape[0])
        sd = p.std(ddof=1)
        protos[:, k] = p / sd if sd > 0 else p
    return protos


def cross_half_R(protoA, protoB):
    K = protoA.shape[1]
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.corrcoef(protoA, protoB, rowvar=False)
    return np.nan_to_num(R[:K, K:], nan=-1.0)


def hungarian_match(R):
    rows, cols = linear_sum_assignment(-R)
    return cols, R[:, cols]


def rep_sep_scores(R_matched):
    d = np.diag(R_matched)
    mask = np.eye(len(d), dtype=bool)
    sep_row = d - np.max(np.where(mask, -np.inf, R_matched), axis=1)
    sep_col = d - np.max(np.where(mask, -np.inf, R_matched), axis=0)
    return {
        "Rep": float(d.mean()),
        "Sep_row": float(sep_row.mean()),
        "Sep_col": float(sep_col.mean()),
        "diag": d,
    }


def global_regressors(X, n_pcs):
    Xc = X - X.mean(axis=0)
    P = Xc.sum(axis=1, keepdims=True)
    C = np.corrcoef(Xc, rowvar=False)
    _, vecs = np.linalg.eigh(C)
    pcs = Xc @ vecs[:, -n_pcs:]
    return np.hstack([np.ones((len(X), 1)), P, pcs])


def residualize(protos, G):
    beta, *_ = np.linalg.lstsq(G, protos, rcond=None)
    res = protos - G @ beta
    sd = res.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    return res / sd


def cross_fit(XA, XB, labelsA, labelsB, K, iters=3):
    for _ in range(iters):
        pA, pB = prototypes(XA, labelsA, K), prototypes(XB, labelsB, K)
        with np.errstate(invalid="ignore", divide="ignore"):
            corrA = np.corrcoef(XA.T, pB.T)[:XA.shape[1], XA.shape[1]:]
            corrB = np.corrcoef(XB.T, pA.T)[:XB.shape[1], XB.shape[1]:]
        corrA = np.nan_to_num(corrA, nan=-2.0)
        corrB = np.nan_to_num(corrB, nan=-2.0)
        newA, newB = corrA.argmax(axis=1), corrB.argmax(axis=1)
        if np.array_equal(newA, labelsA) and np.array_equal(newB, labelsB):
            break
        labelsA, labelsB = newA, newB
    return labelsA, labelsB


def validate_split(X, s, K_range, method="hierarchical", n_pcs=2, crossfit_iters=3):
    A, B = np.where(s == 1)[0], np.where(s == -1)[0]
    XA, XB = X[:, A], X[:, B]
    G = global_regressors(X, n_pcs)
    out = {}
    for K in K_range:
        lA = cluster_half(XA, K, method)
        lB = cluster_half(XB, K, method)
        lA, lB = cross_fit(XA, XB, lA, lB, K, iters=crossfit_iters)
        pA, pB = prototypes(XA, lA, K), prototypes(XB, lB, K)
        R = cross_half_R(pA, pB)
        perm, Rm = hungarian_match(R)
        scores = rep_sep_scores(Rm)
        pA_r = residualize(pA, G)
        pB_r = residualize(pB, G)[:, perm]
        R_res = cross_half_R(pA_r, pB_r)
        scores_res = rep_sep_scores(R_res)
        out[K] = {
            "labelsA": lA,
            "labelsB": lB,
            "perm": perm,
            "R_matched": Rm,
            "R_res_matched": R_res,
            "raw": scores,
            "residualized": scores_res,
            "sizesA": np.bincount(lA, minlength=K),
            "sizesB": np.bincount(lB, minlength=K),
        }
    return out, (A, B)