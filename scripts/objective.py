import numpy as np

def check_balanced(s: np.ndarray) -> None:
    if int(s.sum()) != 0:
        raise ValueError(f"unbalanced sings: sum={int(s.sum())}")
    if not np.all(np.abs(s) == 1):
        raise ValueError("signs must be +1/-1")
    
def surrogate(C: np.ndarray, s:np.ndarray) -> float:
    """variance of the spread s'Cs"""
    return float(s @ C @ s)

def precompute_P(X: np.ndarray) -> dict:
    P = X.sum(axis=1)
    P0 = P - P.mean()
    return {"P0": P0, "q": float(P0 @ P0 / len(P0))}

def rho_exact_from_D(D: np.ndarray, pstats: dict) -> float:
    """Exact corr(SA,SB) from D=X@s"""
    D0 = D - D.mean()
    T = len(D0)
    v = float(D0 @ D0 / T)
    u = float(pstats["P0"] @ D0 / T)
    q = pstats["q"]
    denom_sq = (q+v ** 2 - 4.0 * u * u)
    if denom_sq <= 1e-18:
        return -1.0
    return (q-v) / np.sqrt(denom_sq)

def rho_exact(X: np.ndarray, s: np.ndarray,pstats: dict | None = None) -> float:
    if pstats is None:
        pstats = precompute_P(X)
    return rho_exact_from_D(X @ s,pstats)

def basket_correlation(X: np.ndarray, s: np.ndarray) -> float:
    a = X[:, s > 0].sum(axis=1)
    b = X[:, s < 0].sum(axis=1)
    return float(np.corrcoef(a, b)[0, 1])