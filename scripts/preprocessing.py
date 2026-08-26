import numpy as np
import pandas as pd

def load_returns(path: str, data_type: str = "prices", min_obs: int = 250) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[0], index_col=0).sort_index()
    df = df.dropna(axis=1, thresh=min_obs)
    if data_type == "prices":
        df = df.where(df > 0)          
        df = np.log(df).diff()
    df = df.dropna(how="all").ffill(limit=2).dropna()
    return df

def chrono_split(X: np.ndarray, train_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    cut = int(len(X) * train_fraction)
    return X[:cut], X[cut:]

def vol_normalize(X_train: np.ndarray, X_valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sd = X_train.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    return X_train / sd, X_valid / sd

def residualize_market(X_train: np.ndarray, X_valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0)
    Xc_train = X_train - mean
    C = np.corrcoef(Xc_train, rowvar=False)
    w = np.linalg.eigh(C)[1][:, -1]
    f_train = Xc_train @ w
    beta = (Xc_train.T @ f_train) / (f_train @ f_train)
    resid_train = Xc_train - np.outer(f_train, beta)

    Xc_valid = X_valid - mean
    f_valid = Xc_valid @ w
    resid_valid = Xc_valid - np.outer(f_valid, beta)
    return resid_train, resid_valid

def estimate_C(X_train: np.ndarray, method: str = "sample") -> np.ndarray:
    C = np.corrcoef(X_train, rowvar=False)
    if method == "sample":
        return C
    if method == "mp_clip":
        T, n = X_train.shape
        lam_max = (1 + np.sqrt(n / T)) ** 2
        vals, vecs = np.linalg.eigh(C)
        noise = vals < lam_max
        if noise.any():
            vals[noise] = vals[noise].mean()   
        C = (vecs * vals) @ vecs.T
        d = np.sqrt(np.diag(C))
        C = C / np.outer(d, d)
        np.fill_diagonal(C, 1.0)
        return C
    if method == "ledoit_wolf":
        from sklearn.covariance import ledoit_wolf
        S, _ = ledoit_wolf(X_train)
        d = np.sqrt(np.diag(S))
        return S / np.outer(d, d)
    raise ValueError(f"unknown corr_method: {method}")