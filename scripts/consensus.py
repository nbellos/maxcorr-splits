import numpy as np
from sklearn.cluster import AgglomerativeClustering

from clustering import validate_split


def consensus_over_library(X, library, K, method="hierarchical", n_pcs=2):
    n = X.shape[1]
    H = np.zeros((n, n))
    for entry in library:
        res, (A, B) = validate_split(X, entry["s"], [K], method=method, n_pcs=n_pcs)
        r = res[K]
        labels = np.empty(n, dtype=int)
        labels[A] = r["labelsA"]
        labels[B] = r["perm"][r["labelsB"]]
        H += labels[:, None] == labels[None, :]
    H /= len(library)

    model = AgglomerativeClustering(n_clusters=K, metric="precomputed", linkage="average")
    consensus = model.fit_predict(1.0 - H)
    confidence = np.array([H[i, consensus == consensus[i]].mean() for i in range(n)])
    return {"H": H, "labels": consensus, "confidence": confidence}