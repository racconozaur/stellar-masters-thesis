import os, pickle, warnings
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from collections import Counter

warnings.filterwarnings("ignore", category=UserWarning)



LBL_NORM = os.path.expanduser("~/stellar-clustering/network/labled-data/labels/label-normalization/labels_entities_normalized.csv")


RANDOM_STATE = 42
EMBED_DIM = 64 
PLOT = True

GRAPH_PKL = os.path.expanduser("~/stellar-clustering/network/LCC/transactions/LCC_G_tx_undirected_weighted.pkl")
SPEC_FN   = os.path.expanduser("transactions/spectral_labels_all_k.csv")





def load_graph(path: str) -> nx.Graph:
    with open(path, "rb") as f:
        G = pickle.load(f)
    if isinstance(G, nx.DiGraph):
        G = nx.Graph(G)
    return G

def affinity_from_graph(G: nx.Graph):
    nodes = list(G.nodes())
    pos = {n:i for i,n in enumerate(nodes)}
    rows, cols, data = [], [], []
    for u, v, d in G.edges(data=True):
        i, j = pos[u], pos[v]
        w = float(d.get("weight", 1.0))
        rows.append(i); cols.append(j); data.append(w)
    n = len(nodes)
    A = sparse.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)
    A.setdiag(0.0)
    A = A.maximum(A.T).tocsr()
    A.sort_indices()
    return A, nodes


print("Loading the graph")
G = load_graph(GRAPH_PKL)
A_full, nodes_full = affinity_from_graph(G)
spec = pd.read_csv(SPEC_FN)

assert "account_id" in spec.columns, "spectral CSV must have 'account_id'."


idx_in_csv = {aid: i for i, aid in enumerate(spec["account_id"].tolist())}
row_idx = [idx_in_csv.get(n, -1) for n in nodes_full]
mask = np.array([i >= 0 for i in row_idx], dtype=bool)

if (~mask).any():
    print(f"Note: {(~mask).sum():,} graph nodes missing from spectral CSV (skipp).")

nodes = [nodes_full[i] for i in np.where(mask)[0]]
A = A_full[np.ix_(np.where(mask)[0], np.where(mask)[0])]
spec_a = spec.iloc[[i for i in row_idx if i >= 0]].reset_index(drop=True)

print("Starting Spectral Embedding")

X = SpectralEmbedding(n_components=EMBED_DIM, affinity="precomputed",
                      random_state=RANDOM_STATE).fit_transform(A)

print("Embedding shape:", X.shape)



print("compute intrinsic metrics for K")
k_cols = sorted([c for c in spec_a.columns if c.startswith("cluster_k")],
                key=lambda s: int(s.split("cluster_k")[-1]))
rows = []
for col in k_cols:
    k = int(col.split("cluster_k")[-1])
    y = spec_a[col].astype(int).to_numpy()
    print(f"Scoring k={k} ...")
    try:
        sil = silhouette_score(X, y, metric="euclidean")
    except Exception:
        sil = np.nan
    try:
        db  = davies_bouldin_score(X, y)
    except Exception:
        db = np.nan
    try:
        ch  = calinski_harabasz_score(X, y)
    except Exception:
        ch = np.nan
    rows.append({"k": k, "silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch})

out = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
out.to_csv("transactions/transaction_spectral_intrinsic_scores.csv", index=False)
print(out)
print("\nSaved")


print("Load Labels")
spec    = pd.read_csv(SPEC_FN)
lblnorm = pd.read_csv(LBL_NORM)[["account_id","name"]]

for col in ["account_id"]:
    try:
        spec[col]    = spec[col].astype("Int64")
        lblnorm[col] = lblnorm[col].astype("Int64")
    except Exception:
        spec[col]    = spec[col].astype(str)
        lblnorm[col] = lblnorm[col].astype(str)

# evaluate only on joint accounts
df = spec.merge(lblnorm, on="account_id", how="inner")
print(f"Matched accounts for extrinsic eval: {len(df):,} / {len(spec):,}")

k_cols = sorted([c for c in df.columns if c.startswith("cluster_k")],
                key=lambda s: int(s.split("cluster_k")[-1]))

def purity(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    if len(y_true) == 0: return float("nan")
    score = 0
    for c in np.unique(y_pred):
        idx = (y_pred == c)
        if idx.any():
            score += Counter(y_true[idx]).most_common(1)[0][1]
    return score / len(y_true)



y_ref = df["name"].to_numpy()

rows = []
for kcol in k_cols:
    k = int(kcol.split("cluster_k")[-1])
    y = df[kcol].to_numpy()
    rows.append({
        "k": k,
        "NMI_vs_norm":  normalized_mutual_info_score(y_ref, y),
        "ARI_vs_norm":  adjusted_rand_score(y_ref, y),
        "Purity_vs_norm": purity(y_ref, y),
    })

ext = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
# display(ext)
print(ext.to_string())


out_csv = os.path.expanduser("transactions/spectral_external_vs_norm.csv")
ext.to_csv(out_csv, index=False)
print("Saved:", out_csv)
