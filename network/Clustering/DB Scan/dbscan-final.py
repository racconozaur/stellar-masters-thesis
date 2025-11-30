import os, numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

EMB_PATH = os.path.expanduser('~/stellar-clustering/network/Node2Vec/txlcc_node2vec_d128_p1_q2_wl30_nw4_w2.csv')
OUT = "Node2Vec_res/transactions/tx_node2vec_dbscan_cosine_pca64_kgrid_test.csv"


MIN_SAMPLES_LIST = [5, 10, 15]
PERCENTILES = [70, 80, 85, 90, 95]   

os.makedirs(os.path.dirname(OUT), exist_ok=True)
emb = pd.read_csv(EMB_PATH)
X = emb.drop(columns=["account_id"]).to_numpy(dtype=float)

# PCA 
X = PCA(n_components=64, random_state=42).fit_transform(X)


for k in MIN_SAMPLES_LIST:
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(X)
    dists, _ = nn.kneighbors(X)
    kth = np.sort(dists[:, -1])
    seeds = {p: float(np.percentile(kth, p)) for p in PERCENTILES}
    print(f"k={k} seeds:", {p: round(e, 4) for p, e in seeds.items()})

    for p, eps in seeds.items():
        labels = DBSCAN(eps=eps, min_samples=k, metric="cosine", algorithm="brute", n_jobs=1).fit_predict(X)
        col = f"dbscan_ms{k}_p{p}_eps_{eps:.6f}"
        emb[col] = labels

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        print(f"[ms={k} p={p} eps={eps:.4f}] clusters={n_clusters} | noise={n_noise}")

emb.to_csv(OUT, index=False)
print("Saved:", OUT)
