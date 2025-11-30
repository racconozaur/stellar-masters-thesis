

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN



EMB_PATH = os.path.expanduser('~/stellar-clustering/network/Role2Vec/embeddings/tx_lcc_role2vec_d128.csv')


METRIC = 'cosine' 
PERCENTILES = [70, 80, 85, 90, 95]   
MIN_SAMPLES = 15


OUT = f"Role2Vec_res/transactions/tx_role2vec_dbscan_{METRIC}_pca64_pcts.csv"

print('emb: ', EMB_PATH)
print('out: ', OUT)

os.makedirs(os.path.dirname(OUT), exist_ok=True)

print('read emb')
embeddings = pd.read_csv(EMB_PATH)
X = embeddings.drop(columns=["account_id"]).to_numpy(dtype=float)


print('Standardize + PCA')
X = StandardScaler().fit_transform(X)
X = PCA(n_components=64, random_state=42).fit_transform(X)



# print(f'Compute k-distance percentiles (metric={METRIC}, min_samples={MIN_SAMPLES})')
# nn = NearestNeighbors(n_neighbors=MIN_SAMPLES, metric=METRIC).fit(X)
# dists, _ = nn.kneighbors(X)
# kth = np.sort(dists[:, -1])

# pct_to_eps = {p: float(np.percentile(kth, p)) for p in PERCENTILES}
# print('k-dist seeds:', {p: round(e, 6) for p, e in pct_to_eps.items()})


# # Run DBSCAN
# print("Run DBSCAN")
# for p in PERCENTILES:
#     eps = pct_to_eps[p]
#     labels = DBSCAN(
#         eps=eps,
#         min_samples=MIN_SAMPLES,
#         metric=METRIC,
#         algorithm="brute" if METRIC == "cosine" else "auto",
#         n_jobs=1  # safer wrt BLAS threads
#     ).fit_predict(X)

#     col_name = f"dbscan_p{p}_eps_{eps:.6f}"
#     embeddings[col_name] = labels

#     n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
#     n_noise = int((labels == -1).sum())
#     print(f"[p={p:>2} eps={eps:.6f}] clusters(excl noise)={n_clusters} | noise={n_noise}")


# embeddings.to_csv(OUT, index=False)
# print("Saved:", OUT)

#------------------------------------

#trustlines 

k = 15

eps_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
print("eps vals:", eps_vals)

for e in eps_vals:
    labels = DBSCAN(eps=e, min_samples=k, metric=METRIC).fit_predict(X)
    col_name = f"dbscan_label_eps_{e}"
    embeddings[col_name] = labels

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()

    print(f"[eps={e}] Clusters (excluding noise): {n_clusters} | Noise points: {n_noise}")

# save once with all label columns
embeddings.to_csv(OUT, index=False)
print("Saved: ", OUT)

# ---------------------------------

# embeddings["dbscan_label"] = labels

# embeddings.to_csv(OUT, index=False)

# print("Clusters (excluding noise):", len(set(labels)) - (1 if -1 in labels else 0))
# print("Noise points:", (labels == -1).sum())
# print("Saved: ", OUT)
