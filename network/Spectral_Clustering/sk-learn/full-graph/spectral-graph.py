import os
import pickle
import warnings
from typing import List

import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from sklearn.cluster import SpectralClustering


GRAPH_PKL = os.path.expanduser("~/stellar-clustering/network/LCC/trustlines/trust_proj_LCC_idf.pkl")
OUT_DIR = os.path.expanduser("outputs/trustlines")
OUT_CSV = os.path.join(OUT_DIR, "spectral_labels_all_k.csv")

K_LIST = [10, 15, 20, 30, 40, 50, 65, 70, 75, 80, 100, 120, 150, 180, 210, 250, 300, 350, 400]

RANDOM_STATE = 42
N_INIT = 10


os.makedirs(OUT_DIR, exist_ok=True)
warnings.filterwarnings("ignore", category=UserWarning)

def load_graph(path: str) -> nx.Graph:
    with open(path, "rb") as f:
        G = pickle.load(f)
    return G

def build_affinity_from_graph(G: nx.Graph):
    
    # Build affinity
    
    nodes = list(G.nodes())
    index = {n: i for i, n in enumerate(nodes)}

    rows, cols, data = [], [], []

    for u, v, d in G.edges(data=True):
        i, j = index[u], index[v]
        w = float(d.get("weight", 1.0))
        rows.append(i); cols.append(j); data.append(w)

    n = len(nodes)
    A = sparse.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)


    A.setdiag(0.0)
    A.eliminate_zeros()
    return A, nodes

def prepare_affinity_for_spectral(A: sparse.spmatrix):
    # make symmetric    
    A = A.maximum(A.T)

    # ensure finite non-negative
    if A.nnz > 0:
        if not np.isfinite(A.data).all():
            raise ValueError("Affinity has NaN.")
        if A.data.min() < 0:
            raise ValueError("Affinity has negative weights.")

    # convert to CSC and sort indices
    A = A.tocsc()
    A.sort_indices()

def run_spectral_precomputed(A: sparse.spmatrix, k: int) -> np.ndarray:
    sc = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        assign_labels="kmeans",
        eigen_solver="lobpcg",
        random_state=RANDOM_STATE,
        n_init=N_INIT,
    )
    return sc.fit_predict(A)

def main():
    print("Loading graph")
    G = load_graph(GRAPH_PKL)
    print(f"Graph loaded: |V|={G.number_of_nodes():,}  |E|={G.number_of_edges():,}")

    print("Building affinity")
    A, nodes = build_affinity_from_graph(G)
    print(f"Affinity: shape={A.shape}, nnz={A.nnz:,}")

    print("Preparing affinity for Spectral")
    A = prepare_affinity_for_spectral(A)
    print(f"Prepared affinity: format={A.getformat()}, nnz={A.nnz:,}")


    df = pd.DataFrame({"account_id": nodes})

    for k in K_LIST:
        print(f"\nSpectralClustering k={k}")
        labels = run_spectral_precomputed(A, k)
        df[f"cluster_k{k}"] = labels.astype(int)
        print(f"done: added column cluster_k{k}")

    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved all labels to {OUT_CSV}")

if __name__ == "__main__":
    main()
