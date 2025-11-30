
import os
import pickle
import pandas as pd
import networkx as nx
from pathlib import Path

TX_GRAPH_PKL = os.path.expanduser(
    "~/stellar-clustering/network/Community Detection/Louvian/transaction_graph/TX-LCC/LCC_G_tx_undirected_weighted.pkl"
)
OUT_DIR_TX   = "./role2vec_io/tx_lcc"


TRUST_GRAPH_PKL = os.path.expanduser(
    "~/stellar-clustering/network/Community Detection/Louvian/trustline_graph/trust_proj_LCC_idf/trust_proj_LCC_idf.pkl"
)
OUT_DIR_TRUST   = "./role2vec_io/trust_lcc"




def load_and_normalize_graph(path: str) -> nx.Graph:
    with open(path, "rb") as f:
        G = pickle.load(f)
    if isinstance(G, nx.DiGraph):
        G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    


def relabel_to_int(G: nx.Graph):
    nodes = list(G.nodes())
    orig2int = {n: i for i, n in enumerate(nodes)}
    int2orig = {i: n for n, i in orig2int.items()}
    G_int = nx.relabel_nodes(G, orig2int, copy=True)
    return G_int, orig2int, int2orig

def export_role2vec_io(G: nx.Graph, out_dir: str, label: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    G_int, _, int2orig = relabel_to_int(G)

    edgelist_csv = out / "edges.csv"
    with edgelist_csv.open("w") as fw:
        fw.write("source,target\n")
        for u, v in G_int.edges():
            fw.write(f"{u},{v}\n")

    mapping_csv = out / "node_mapping.csv"
    pd.DataFrame(
        {
            "node_idx": list(int2orig.keys()),
            "account_id": [int2orig[i] for i in range(len(int2orig))],
        }
    ).sort_values("node_idx").to_csv(mapping_csv, index=False)

    print(f"[{label}] Wrote edgelist: {edgelist_csv}")
    print(f"[{label}] Wrote mapping : {mapping_csv}")
    print(f"[{label}] Nodes={G_int.number_of_nodes()} Edges={G_int.number_of_edges()}")

def main():
    print("[TX] Loading:", TX_GRAPH_PKL)
    G_tx = load_and_normalize_graph(TX_GRAPH_PKL)
    export_role2vec_io(G_tx, OUT_DIR_TX, "TX")

    print("[TRUST] Loading:", TRUST_GRAPH_PKL)
    G_tr = load_and_normalize_graph(TRUST_GRAPH_PKL)
    export_role2vec_io(G_tr, OUT_DIR_TRUST, "TRUST")

if __name__ == "__main__":
    main()



# command runs

# python src/main.py \
#   --graph-input  ../role2vec_io/tx_lcc/edges.csv \
#   --output       ../role2vec_io/tx_lcc/role2vec_raw.csv \
#   --features wl --labeling-iterations 2 --log-base 2 \
#   --sampling second --P 1.0 --Q 4.0 \
#   --dimensions 128 --window-size 5 --walk-number 10 --walk-length 80 \
#   --epochs 1 --workers 8 --seed 42


# python src/main.py \
#   --graph-input  ./role2vec_io/trust_lcc/edges.csv \
#   --output       ./role2vec_io/trust_lcc/role2vec_raw.csv \
#   --features wl --labeling-iterations 2 --log-base 2 \
#   --sampling second --P 1.0 --Q 4.0 \
#   --dimensions 128 --window-size 5 --walk-number 10 --walk-length 80 \
#   --epochs 1 --workers 8 --seed 42
