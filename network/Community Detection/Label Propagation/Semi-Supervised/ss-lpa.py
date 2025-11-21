import os
import pickle
import networkx as nx
import pandas as pd
from collections import defaultdict
from networkx.algorithms.community import modularity

LABELS = os.path.expanduser('~/stellar-clustering/network/labled-data/labels/label-normalization/labels_entities_normalized.csv')

def run_sslpa_on_graph(graph_path, output_prefix):
    print(f"\nRunning Semi-Supervised LPA on {graph_path}")
    graph_path = os.path.expanduser(graph_path)
    output_prefix = os.path.expanduser(output_prefix)

    print(f"Resolved graph path: {graph_path}")
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph file not found at: {graph_path}")

    with open(graph_path, "rb") as f:
        G = pickle.load(f)
    if isinstance(G, nx.DiGraph):
        G = nx.Graph(G)
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    if not os.path.exists(LABELS):
        raise FileNotFoundError(f"Labels CSV not found at: {LABELS}")
    seeds_df = pd.read_csv(LABELS, usecols=["account_id", "name"]).dropna().drop_duplicates()
    try:
        seeds_df["account_id"] = seeds_df["account_id"].astype("int64")
    except Exception:
        pass
    seeds_all = dict(zip(seeds_df["account_id"].tolist(), seeds_df["name"].tolist()))
    seeds = {n: lbl for n, lbl in seeds_all.items() if n in G}
    print(f"Loaded seeds: {len(seeds_all)} total, {len(seeds)} present in graph")

    labels = {}
    frozen = set(seeds.keys())
    for n in G.nodes():
        labels[n] = seeds[n] if n in frozen else f"UNLAB_{n}"

    def w(u, v):
        return G[u][v].get("weight", 1.0)

    unlabeled_nodes = [n for n in G.nodes() if n not in frozen]
    unlabeled_nodes.sort(key=lambda x: G.degree(x), reverse=True)

    for _ in range(50):
        changed = 0
        for n in unlabeled_nodes:
            acc = defaultdict(float)
            for nbr in G.neighbors(n):
                lbl = labels.get(nbr)
                if lbl is None:
                    continue
                acc[lbl] += w(n, nbr)
            if not acc:
                continue
            best_w = max(acc.values())
            cands = [lbl for lbl, val in acc.items() if val == best_w]
            old = labels[n]
            new = old if old in cands else sorted(cands, key=lambda s: str(s))[0]
            if new != old:
                labels[n] = new
                changed += 1
        if changed == 0:
            break

    groups = {}
    for node, lbl in labels.items():
        groups.setdefault(lbl, set()).add(node)
    comms = list(groups.values())
    sizes = [len(c) for c in comms]

    try:
        mod = modularity(G, comms, weight="weight")
    except Exception:
        mod = float("nan")

    print(f"Semi-supervised LPA produced {len(comms)} label-groups")
    print("Community size stats:")
    print(f"min={min(sizes)}, max={max(sizes)}, mean={sum(sizes)/len(sizes):.2f}, median={sorted(sizes)[len(sizes)//2]}")
    print(f"Modularity: {mod:.4f}")


    df = pd.DataFrame(list(labels.items()), columns=["node", "label"])
    out_labels = f"{output_prefix}_sslpa_labels.csv"
    os.makedirs(os.path.dirname(out_labels) or ".", exist_ok=True)
    df.to_csv(out_labels, index=False)
    print(f"Saved node label mapping to {out_labels}")

    
    df_comm = df.copy()
    df_comm["community"] = df_comm["label"].astype(str).astype("category").cat.codes
    df_comm = df_comm[["node", "community"]]
    out_comm = f"{output_prefix}_lpa_communities.csv"
    df_comm.to_csv(out_comm, index=False)
    print(f"Saved LPA-style partition to {out_comm}")


    stats = {
        "graph": graph_path,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "num_label_groups": len(comms),
        "modularity": mod,
        "min_size": min(sizes),
        "max_size": max(sizes),
        "mean_size": sum(sizes)/len(sizes),
        "median_size": sorted(sizes)[len(sizes)//2],
        "num_frozen_seeds": len(frozen),
        "seeds_csv": LABELS,
    }
    out_stats = f"{output_prefix}_sslpa_stats.csv"
    os.makedirs(os.path.dirname(out_stats) or ".", exist_ok=True)
    pd.DataFrame([stats]).to_csv(out_stats, index=False)
    print(f"Saved summary stats to {out_stats}")

    return comms, labels, stats


if __name__ == "__main__":
    run_sslpa_on_graph(
        "~/stellar-clustering/network/LCC/transactions/LCC_G_tx_undirected_weighted.pkl",
        "transaction/normalized/sslpa_tx_lcc"
    )
    run_sslpa_on_graph(
        "~/stellar-clustering/network/LCC/trustlines/trust_proj_LCC_idf.pkl",
        "trustline/normalized/sslpa_trust_lcc"
    )
