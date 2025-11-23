import os
import pickle
import igraph as ig
import networkx as nx
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

LABELS = os.path.expanduser('~/stellar-clustering/network/labled-data/labels/label-normalization/labels_entities_normalized.csv')

def run_sslpa_on_graph(graph_path, output_prefix):
    print(f"\nrunning SSLPA on {graph_path}")
    graph_path = os.path.expanduser(graph_path)
    output_prefix = os.path.expanduser(output_prefix)

    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"no graph: {graph_path}")

    with open(graph_path, "rb") as f:
        G_nx = pickle.load(f)
    if isinstance(G_nx, nx.DiGraph):
        G_nx = nx.Graph(G_nx)
    print(f"Graph loaded: {G_nx.number_of_nodes()} nodes, {G_nx.number_of_edges()} edges")

    # Convert to igraph
    G = ig.Graph.from_networkx(G_nx)
    print(f"Converted to igraph: {G.vcount()} vertices, {G.ecount()} edges")

    # seeds
    if not os.path.exists(LABELS):
        raise FileNotFoundError(f"no labels at: {LABELS}")
    seeds_df = pd.read_csv(LABELS, usecols=["account_id", "name"]).dropna().drop_duplicates()
    try:
        seeds_df["account_id"] = seeds_df["account_id"].astype("int64")
    except Exception:
        pass
    seeds_all = dict(zip(seeds_df["account_id"].tolist(), seeds_df["name"].tolist()))
    
    # Filter seeds present in graph
    nx_nodes = set(G_nx.nodes())
    seeds = {n: lbl for n, lbl in seeds_all.items() if n in nx_nodes}
    print(f"Loaded seeds: {len(seeds_all)} total, {len(seeds)} present in graph")


    le = LabelEncoder()
    unique_labels = sorted(set(seeds.values()))
    le.fit(unique_labels)
    
    initial_labels = []
    fixed_mask = []
    node_id_map = {}
    
    for i, v in enumerate(G.vs):
        node_id = v['_nx_name']
        node_id_map[i] = node_id
        
        if node_id in seeds:
            # label to numeric td
            label_id = le.transform([seeds[node_id]])[0]
            initial_labels.append(label_id)
            fixed_mask.append(True)
        else:
            initial_labels.append(0)
            fixed_mask.append(False)
    
    print(f"Prepared {sum(fixed_mask)} fixed seed nodes for propagation")
    
    print("Running sslpa")
    communities = G.community_label_propagation(
        weights='weight',
        initial=initial_labels,
        fixed=fixed_mask
    )
    
    membership = communities.membership
    
    # map results back to original node id
    labels = {}
    for i, comm_id in enumerate(membership):
        node_id = node_id_map[i]
        
        if node_id in seeds:
            labels[node_id] = seeds[node_id]
        else:
            # For unlabeled nodes, find a seed in same community
            found_label = None
            for j, other_comm_id in enumerate(membership):
                if other_comm_id == comm_id:
                    other_node_id = node_id_map[j]
                    if other_node_id in seeds:
                        found_label = seeds[other_node_id]
                        break
            
            if found_label:
                labels[node_id] = found_label
            else:
                labels[node_id] = f"CLUSTER_{comm_id}"
    
    # Group nodes by label
    groups = defaultdict(set)
    for node, lbl in labels.items():
        groups[lbl].add(node)
    comms = list(groups.values())
    sizes = [len(c) for c in comms]

    # modularity 
    try:
        mod = G.modularity(membership, weights='weight')
    except Exception as e:
        print(f"Cant calculate modularity: {e}")
        mod = float("nan")

    print(f"Semi-supervised LPA produced {len(comms)} label-groups")
    print("Community size stats:")
    print(f"min={min(sizes)}, max={max(sizes)}, mean={sum(sizes)/len(sizes):.2f}, median={sorted(sizes)[len(sizes)//2]}")
    print(f"Modularity: {mod:.4f}")

    # node label mapping
    df = pd.DataFrame(list(labels.items()), columns=["node", "label"])
    out_labels = f"{output_prefix}_sslpa_labels.csv"
    os.makedirs(os.path.dirname(out_labels) or ".", exist_ok=True)
    df.to_csv(out_labels, index=False)
    print(f"Saved node label mapping to {out_labels}")

    # numeric community id
    df_comm = df.copy()
    df_comm["community"] = df_comm["label"].astype(str).astype("category").cat.codes
    df_comm = df_comm[["node", "community"]]
    out_comm = f"{output_prefix}_lpa_communities.csv"
    df_comm.to_csv(out_comm, index=False)
    print(f"Saved LPA-style partition to {out_comm}")

    stats = {
        "graph": graph_path,
        "nodes": G.vcount(),
        "edges": G.ecount(),
        "num_label_groups": len(comms),
        "modularity": mod,
        "min_size": min(sizes),
        "max_size": max(sizes),
        "mean_size": sum(sizes)/len(sizes),
        "median_size": sorted(sizes)[len(sizes)//2],
        "num_frozen_seeds": len(seeds),
        "seeds_csv": LABELS,
        "method": "igraph_label_propagation"
    }
    out_stats = f"{output_prefix}_sslpa_stats.csv"
    os.makedirs(os.path.dirname(out_stats) or ".", exist_ok=True)
    pd.DataFrame([stats]).to_csv(out_stats, index=False)
    print(f"Saved summary to {out_stats}")

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