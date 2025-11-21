import pickle
import networkx as nx
import pandas as pd
from networkx.algorithms.community import asyn_lpa_communities, modularity

def run_lpa_on_graph(graph_path, output_prefix, seed=42):
    print(f"\nRunning LPA on {graph_path}")
    

    with open(graph_path, "rb") as f:
        G = pickle.load(f)
    print(f"Graph {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # LPA
    comms = list(asyn_lpa_communities(G, weight="weight", seed=seed))
    node2cid = {node: i for i, comm in enumerate(comms) for node in comm}
    print(f"LPA found {len(comms)} communities")


    sizes = [len(c) for c in comms]
    mod = modularity(G, comms, weight="weight")

    print(f"Community size stats:")
    print(f"min={min(sizes)}, max={max(sizes)}, mean={sum(sizes)/len(sizes):.2f}, median={sorted(sizes)[len(sizes)//2]}")
    print(f"Modularity: {mod:.4f}")

    df = pd.DataFrame(list(node2cid.items()), columns=["node", "community"])
    df.to_csv(f"{output_prefix}_lpa_communities.csv", index=False)
    print(f"Saved node community mapping to {output_prefix}_lpa_communities.csv")

    stats = {
        "graph": graph_path,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "num_communities": len(comms),
        "modularity": mod,
        "min_size": min(sizes),
        "max_size": max(sizes),
        "mean_size": sum(sizes)/len(sizes),
        "median_size": sorted(sizes)[len(sizes)//2],
    }
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(f"{output_prefix}_lpa_stats.csv", index=False)
    print(f"Saved to {output_prefix}_lpa_stats.csv")

    return comms, node2cid, stats


if __name__ == "__main__":
    
    run_lpa_on_graph("~/stellar-clustering/network/LCC/transactions/LCC_G_tx_undirected_weighted.pkl", "transaction/lpa_tx_lcc")
    run_lpa_on_graph("~/stellar-clustering/network/LCC/trustlines/trust_proj_LCC_idf.pkl", "trustline/lps_trust_lcc")

