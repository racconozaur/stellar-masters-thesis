import os
import pickle
import networkx as nx
import pandas as pd
from networkx.algorithms.community import louvain_communities


RESOLUTIONS = [0.5, 0.8, 1.0, 1.2]
THRESHOLD = 1e-7
SEED = 42
WEIGHT = "weight"

GRAPH_PATH = "../trust_proj_LCC_idf/trust_proj_LCC_idf.pkl"

OUT_DIR = "./"
SUMMARY_CSV = os.path.join(OUT_DIR, "louvain_summary_by_resolution.csv")


print("Loading graph")
with open(GRAPH_PATH, "rb") as f:
    G_lcc = pickle.load(f)

print(f"LCC nodes: {G_lcc.number_of_nodes():,}")
print(f"LCC edges: {G_lcc.number_of_edges():,}")


isolated_nodes = list(nx.isolates(G_lcc))
if isolated_nodes:
    print(f"Found {len(isolated_nodes):,} isolates")
    G_lcc = G_lcc.copy()
    G_lcc.remove_nodes_from(isolated_nodes)
    print(f"After removal: {G_lcc.number_of_nodes():,} nodes, {G_lcc.number_of_edges():,} edges")
else:
    print("No isolates detected in LCC")



def stable_sort_communities(communities):

    def key_fn(c):
        try:
            mn = min(c)
        except ValueError:
            mn = float("inf")
        return (-len(c), mn)
    return sorted(communities, key=key_fn)


summary_rows = []

for RESOLUTION in RESOLUTIONS:
    print(f"\nRunning Louvain | resolution={RESOLUTION} | threshold={THRESHOLD} | seed={SEED} | weight='{WEIGHT}'")

    communities = louvain_communities(
        G_lcc,
        resolution=RESOLUTION,
        threshold=THRESHOLD,
        seed=SEED,
        weight=WEIGHT
    )

    communities_sorted = stable_sort_communities(communities)
    node_to_community = {}
    for cid, community in enumerate(communities_sorted):
        for node in community:
            node_to_community[node] = cid

    # node to community mapping
    df_result = pd.DataFrame({
        "account_id": list(node_to_community.keys()),
        "community": list(node_to_community.values()),
        "resolution": RESOLUTION
    })
    out_path = os.path.join(OUT_DIR, f"louvain_result_res{RESOLUTION}.csv")
    df_result.to_csv(out_path, index=False)
    print(f"Saved partition → {out_path}  ({len(df_result):,} rows)")

    # weighted odularity summary 
    modularity_score = nx.algorithms.community.quality.modularity(
        G_lcc, communities_sorted, weight=WEIGHT
    )
    n_comms = len(communities_sorted)
    print(f"Found {n_comms:,} communities | Modularity = {modularity_score:.6f}")

    summary_rows.append({
        "resolution": RESOLUTION,
        "n_communities": n_comms,
        "modularity": modularity_score
    })


df_summary = pd.DataFrame(summary_rows).sort_values("resolution")
df_summary.to_csv(SUMMARY_CSV, index=False)
print(f"\nSaved summary → {SUMMARY_CSV}")
print(df_summary.to_string(index=False))
