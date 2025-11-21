import pickle, numpy as np, pandas as pd, networkx as nx
from scipy.sparse import csr_matrix, diags
from collections import defaultdict



TX_LCC_PKL = "../transactions/LCC_G_tx_undirected_weighted.pkl"
TRUSTLINES_CSV = "~/stellar-clustering/network/data/trustline_edges_mv.csv"
OUT_PROJ_PKL = "trust_proj_LCC_idf.pkl" 



TOPK = 20 
MAX_ISSUER_DEG = 5000
MIN_IDF = 0.05
MIN_ISSUERS_ACC = 1

# transactions llc accounts accounts
with open(TX_LCC_PKL, "rb") as f:
    G_tx = pickle.load(f)
tx_accounts = np.array(list(G_tx.nodes()), dtype=np.int64)
Aset = set(tx_accounts)
print(f"TX-LCC accounts: {len(Aset):,}")

# trustlines and filter
tl = pd.read_csv(TRUSTLINES_CSV, usecols=["account_id","issuer_id","asset_code","balance"])
tl = tl[tl["account_id"].isin(Aset)].copy()

# aggregate per account, issuer
agg = (tl.groupby(["account_id","issuer_id"])
         .agg(n_assets=("asset_code","nunique"))
         .reset_index())
print(f"(account, issuer) pairs after filtering: {len(agg):,}")

# Build indices
acc_ids = np.array(sorted(agg["account_id"].unique()), dtype=np.int64)
iss_ids = np.array(sorted(agg["issuer_id"].unique()), dtype=np.int64)
acc2idx = {a:i for i,a in enumerate(acc_ids)}
iss2idx = {i:j for j,i in enumerate(iss_ids)}

# per account issuer lists and issuer degrees
acc_issuers = [[] for _ in range(len(acc_ids))]
iss_degree  = np.zeros(len(iss_ids), dtype=np.int64)

for r in agg.itertuples(index=False):
    ai = acc2idx[r.account_id]
    ij = iss2idx[r.issuer_id]
    acc_issuers[ai].append(ij)
    iss_degree[ij] += 1

# drop accounts with very few issuers
active_mask = np.array([len(L) >= MIN_ISSUERS_ACC for L in acc_issuers])
acc_keep_idx = np.where(active_mask)[0]
print(f"Accounts with >= {MIN_ISSUERS_ACC} issuers: {len(acc_keep_idx):,} / {len(acc_ids):,}")

# IDF and issuer 
N = len(acc_ids)
idf = np.log1p(N / (1.0 + iss_degree.astype(np.float64)))


iss_keep_mask = (iss_degree <= MAX_ISSUER_DEG) & (idf > MIN_IDF)
n_kept = int(iss_keep_mask.sum())
print(f"Issuers kept: {n_kept:,} / {len(iss_ids):,} "
      f"(MAX_ISSUER_DEG={MAX_ISSUER_DEG}, MIN_IDF={MIN_IDF})")

# Build inverted index issuer  list of account idx
postings = [[] for _ in range(len(iss_ids))]
for ai in acc_keep_idx:
    for ij in acc_issuers[ai]:
        if iss_keep_mask[ij]:
            postings[ij].append(ai)

# For each account accumulate neighbors and keep top-K
edges_i = []
edges_j = []
edges_w = []

for cnt, u in enumerate(acc_keep_idx, 1):
    scores = defaultdict(float)
    # gather accaunts via issuers of u
    for ij in acc_issuers[u]:
        if not iss_keep_mask[ij]:
            continue
        w_ij = float(idf[ij])
        for v in postings[ij]:
            if v == u:
                continue
            scores[v] += w_ij

    if not scores:
        continue

    # take top-K by weight
    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:TOPK]


    if cnt % 10000 == 0:
        print(f"processed accounts: {cnt:,} (edges so far: {len(edges_w):,})")

print(f"Edges kept after top-{TOPK} & TAU>{TAU}: {len(edges_w):,}")

# Build graph & LCC
G = nx.Graph()

G.add_nodes_from(acc_ids.tolist())

for u_idx, v_idx, w in zip(edges_i, edges_j, edges_w):
    u_id = int(acc_ids[u_idx]); v_id = int(acc_ids[v_idx])
    G.add_edge(u_id, v_id, weight=float(w))

print(f"graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
    lcc_nodes = max(nx.connected_components(G), key=len)
    G_lcc = G.subgraph(lcc_nodes).copy()
else:
    G_lcc = G

print(f"LCC: {G_lcc.number_of_nodes():,} nodes, {G_lcc.number_of_edges():,} edges")

with open(OUT_PROJ_PKL, "wb") as f:
    pickle.dump(G_lcc, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Saved: {OUT_PROJ_PKL}")
