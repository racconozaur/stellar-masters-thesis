import pandas as pd
import networkx as nx
import pickle


print("Loading CSV")
transaction_edges = pd.read_csv("../transaction_edges_mv.csv")
trustline_edges   = pd.read_csv("../trustline_edges_mv.csv")
account_node_features = pd.read_csv("../new/account_node_features_full_mv.csv")
account_asset_holdings = pd.read_csv("../new/account_asset_holdings_mv.csv")


try:
    account_nodes_all = pd.read_csv("../new/account_nodes_all_mv.csv")
except Exception:
    account_nodes_all = None


transaction_edges = transaction_edges.dropna(subset=["sender_id","receiver_id"])
trustline_edges   = trustline_edges.dropna(subset=["account_id","issuer_id"])

# remove self loops (u == v)
transaction_edges = transaction_edges[transaction_edges["sender_id"] != transaction_edges["receiver_id"]]
trustline_edges   = trustline_edges[trustline_edges["account_id"] != trustline_edges["issuer_id"]]


for col in ["amount"]:
    if col in transaction_edges.columns:
        transaction_edges[col] = pd.to_numeric(transaction_edges[col], errors="coerce").fillna(0)

for col in ["balance", "token_balance"]:
    if col in trustline_edges.columns:
        trustline_edges[col] = pd.to_numeric(trustline_edges[col], errors="coerce").fillna(0)
    if col in account_asset_holdings.columns:
        account_asset_holdings[col] = pd.to_numeric(account_asset_holdings[col], errors="coerce").fillna(0)




print("aggregating transaction edges for directed")
tx_agg = (
    transaction_edges
    .groupby(["sender_id","receiver_id"], as_index=False)
    .agg(
        tx_count   = ("operation_id","count"),
        amount_sum = ("amount","sum"),
        first_tx   = ("tx_date","min"),
        last_tx    = ("tx_date","max"),
    )
)

print("aggregating trustline edges for undirected")
tl_agg = (
    trustline_edges
    .groupby(["account_id","issuer_id"], as_index=False)
    .agg(
        n_assets    = ("asset_code","nunique"),
        balance_sum = ("balance","sum")
    )
)

# Build graphs

print("building directed transaction graph G_tx")
G_tx = nx.DiGraph()

# Add edges with attributes
for r in tx_agg.itertuples(index=False):
    u = r.sender_id
    v = r.receiver_id
    G_tx.add_edge(
        u, v,
        type="transaction",
        tx_count=int(r.tx_count),
        amount_sum=float(r.amount_sum),
        first_tx=str(r.first_tx),
        last_tx=str(r.last_tx)
    )

# Undirected trustline graph
print("Building undirected trustline graph G_trust")
G_trust = nx.Graph()
for r in tl_agg.itertuples(index=False):
    u = r.account_id
    v = r.issuer_id
    G_trust.add_edge(
        u, v,
        type="trustline",
        n_assets=int(r.n_assets),
        balance_sum=float(r.balance_sum)
    )


print("adding node attributes...")

numeric_cols = [c for c in account_node_features.columns if c not in ["account_id","address"]]
numeric_features = (
    account_node_features
    .set_index("account_id")[numeric_cols]
    .fillna(0)
    .to_dict("index")
)

addr_map = (
    account_node_features
    .set_index("account_id")["address"]
    .dropna()
    .to_dict()
)

holdings_map = (
    account_asset_holdings
    .groupby("account_id")
    .apply(lambda df: df[["asset_code","asset_issuer","issuer_id","token_balance"]].to_dict("records"))
    .to_dict()
)


def apply_node_attrs(G):
    nodes = set(G.nodes())
    feats = {k: v for k, v in numeric_features.items() if k in nodes}
    addrs = {k: v for k, v in addr_map.items() if k in nodes}
    holds = {k: v for k, v in holdings_map.items() if k in nodes}
    nx.set_node_attributes(G, feats)
    nx.set_node_attributes(G, addrs, "address")
    nx.set_node_attributes(G, holds, "token_holdings")

apply_node_attrs(G_tx)
apply_node_attrs(G_trust)

#  undirected tx graph 
print("undirected, weighted transaction graph")
tx_pairs = tx_agg.copy()

# build undirected pair (u,v) with u < v to merge both directions
tx_pairs["u"] = tx_pairs[["sender_id","receiver_id"]].min(axis=1)
tx_pairs["v"] = tx_pairs[["sender_id","receiver_id"]].max(axis=1)

tx_und = (
    tx_pairs
    .groupby(["u","v"], as_index=False)
    .agg(
        tx_count_total   = ("tx_count","sum"),
        amount_sum_total = ("amount_sum","sum"),
        first_tx         = ("first_tx","min"),
        last_tx          = ("last_tx","max")
    )
)

G_tx_und = nx.Graph()
for r in tx_und.itertuples(index=False):
    G_tx_und.add_edge(
        r.u, r.v,
        type="transaction",
        weight=int(r.tx_count_total),
        w_tx_count=int(r.tx_count_total),
        w_amount_sum=float(r.amount_sum_total),
        first_tx=str(r.first_tx),
        last_tx=str(r.last_tx)
    )


apply_node_attrs(G_tx_und)




def summarize(G, name):
    print(f"{name}: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    if G.number_of_nodes() > 0:
        n_isolates = sum(1 for _ in nx.isolates(G))
        print(f"  isolates: {n_isolates:,}")

summarize(G_tx, "G_tx (directed transactions)")
summarize(G_trust, "G_trust (undirected trustlines)")
summarize(G_tx_und, "G_tx_und (undirected transactions)")



print("Saving PKL files...")
with open("stellar_G_tx_directed.pkl", "wb") as f:
    pickle.dump(G_tx, f, protocol=pickle.HIGHEST_PROTOCOL)

with open("stellar_G_trust_undirected.pkl", "wb") as f:
    pickle.dump(G_trust, f, protocol=pickle.HIGHEST_PROTOCOL)

with open("stellar_G_tx_undirected_weighted.pkl", "wb") as f:
    pickle.dump(G_tx_und, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Done.")
