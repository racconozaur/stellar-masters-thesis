import pandas as pd
import networkx as nx
import os
from nodevectors import Node2Vec
import numpy as np


START_DATE = pd.to_datetime("2025-02-05")
END_DATE = pd.to_datetime("2025-02-06")

EMBEDDING_DIM = 64
WALK_NUM = 10 
WALK_LEN = 80
WINDOW_SIZE = 5

tx_path = "../data/transaction_edges_mv.csv"
trustline_path = "../data/trustline_edges_mv.csv"
nodes_path = "../data/account_nodes_mv.csv"
features_path = "../data/account_node_features_mv.csv"
holdings_path = "../data/account_asset_holdings_mv.csv"

print("Loading CSVs")

tx_df = pd.read_csv(tx_path, parse_dates=["tx_date"])
trust_df = pd.read_csv(trustline_path)
nodes_df = pd.read_csv(nodes_path)
features_df = pd.read_csv(features_path)
holdings_df = pd.read_csv(holdings_path)

# Filter by date
print(f"Filtering transactions from {START_DATE.date()} to {END_DATE.date()}")
tx_range = tx_df[
    (tx_df["tx_date"] >= START_DATE) &
    (tx_df["tx_date"] <= END_DATE)
]

# Build the graph
print("Building graph...")
G = nx.Graph()

active_accounts = set(tx_range["sender_id"]).union(tx_range["receiver_id"])
G.add_nodes_from(active_accounts)

# Add transaction edges
for _, row in tx_range.iterrows():
    G.add_edge(row["sender_id"], row["receiver_id"], weight=row["amount"], type="transaction")

# Add filtered trustline edges
trust_df_filtered = trust_df[
    trust_df["account_id"].isin(active_accounts) |
    trust_df["issuer_id"].isin(active_accounts)
]

for _, row in trust_df_filtered.iterrows():
    G.add_edge(row["account_id"], row["issuer_id"], weight=row["balance"], type="trustline")

# node features and holdings 
features_dict = features_df.set_index("account_id").to_dict("index")
nx.set_node_attributes(G, features_dict)

holdings_by_account = holdings_df.groupby("account_id").apply(
    lambda df: df[["asset_code", "token_balance"]].to_dict("records")
).to_dict()
nx.set_node_attributes(G, holdings_by_account, name="asset_holdings")

print(f"Graph ready with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")



if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
    print("Graph is empty. Skipping Node2Vec for this time range.")
    exit()


# Prepare graph
print("Mapping nodes to integer indices")
node_id_mapping = {node: idx for idx, node in enumerate(G.nodes())}
G_mapped = nx.relabel_nodes(G, node_id_mapping)
reverse_mapping = {v: k for k, v in node_id_mapping.items()}

print("Running Node2Vec")

model = Node2Vec(
    n_components=64,
    walklen=80,            
    return_weight=1.0,      
    neighbor_weight=0.25,   
    epochs=5,                
    threads=10           
)
model.fit(G_mapped)



print("Saving embeddings")
embeddings = model.predict(list(G_mapped.nodes()))
original_ids = [reverse_mapping[i] for i in range(len(embeddings))]
embedding_df = pd.DataFrame(embeddings, index=original_ids)
embedding_df.index.name = 'account_id'
embedding_df.columns = [f"dim_{i}" for i in range(embedding_df.shape[1])]

output_name = f"node2vec_{START_DATE.date()}_to_{END_DATE.date()}.csv"
output_path = f"data/embeddings/{output_name}"
os.makedirs("data/embeddings", exist_ok=True)
embedding_df.to_csv(output_path)

print(f"Saved Node2Vec embeddings to '{output_path}'")
