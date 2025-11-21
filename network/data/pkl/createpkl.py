import pandas as pd
import networkx as nx
import pickle

from networkx import relabel_nodes

print('Loading csvs')

transaction_edges = pd.read_csv("../transaction_edges_mv.csv")
trustline_edges = pd.read_csv("../trustline_edges_mv.csv")
account_node_features = pd.read_csv("../new/account_node_features_full_mv.csv")
account_asset_holdings = pd.read_csv("../new/account_asset_holdings_mv.csv")
account_nodes = pd.read_csv("../new/account_nodes_all_mv.csv")

print("constructing the graph..")
G = nx.Graph()

print("transaction edges")
for _, row in transaction_edges.iterrows():
    G.add_edge(
        row['sender_id'],
        row['receiver_id'],
        weight=row['amount'],
        type='transaction',
        tx_date=row['tx_date'],
        operation_id=row['operation_id']
    )



print("trustline edges...")
for _, row in trustline_edges.iterrows():
    G.add_edge(
        row['account_id'],
        row['issuer_id'],
        weight=row['balance'],
        type='trustline',
        asset_code=row['asset_code']
    )



G.add_nodes_from(account_nodes['account_id'].tolist())

feature_dict = account_node_features.set_index('account_id').to_dict('index')

nx.set_node_attributes(G, feature_dict)

asset_map = account_asset_holdings.groupby('account_id').apply(lambda df: df[['asset_code', 'token_balance']].to_dict('records')).to_dict()

nx.set_node_attributes(G, asset_map, "token_holdings")


print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

with open("account-transaction-trustline-full-new.pkl", "wb") as f:
    pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Graph saved as 'account-transaction-trustline-full-new.pkl'")
