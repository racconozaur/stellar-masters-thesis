import psycopg2
import pandas as pd

conn = psycopg2.connect(
    dbname='captivecoredb',
    user='captivecore',
    password='stellar123',
    host='localhost',
    port='5432'
)

views = [
    "transaction_edges_mv",
    "trustline_edges_mv",
    "account_nodes_mv",
    "account_node_features_mv",
    "account_asset_holdings_mv"
]

for view in views:
    df = pd.read_sql(f"SELECT * FROM {view};", conn)
    df.to_csv(f"{view}.csv", index=False)

conn.close()
