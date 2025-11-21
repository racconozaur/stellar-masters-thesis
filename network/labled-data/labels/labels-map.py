import os
import json
import pickle
import pandas as pd
import networkx as nx


META_CSV = os.path.expanduser("~/stellar-clustering/network/data/new/account_node_features_full_mv.csv")
LABELS_JSON = os.path.expanduser("~/stellar-clustering/network/labled-data/full_stellar_directory.json")
TX_LCC_PKL = "../TX-LCC/LCC_G_tx_undirected_weighted.pkl" 
OUT_RAW = "labels_map.csv"
OUT_RAW_LCC = "labels_map_lcc.csv"

print("Load on-chain metadata")
df_meta = (
    pd.read_csv(META_CSV, usecols=["account_id", "address"])
      .dropna()
      .drop_duplicates()
)

df_meta["address"] = df_meta["address"].astype(str).str.strip().str.upper()

print("Load StellarExpert labels")
with open(LABELS_JSON) as f:
    labeled = json.load(f)

df_labels = pd.DataFrame(labeled)[["address", "name"]].dropna().drop_duplicates()
df_labels["address"] = df_labels["address"].astype(str).str.strip().str.upper()
df_labels["name"]    = df_labels["name"].astype(str).str.strip()

print("Join by address")
joined = (
    df_meta.merge(df_labels, on="address", how="inner")[["account_id", "address", "name"]]
            .drop_duplicates()
)



joined["name_len"] = joined["name"].str.len()
joined = (joined.sort_values(["account_id", "name_len", "name"])
                .drop_duplicates(subset=["account_id"], keep="first")
                .drop(columns=["name_len", "address"]))


joined.rename(columns={"name": "name"}).to_csv(OUT_RAW, index=False)



print("Load TX-LCC")
with open(TX_LCC_PKL, "rb") as f:
    G_lcc = pickle.load(f)

lcc_nodes = set(G_lcc.nodes())
labels_lcc = joined[joined["account_id"].isin(lcc_nodes)].copy()

coverage = len(labels_lcc) / len(joined) if len(joined) else 0.0
labels_lcc.rename(columns={"label_raw": "label_raw"}).to_csv(OUT_RAW_LCC, index=False)

print(f"Saved RAW labels filtered to TX-LCC: {len(labels_lcc):,} rows → {OUT_RAW_LCC}")
print(f"Coverage on TX-LCC: {coverage:.4%} ({len(labels_lcc)}/{len(joined)})")
