import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, normalized_mutual_info_score, adjusted_rand_score
import numpy as np, re
from collections import Counter
from sklearn.preprocessing import normalize
from pathlib import Path

LBL_NORM = "~/stellar-clustering/network/labled-data/labels/label-normalization/labels_entities_normalized.csv"

TYPE = 'transactions'
FN = f"{TYPE}/tx_role2vec_dbscan_cosine_pca64_kgrid_test.csv"

EMB_COL = 'role'

CORE_MIN = 500
COV_MIN, COV_MAX = 0.05, 0.80

def parse_eps_token(token: str):
    if token is None:
        return np.nan
    try:
        return float(token)
    except Exception:
        try:
            return float(token.replace("_", "."))
        except Exception:
            return np.nan

def parse_ms_eps(col: str):

    m = re.match(r"dbscan_ms(\d+)_p(\d+)_eps_(.+)$", col)
    if m:
        return int(m.group(1)), parse_eps_token(m.group(3))
    m = re.match(r"dbscan_label_eps_(.+)$", col)
    if m:
        return np.nan, parse_eps_token(m.group(1))
    return np.nan, np.nan

# Intrinsic
print("read data:", FN)
df_intr = pd.read_csv(FN)
X = df_intr[[c for c in df_intr.columns if c.startswith(EMB_COL)]].values

db_cols = [c for c in df_intr.columns if c.startswith("dbscan_")]
print(f"Found {len(db_cols)} DBSCAN labels")

print("run intr metrics")
rows = []
for i, col in enumerate(db_cols, 1):
    print(f"[Intrinsic {i}/{len(db_cols)}] column = {col}")
    y = df_intr[col].to_numpy()
    mask = (y != -1)
    core_size = int(mask.sum())
    n_clusters = len(set(y[mask])) if core_size > 0 else 0
    coverage = float(mask.mean())
    print(f"core_size={core_size} coverage={coverage:.3f} n_clusters={n_clusters}")

    sil = dbi = ch = np.nan
    if core_size >= CORE_MIN and n_clusters >= 2:
        X_core = X[mask]
        X_core_unit = normalize(X_core, norm="l2", axis=1)
        sil = silhouette_score(X_core, y[mask], metric="cosine")   
        dbi = davies_bouldin_score(X_core_unit, y[mask])      
        ch  = calinski_harabasz_score(X_core_unit, y[mask])     
        print(f"silhouette={sil:.4f}  DBI={dbi:.4f}  CH={ch:.2f}")
    else:
        print("Skipped too small or <2 clusters.")

    ms, eps_val = parse_ms_eps(col)
    rows.append({
        "label_col": col,
        "min_samples": ms,
        "eps_value": eps_val,
        "core_size": core_size,
        "n_clusters": n_clusters,
        "coverage": coverage,
        "silhouette_core": sil,
        "davies_bouldin_core": dbi,
        "calinski_harabasz_core": ch
    })

intr = pd.DataFrame(rows).sort_values(["min_samples","eps_value","label_col"], na_position="last")
intr_out = f"{TYPE}/evaluation_res/{TYPE}_dbscan_intrinsic_scores_kgrid_test.csv"
out_path = Path(intr_out)
out_path.parent.mkdir(parents=True, exist_ok=True) 

intr.to_csv(intr_out, index=False)
print("\nIntrinsic summary (first 5 rows):")
print(intr.head())
print("Saved intrinsic to:", intr_out)




# Extrinsic
print("\nrun extr metrics")
def purity(y_true, y_pred):
    if len(y_true) == 0:
        return float("nan")
    total = len(y_true); score = 0
    for c in set(y_pred):
        idx = (y_pred == c)
        if idx.any():
            score += Counter(y_true[idx]).most_common(1)[0][1]
    return score / total

emb_all = pd.read_csv(FN)
lblnorm = pd.read_csv(LBL_NORM)[["account_id", "name"]]

for c in ["account_id"]:
    try:
        emb_all[c] = emb_all[c].astype("Int64")
        lblnorm[c] = lblnorm[c].astype("Int64")
    except Exception:
        emb_all[c] = emb_all[c].astype(str)
        lblnorm[c] = lblnorm[c].astype(str)

df_ext = emb_all.merge(lblnorm, on="account_id", how="inner")
print(f"Matched accounts: {len(df_ext):,} / {len(emb_all):,}")

db_cols_ext = [c for c in df_ext.columns if c.startswith("dbscan_")]
print(f"Found {len(db_cols_ext)} DBSCAN labels")

y_ref = df_ext["name"].to_numpy()

rows = []
for j, col in enumerate(sorted(db_cols_ext), 1):
    print(f"[Extrinsic {j}/{len(db_cols_ext)}] column = {col}")
    y = df_ext[col].to_numpy()
    mask = (y != -1)
    core_size = int(mask.sum())
    coverage = float(mask.mean())
    n_clusters = len(set(y[mask])) if core_size > 0 else 0
    print(f"  core_size={core_size} coverage={coverage:.3f} n_clusters={n_clusters}")

    ARI_masked = NMI_masked = Purity_masked = float("nan")
    if core_size > 0 and n_clusters >= 1:
        ARI_masked = adjusted_rand_score(y_ref[mask], y[mask])
        NMI_masked = normalized_mutual_info_score(y_ref[mask], y[mask])
        Purity_masked = purity(y_ref[mask], y[mask])
        print(f"MASKED  NMI={NMI_masked:.4f} ARI={ARI_masked:.4f} Purity={Purity_masked:.4f}")
    else:
        print("Skipped masked.")

    ARI_strict = adjusted_rand_score(y_ref, y)
    NMI_strict = normalized_mutual_info_score(y_ref, y)
    print(f"STRICT  NMI={NMI_strict:.4f} ARI={ARI_strict:.4f}")

    ms, eps_val = parse_ms_eps(col)
    rows.append({
        "label_col": col,
        "min_samples": (ms if not (isinstance(ms, float) and np.isnan(ms)) else np.nan),
        "eps_value": eps_val,
        "core_size": core_size,
        "coverage": coverage,
        "n_clusters": n_clusters,
        "ARI_masked": ARI_masked,
        "NMI_masked": NMI_masked,
        "Purity_masked": Purity_masked,
        "ARI_strict": ARI_strict,
        "NMI_strict": NMI_strict,
    })

ext = pd.DataFrame(rows).sort_values(["min_samples","eps_value","label_col"], na_position="last").reset_index(drop=True)
ext_out = f"{TYPE}/evaluation_res/{TYPE}_dbscan_extrinsic_scores_kgrid_test.csv"
out_path = Path(ext_out)
out_path.parent.mkdir(parents=True, exist_ok=True)

ext.to_csv(ext_out, index=False)
print("\nExtrinsic summary:")
print(ext.head())
print("Saved extrinsic to:", ext_out)
