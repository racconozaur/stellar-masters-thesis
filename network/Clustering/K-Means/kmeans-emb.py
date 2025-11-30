import pandas as pd
from sklearn.cluster import KMeans


embeddings = pd.read_csv("../Role2Vec/embeddings/trust_lcc_role2vec_d128.csv")
X = embeddings.drop(columns=["account_id"]).values


K_VALUES = [10, 15, 20, 30, 40, 50, 65, 70, 75, 80, 100, 120, 150, 180, 210, 250, 300, 350, 400]

for k in K_VALUES:
    print(f"Running K-means with k={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    embeddings[f"kmeans_{k}"] = kmeans.fit_predict(X)


embeddings.to_csv("Role2Vec_res/trustlines/trustlines_role2vec_kmeans_results.csv", index=False)
print("Saved results")
