"""
Week 3: Unsupervised Learning and Clustering Analysis
Dataset: Iris dataset bundled with scikit-learn (public benchmark dataset).
The species column is retained only for optional post-hoc comparison; it is NOT
used to train the clustering model.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

OUT = Path("outputs")
PLOTS = OUT / "visualizations"
PLOTS.mkdir(parents=True, exist_ok=True)

iris = load_iris(as_frame=True)
X = iris.data.copy()
X.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in X.columns]
reference_species = pd.Series(iris.target).map(dict(enumerate(iris.target_names)))

data = X.copy()
data["species_reference"] = reference_species
data.to_csv(OUT / "iris_clustering_dataset.csv", index=False)

# 1. Scale features because measurements can have different ranges.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Compare several k values using inertia and silhouette score.
candidate_k = range(2, 9)
inertias, silhouettes = [], []
for k in candidate_k:
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    cluster_labels = model.fit_predict(X_scaled)
    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(X_scaled, cluster_labels))

fig, ax = plt.subplots()
ax.plot(list(candidate_k), inertias, marker="o")
ax.set(title="Elbow Method for Choosing k", xlabel="Number of clusters (k)",
       ylabel="Within-cluster sum of squares (inertia)")
fig.tight_layout()
fig.savefig(PLOTS / "01_elbow_method.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots()
ax.plot(list(candidate_k), silhouettes, marker="o")
ax.set(title="Silhouette Scores Across Candidate Cluster Counts",
       xlabel="Number of clusters (k)", ylabel="Silhouette score")
fig.tight_layout()
fig.savefig(PLOTS / "02_silhouette_scores.png", dpi=180)
plt.close(fig)

# 3. Fit final K-Means model. k=3 is selected for interpretable segmentation
# and comparison with the dataset's three known species categories.
model = KMeans(n_clusters=3, random_state=42, n_init=20)
labels = model.fit_predict(X_scaled)
data["cluster"] = labels
data.to_csv(OUT / "iris_clustering_results.csv", index=False)

print("Dataset shape:", X.shape)
print("Missing values:", int(X.isna().sum().sum()))
print("Chosen k:", 3)
print("Silhouette score:", round(silhouette_score(X_scaled, labels), 3))
print("Optional post-hoc ARI vs. species labels:",
      round(adjusted_rand_score(iris.target, labels), 3))
print("\\nCluster sizes:\\n", data["cluster"].value_counts().sort_index())
print("\\nCluster mean profiles:\\n", data.groupby("cluster")[list(X.columns)].mean().round(3))

# 4. Visualize in 2D with PCA. PCA is used only for plotting.
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
centers_pca = pca.transform(model.cluster_centers_)
fig, ax = plt.subplots()
for cluster_id in sorted(data["cluster"].unique()):
    mask = labels == cluster_id
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f"Cluster {cluster_id}", alpha=0.8)
ax.scatter(centers_pca[:, 0], centers_pca[:, 1], marker="X", s=180, label="Centroids")
ax.set(title="K-Means Clusters in PCA Projection",
       xlabel=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
       ylabel=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.legend()
fig.tight_layout()
fig.savefig(PLOTS / "03_pca_cluster_plot.png", dpi=180)
plt.close(fig)

# 5. Standardized cluster profiles
profile = pd.DataFrame(X_scaled, columns=X.columns).assign(cluster=labels).groupby("cluster").mean()
fig, ax = plt.subplots()
im = ax.imshow(profile.values, aspect="auto")
fig.colorbar(im, ax=ax, label="Mean standardized feature value")
ax.set_xticks(range(len(X.columns)), [c.replace("_", " ") for c in X.columns], rotation=25, ha="right")
ax.set_yticks(range(len(profile.index)), [f"Cluster {i}" for i in profile.index])
ax.set_title("Cluster Feature Profiles (Standardized Means)")
fig.tight_layout()
fig.savefig(PLOTS / "04_cluster_feature_profiles.png", dpi=180)
plt.close(fig)

# 6. Feature distributions by cluster (tick_labels works with current Matplotlib)
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, col in zip(axes.flat, X.columns):
    values = [data.loc[data.cluster == cid, col].values for cid in sorted(data.cluster.unique())]
    ax.boxplot(values, tick_labels=[f"C{cid}" for cid in sorted(data.cluster.unique())])
    ax.set_title(col.replace("_", " ").title())
    ax.set_xlabel("Cluster")
fig.suptitle("Feature Distributions by K-Means Cluster")
fig.tight_layout()
fig.savefig(PLOTS / "05_feature_distributions.png", dpi=180)
plt.close(fig)

print(f"\\nAll outputs saved under: {OUT.resolve()}")
