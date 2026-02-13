from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def run_clustering(X, k=3):
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X)

    sil = silhouette_score(X, labels)
    cluster_sizes = [(labels == i).sum() for i in range(k)]

    return model, labels, sil, cluster_sizes