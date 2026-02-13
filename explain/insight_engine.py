class InsightEngine:
    def __init__(self):
        pass

    def clustering_insight(self, silhouette_score, cluster_sizes):
        report = {}

        # Quality assessment
        if silhouette_score > 0.7:
            quality = "Clusters are well-separated and highly cohesive."
        elif silhouette_score > 0.5:
            quality = "Clusters show moderate separation."
        else:
            quality = "Clusters overlap significantly. Structure may be weak."

        # Balance check
        imbalance_ratio = max(cluster_sizes) / min(cluster_sizes)

        if imbalance_ratio > 5:
            imbalance_note = "Significant imbalance detected between clusters."
        else:
            imbalance_note = "Cluster distribution is relatively balanced."

        report["summary"] = quality
        report["distribution"] = imbalance_note
        report["suggestion"] = "Consider tuning k or feature scaling if separation is weak."

        return report

    def classification_insight(self, accuracy, f1_score):
        report = {}

        if accuracy > 0.85:
            performance = "Model shows strong predictive capability."
        elif accuracy > 0.7:
            performance = "Model performance is moderate."
        else:
            performance = "Model performance is weak and may require feature engineering."

        report["summary"] = performance
        report["risk"] = "Check class imbalance and overfitting."
        report["confidence"] = f"Accuracy: {accuracy:.2f}, F1 Score: {f1_score:.2f}"

        return report