import numpy as np

class InsightEngine:

    def model_insight(self, problem_type, metrics):

        if problem_type == "classification":
            acc = metrics["accuracy"]

            if acc > 0.85:
                quality = "Strong predictive capability."
            elif acc > 0.7:
                quality = "Moderate performance."
            else:
                quality = "Weak predictive power."

        else:
            r2 = metrics["r2"]

            if r2 > 0.7:
                quality = "Model explains variance well."
            elif r2 > 0.4:
                quality = "Moderate explanatory power."
            else:
                quality = "Low explanatory power."

        return quality

    def shap_insight(self, shap_values, feature_names):

        mean_abs = np.abs(shap_values.values).mean(axis=0)
        top_idx = np.argmax(mean_abs)
        top_feature = feature_names[top_idx]

        return f"Feature '{top_feature}' has the highest global influence."
