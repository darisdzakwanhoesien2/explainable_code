import shap
import numpy as np

def compute_shap(model, X_train, X_test, sample_size=200):

    background = shap.sample(X_train, min(100, len(X_train)))
    explainer = shap.Explainer(model, background)

    X_explain = X_test.sample(min(sample_size, len(X_test)))
    shap_values = explainer(X_explain)

    return shap_values, X_explain
