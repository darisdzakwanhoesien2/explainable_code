# core/shap_engine.py

import shap
import streamlit as st


@st.cache_resource
def create_explainer(_model, X_train):
    background = shap.sample(X_train, min(100, len(X_train)))
    explainer = shap.Explainer(_model, background)
    return explainer


@st.cache_data
def compute_shap_values(_explainer, X_test, sample_size):
    X_explain = X_test.sample(
        min(sample_size, len(X_test)),
        random_state=42
    )
    shap_values = _explainer(X_explain)
    return shap_values, X_explain


def prepare_shap_for_plot(shap_values, class_index=None):

    values = shap_values.values

    if len(values.shape) == 2:
        return shap_values

    if len(values.shape) == 3:

        if class_index is None:
            class_index = 0

        shap_values.values = values[:, :, class_index]
        return shap_values

    return shap_values


# # core/shap_engine.py

# import shap
# import streamlit as st


# @st.cache_resource
# def create_explainer(model, X_train):
#     background = shap.sample(X_train, min(100, len(X_train)))
#     explainer = shap.Explainer(model, background)
#     return explainer


# @st.cache_data
# def compute_shap_values(explainer, X_test, sample_size):
#     X_explain = X_test.sample(
#         min(sample_size, len(X_test)),
#         random_state=42
#     )
#     shap_values = explainer(X_explain)
#     return shap_values, X_explain


# def prepare_shap_for_plot(shap_values, class_index=None):

#     values = shap_values.values

#     if len(values.shape) == 2:
#         return shap_values

#     if len(values.shape) == 3:

#         if class_index is None:
#             class_index = 0

#         shap_values.values = values[:, :, class_index]
#         return shap_values

#     return shap_values


# import shap
# import numpy as np


# def compute_shap(model, X_train, X_test, sample_size=200):

#     background = shap.sample(X_train, min(100, len(X_train)))
#     explainer = shap.Explainer(model, background)

#     X_explain = X_test.sample(min(sample_size, len(X_test)), random_state=42)
#     shap_values = explainer(X_explain)

#     return shap_values, X_explain


# def prepare_shap_for_plot(shap_values, class_index=None):
#     """
#     Ensures SHAP values are 2D for beeswarm plotting.
#     Handles:
#     - Regression
#     - Binary classification
#     - Multi-class classification
#     """

#     values = shap_values.values

#     # Regression case (2D)
#     if len(values.shape) == 2:
#         return shap_values

#     # Multi-class case (3D)
#     if len(values.shape) == 3:

#         if class_index is None:
#             class_index = 0  # default

#         # Extract specific class slice
#         shap_values.values = values[:, :, class_index]
#         return shap_values

#     return shap_values


# import shap
# import numpy as np

# def compute_shap(model, X_train, X_test, sample_size=200):

#     background = shap.sample(X_train, min(100, len(X_train)))
#     explainer = shap.Explainer(model, background)

#     X_explain = X_test.sample(min(sample_size, len(X_test)))
#     shap_values = explainer(X_explain)

#     return shap_values, X_explain
