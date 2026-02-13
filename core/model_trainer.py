# core/model_trainer.py

import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    r2_score,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target


def detect_problem_type(y):
    """
    Stronger detection logic.
    """

    # If object or categorical → classification
    if y.dtype == "object":
        return "classification"

    # If numeric
    if np.issubdtype(y.dtype, np.number):

        unique_values = y.nunique()
        n_samples = len(y)

        # If too many unique values → regression
        if unique_values > 0.3 * n_samples:
            return "regression"

        # If small number of unique numeric labels → classification
        if unique_values <= 20:
            return "classification"

        # Default numeric → regression
        return "regression"

    # Fallback
    return "classification"


@st.cache_resource
def train_model(X, y):

    problem_type = detect_problem_type(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if problem_type == "classification":

        # Convert numeric small-class labels to category
        if np.issubdtype(y.dtype, np.number):
            y_train = y_train.astype(str)
            y_test = y_test.astype(str)

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds, average="weighted"),
        }

    else:

        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = {
            "r2": r2_score(y_test, preds),
            "mse": mean_squared_error(y_test, preds),
        }

    return model, X_train, X_test, y_train, y_test, metrics, problem_type


# # core/model_trainer.py

# import streamlit as st
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.metrics import (
#     accuracy_score,
#     f1_score,
#     r2_score,
#     mean_squared_error,
# )
# from sklearn.model_selection import train_test_split
# from sklearn.utils.multiclass import type_of_target


# def detect_problem_type(y):
#     target_type = type_of_target(y)

#     if target_type in ["continuous", "continuous-multioutput"]:
#         return "regression"

#     if target_type in ["binary", "multiclass"]:
#         return "classification"

#     if y.dtype.kind in "if" and y.nunique() > 20:
#         return "regression"

#     return "classification"


# @st.cache_resource
# def train_model(X, y):

#     problem_type = detect_problem_type(y)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     if problem_type == "classification":

#         model = RandomForestClassifier(
#             n_estimators=200,
#             random_state=42,
#             n_jobs=-1
#         )

#         model.fit(X_train, y_train)
#         preds = model.predict(X_test)

#         metrics = {
#             "accuracy": accuracy_score(y_test, preds),
#             "f1": f1_score(y_test, preds, average="weighted"),
#         }

#     else:

#         model = RandomForestRegressor(
#             n_estimators=200,
#             random_state=42,
#             n_jobs=-1
#         )

#         model.fit(X_train, y_train)
#         preds = model.predict(X_test)

#         metrics = {
#             "r2": r2_score(y_test, preds),
#             "mse": mean_squared_error(y_test, preds),
#         }

#     return model, X_train, X_test, y_train, y_test, metrics, problem_type


# # core/model_trainer.py

# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.metrics import (
#     accuracy_score,
#     f1_score,
#     r2_score,
#     mean_squared_error,
# )
# from sklearn.model_selection import train_test_split
# from sklearn.utils.multiclass import type_of_target


# def detect_problem_type(y):
#     """
#     Robust problem-type detection using sklearn internals.
#     """

#     target_type = type_of_target(y)

#     if target_type in ["continuous", "continuous-multioutput"]:
#         return "regression"

#     if target_type in ["binary", "multiclass"]:
#         return "classification"

#     # Fallback safeguard
#     if y.dtype.kind in "if" and y.nunique() > 20:
#         return "regression"

#     return "classification"


# def train_model(X, y):
#     """
#     Automatically trains classification or regression model.
#     Returns:
#         model, X_train, X_test, y_train, y_test, metrics, problem_type
#     """

#     problem_type = detect_problem_type(y)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     if problem_type == "classification":

#         model = RandomForestClassifier(
#             n_estimators=200,
#             random_state=42,
#             n_jobs=-1
#         )

#         model.fit(X_train, y_train)
#         preds = model.predict(X_test)

#         metrics = {
#             "accuracy": accuracy_score(y_test, preds),
#             "f1": f1_score(y_test, preds, average="weighted"),
#         }

#     elif problem_type == "regression":

#         model = RandomForestRegressor(
#             n_estimators=200,
#             random_state=42,
#             n_jobs=-1
#         )

#         model.fit(X_train, y_train)
#         preds = model.predict(X_test)

#         metrics = {
#             "r2": r2_score(y_test, preds),
#             "mse": mean_squared_error(y_test, preds),
#         }

#     else:
#         raise ValueError("Unsupported problem type.")

#     return model, X_train, X_test, y_train, y_test, metrics, problem_type


# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
# from sklearn.model_selection import train_test_split

# def train_model(X, y, problem_type):

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     if problem_type == "classification":
#         model = RandomForestClassifier(random_state=42)
#         model.fit(X_train, y_train)
#         preds = model.predict(X_test)

#         metrics = {
#             "accuracy": accuracy_score(y_test, preds),
#             "f1": f1_score(y_test, preds, average="weighted")
#         }

#     else:
#         model = RandomForestRegressor(random_state=42)
#         model.fit(X_train, y_train)
#         preds = model.predict(X_test)

#         metrics = {
#             "r2": r2_score(y_test, preds),
#             "mse": mean_squared_error(y_test, preds)
#         }

#     return model, X_train, X_test, y_test, metrics
