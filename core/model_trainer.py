# core/model_trainer.py

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
    Robust problem-type detection using sklearn internals.
    """

    target_type = type_of_target(y)

    if target_type in ["continuous", "continuous-multioutput"]:
        return "regression"

    if target_type in ["binary", "multiclass"]:
        return "classification"

    # Fallback safeguard
    if y.dtype.kind in "if" and y.nunique() > 20:
        return "regression"

    return "classification"


def train_model(X, y):
    """
    Automatically trains classification or regression model.
    Returns:
        model, X_train, X_test, y_train, y_test, metrics, problem_type
    """

    problem_type = detect_problem_type(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if problem_type == "classification":

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

    elif problem_type == "regression":

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

    else:
        raise ValueError("Unsupported problem type.")

    return model, X_train, X_test, y_train, y_test, metrics, problem_type


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
