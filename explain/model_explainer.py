# explain/model_explainer.py

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
import shap


def train_classification_model(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")

    return model, acc, f1


def train_regression_model(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)

    return model, r2, mse


def compute_shap_values(model, X_background, X_explain):
    """
    Uses modern SHAP API (stable for classification + regression)
    """
    explainer = shap.Explainer(model, X_background)
    shap_values = explainer(X_explain)
    return shap_values


# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, f1_score
# import shap

# def train_model(X_train, X_test, y_train, y_test):
#     model = RandomForestClassifier(random_state=42)
#     model.fit(X_train, y_train)

#     preds = model.predict(X_test)

#     acc = accuracy_score(y_test, preds)
#     f1 = f1_score(y_test, preds, average="weighted")

#     return model, acc, f1

# def compute_shap(model, X):
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(X)
#     return shap_values