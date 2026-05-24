import os
import uuid

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils.multiclass import type_of_target

from core.insight_engine import InsightEngine
from core.shap_engine import create_explainer, compute_shap_values, prepare_shap_for_plot
from core.storage_engine import init_storage, save_dataset, save_experiment


# =====================================================
# INIT
# =====================================================

st.set_page_config(layout="wide")
st.title("🔥 Advanced Explainable AI Platform")

init_storage()

if st.button("🔄 Force Recompute Everything"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("Cache cleared successfully.")


# =====================================================
# HELPERS
# =====================================================


def detect_problem_type(y: pd.Series) -> str:
    """Robustly infer whether the target is for classification or regression."""
    target_kind = type_of_target(y)
    if target_kind in {"binary", "multiclass", "multiclass-multioutput"}:
        return "classification"
    if target_kind in {"continuous", "continuous-multioutput"}:
        return "regression"

    # Fallback for uncommon target types: numeric with many unique values is usually regression.
    if np.issubdtype(y.dtype, np.number) and y.nunique() > 20:
        return "regression"
    return "classification"


# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file:
    experiment_id = str(uuid.uuid4())[:8]
    st.write(f"🧪 Experiment ID: {experiment_id}")

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Data Profiling")
    st.write("Shape:", df.shape)
    st.write("Missing values:")
    st.write(df.isna().sum())
    st.write("Summary statistics:")
    st.write(df.describe(include="all"))

    target_col = st.selectbox("Select Target Column", df.columns)

    if target_col:
        modeling_df = df.dropna(subset=[target_col]).copy()

        if modeling_df.empty:
            st.error("No rows remain after removing missing target values.")
            st.stop()

        X = modeling_df.drop(columns=[target_col]).select_dtypes(include="number")
        y = modeling_df[target_col]

        if X.shape[1] == 0:
            st.error("No numeric feature columns detected.")
            st.stop()

        # Most sklearn models in this app cannot train with NaN values.
        row_mask = X.notna().all(axis=1)
        X = X.loc[row_mask]
        y = y.loc[row_mask]

        if len(X) < 5:
            st.error("Not enough valid rows after removing missing feature values.")
            st.stop()

        if np.issubdtype(y.dtype, np.number):
            if st.checkbox("Convert regression target to classification (binning)"):
                bins = st.slider("Number of bins", 2, 10, 3)
                discretizer = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy="quantile")
                y = discretizer.fit_transform(y.values.reshape(-1, 1)).flatten().astype(int)
                st.success("Target converted into categorical bins.")

        problem_type = detect_problem_type(pd.Series(y))
        st.info(f"Detected Problem Type: {problem_type}")

        model_option = st.selectbox(
            "Select Model",
            ["Auto Select", "Random Forest", "Linear/Logistic"],
        )

        stratify = y if problem_type == "classification" and pd.Series(y).nunique() > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=stratify,
        )

        if model_option in {"Auto Select", "Random Forest"}:
            model = (
                RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
                if problem_type == "classification"
                else RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            )
        else:
            model = (
                LogisticRegression(max_iter=2000)
                if problem_type == "classification"
                else LinearRegression()
            )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        if problem_type == "classification":
            metrics = {
                "accuracy": accuracy_score(y_test, preds),
                "f1": f1_score(y_test, preds, average="weighted"),
            }
        else:
            metrics = {"r2": r2_score(y_test, preds)}

        st.subheader("📈 Model Metrics")
        st.write(metrics)

        st.subheader("🔬 SHAP Global Explanation")
        sample_size = st.slider("SHAP sample size", 50, 500, 200)

        shap_values = None
        shap_summary = "SHAP failed"

        try:
            explainer = create_explainer(model, X_train)
            shap_values, _ = compute_shap_values(explainer, X_test, sample_size)

            if len(shap_values.values.shape) == 3:
                shap_values = prepare_shap_for_plot(shap_values)

            fig = plt.figure()
            shap.plots.beeswarm(shap_values, show=False)
            st.pyplot(fig)

            engine = InsightEngine()
            shap_summary = engine.shap_insight(shap_values, X.columns)

            st.subheader("📌 SHAP Insight")
            st.success(shap_summary)

        except Exception as exc:
            st.error(f"SHAP explanation failed: {exc}")

        st.subheader("🔗 Feature Interaction")
        if shap_values is not None:
            try:
                fig2 = plt.figure()
                shap.plots.scatter(shap_values[:, 0], show=False)
                st.pyplot(fig2)
            except Exception:
                st.info("Interaction plot not available.")
        else:
            st.info("Interaction plot unavailable because SHAP did not complete.")

        st.subheader("🔍 Local Explanation")
        if len(X_test) > 0:
            idx = st.slider("Select instance", 0, len(X_test) - 1, 0)
            try:
                local_explainer = create_explainer(model, X_train)
                local_shap, _ = compute_shap_values(local_explainer, X_test.iloc[[idx]], 1)

                # Some classifiers produce 3D SHAP tensors [sample, feature, class].
                # Waterfall plots need one 2D class slice, so we normalize shape here.
                if len(local_shap.values.shape) == 3:
                    local_shap = prepare_shap_for_plot(local_shap)

                fig3 = plt.figure()
                shap.plots.waterfall(local_shap[0], show=False)
                st.pyplot(fig3)
            except Exception as exc:
                st.error(f"Local SHAP failed: {exc}")

        if st.button("💾 Save Experiment"):
            save_dataset(df, experiment_id)

            config = {
                "model": type(model).__name__,
                "problem_type": problem_type,
                "target_column": target_col,
                "shap_sample_size": sample_size,
            }

            exp_path = save_experiment(experiment_id, metrics, config, shap_summary)
            st.success(f"Experiment saved at: {exp_path}")

            report_path = os.path.join(exp_path, "report.txt")
            with open(report_path, "r", encoding="utf-8") as report_file:
                report_content = report_file.read()

            st.download_button(
                label="⬇ Download Report",
                data=report_content,
                file_name=f"{experiment_id}_report.txt",
                mime="text/plain",
            )
