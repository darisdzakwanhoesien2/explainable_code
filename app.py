# https://chatgpt.com/c/698e8032-0a14-8320-918b-b2fedd186f4b

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import uuid
import os
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.model_selection import train_test_split

from core.shap_engine import (
    create_explainer,
    compute_shap_values,
    prepare_shap_for_plot
)
from core.storage_engine import (
    init_storage,
    save_dataset,
    save_experiment
)
from core.insight_engine import InsightEngine


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
# FILE UPLOAD
# =====================================================

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file:

    experiment_id = str(uuid.uuid4())[:8]
    st.write(f"🧪 Experiment ID: {experiment_id}")

    df = pd.read_csv(uploaded_file)

    # =====================================================
    # DATA PROFILING
    # =====================================================

    st.subheader("📊 Data Profiling")

    st.write("Shape:", df.shape)
    st.write("Missing values:")
    st.write(df.isna().sum())
    st.write("Summary statistics:")
    st.write(df.describe())

    target_col = st.selectbox("Select Target Column", df.columns)

    if target_col:

        X = df.drop(columns=[target_col]).select_dtypes(include="number")
        y = df[target_col]

        if X.shape[1] == 0:
            st.error("No numeric feature columns detected.")
            st.stop()

        # =====================================================
        # AUTO BINNING
        # =====================================================

        if np.issubdtype(y.dtype, np.number):

            if st.checkbox("Convert regression target to classification (binning)"):

                bins = st.slider("Number of bins", 2, 10, 3)

                discretizer = KBinsDiscretizer(
                    n_bins=bins,
                    encode="ordinal",
                    strategy="quantile"
                )

                y = discretizer.fit_transform(
                    y.values.reshape(-1, 1)
                ).flatten()

                y = y.astype(int)

                st.success("Target converted into categorical bins.")

        # =====================================================
        # PROBLEM TYPE
        # =====================================================

        problem_type = "classification" if len(np.unique(y)) < 20 else "regression"

        st.info(f"Detected Problem Type: {problem_type}")

        # =====================================================
        # MODEL SELECTION
        # =====================================================

        model_option = st.selectbox(
            "Select Model",
            ["Auto Select", "Random Forest", "Linear/Logistic"]
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        if model_option == "Auto Select":
            model = (
                RandomForestClassifier(n_estimators=200)
                if problem_type == "classification"
                else RandomForestRegressor(n_estimators=200)
            )

        elif model_option == "Random Forest":
            model = (
                RandomForestClassifier(n_estimators=200)
                if problem_type == "classification"
                else RandomForestRegressor(n_estimators=200)
            )

        else:
            model = (
                LogisticRegression(max_iter=2000)
                if problem_type == "classification"
                else LinearRegression()
            )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # =====================================================
        # METRICS
        # =====================================================

        if problem_type == "classification":
            metrics = {
                "accuracy": accuracy_score(y_test, preds),
                "f1": f1_score(y_test, preds, average="weighted")
            }
        else:
            metrics = {
                "r2": r2_score(y_test, preds)
            }

        st.subheader("📈 Model Metrics")
        st.write(metrics)

        # =====================================================
        # SHAP GLOBAL
        # =====================================================

        st.subheader("🔬 SHAP Global Explanation")

        sample_size = st.slider("SHAP sample size", 50, 500, 200)

        try:
            explainer = create_explainer(model, X_train)
            shap_values, X_explain = compute_shap_values(
                explainer,
                X_test,
                sample_size
            )

            if len(shap_values.values.shape) == 3:
                shap_values = prepare_shap_for_plot(shap_values)

            fig = plt.figure()
            shap.plots.beeswarm(shap_values, show=False)
            st.pyplot(fig)

            engine = InsightEngine()
            shap_summary = engine.shap_insight(shap_values, X.columns)

            st.subheader("📌 SHAP Insight")
            st.success(shap_summary)

        except Exception as e:
            st.error(f"SHAP explanation failed: {e}")
            shap_summary = "SHAP failed"

        # =====================================================
        # FEATURE INTERACTION
        # =====================================================

        st.subheader("🔗 Feature Interaction")

        try:
            fig2 = plt.figure()
            shap.plots.scatter(shap_values[:, 0], show=False)
            st.pyplot(fig2)
        except:
            st.info("Interaction plot not available.")

        # =====================================================
        # LOCAL EXPLANATION
        # =====================================================

        st.subheader("🔍 Local Explanation")

        if len(X_test) > 0:

            idx = st.slider("Select instance", 0, len(X_test) - 1, 0)

            try:
                local_explainer = create_explainer(model, X_train)
                local_shap, _ = compute_shap_values(
                    local_explainer,
                    X_test.iloc[[idx]],
                    1
                )

                if len(local_shap.values.shape) == 3:
                    local_shap = prepare_shap_for_plot(local_shap)

                fig3 = plt.figure()
                shap.plots.waterfall(local_shap[0], show=False)
                st.pyplot(fig3)

            except Exception as e:
                st.error(f"Local SHAP failed: {e}")

        # =====================================================
        # SAVE EXPERIMENT
        # =====================================================

        if st.button("💾 Save Experiment"):

            save_dataset(df, experiment_id)

            config = {
                "model": type(model).__name__,
                "problem_type": problem_type,
                "target_column": target_col,
                "shap_sample_size": sample_size
            }

            exp_path = save_experiment(
                experiment_id,
                metrics,
                config,
                shap_summary
            )

            st.success(f"Experiment saved at: {exp_path}")

            report_path = os.path.join(exp_path, "report.txt")

            with open(report_path, "r") as f:
                report_content = f.read()

            st.download_button(
                label="⬇ Download Report",
                data=report_content,
                file_name=f"{experiment_id}_report.txt",
                mime="text/plain"
            )


# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import shap
# import uuid
# from sklearn.preprocessing import KBinsDiscretizer
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.linear_model import LogisticRegression, LinearRegression
# from sklearn.metrics import accuracy_score, f1_score, r2_score
# from sklearn.model_selection import train_test_split

# from core.shap_engine import (
#     create_explainer,
#     compute_shap_values,
#     prepare_shap_for_plot
# )
# from core.insight_engine import InsightEngine


# st.set_page_config(layout="wide")
# st.title("🔥 Advanced Explainable AI Platform")

# # =====================================================
# # EXPERIMENT TRACKING
# # =====================================================

# if "experiments" not in st.session_state:
#     st.session_state.experiments = []

# experiment_id = str(uuid.uuid4())[:8]
# st.write(f"Experiment ID: {experiment_id}")

# # =====================================================
# # FILE UPLOAD
# # =====================================================

# uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# if uploaded_file:

#     df = pd.read_csv(uploaded_file)

#     st.subheader("📊 Data Profiling")

#     st.write("Shape:", df.shape)
#     st.write("Missing values:")
#     st.write(df.isna().sum())
#     st.write("Summary statistics:")
#     st.write(df.describe())

#     target_col = st.selectbox("Select Target", df.columns)

#     if target_col:

#         X = df.drop(columns=[target_col]).select_dtypes(include="number")
#         y = df[target_col]

#         # =====================================================
#         # AUTO BINNING OPTION
#         # =====================================================

#         if np.issubdtype(y.dtype, np.number):

#             if st.checkbox("Convert regression target to classification (binning)"):

#                 bins = st.slider("Number of bins", 2, 10, 3)

#                 discretizer = KBinsDiscretizer(
#                     n_bins=bins,
#                     encode="ordinal",
#                     strategy="quantile"
#                 )

#                 y = discretizer.fit_transform(y.values.reshape(-1, 1)).flatten()
#                 y = y.astype(int)

#                 st.success("Target converted to categorical bins.")

#         # =====================================================
#         # MODEL SELECTION
#         # =====================================================

#         problem_type = "classification" if len(np.unique(y)) < 20 else "regression"

#         model_option = st.selectbox(
#             "Model",
#             ["Auto Select", "Random Forest", "Logistic/Linear"]
#         )

#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42
#         )

#         if model_option == "Auto Select":
#             model = RandomForestClassifier() if problem_type == "classification" else RandomForestRegressor()

#         elif model_option == "Random Forest":
#             model = RandomForestClassifier() if problem_type == "classification" else RandomForestRegressor()

#         else:
#             model = LogisticRegression(max_iter=2000) if problem_type == "classification" else LinearRegression()

#         model.fit(X_train, y_train)

#         preds = model.predict(X_test)

#         # =====================================================
#         # METRICS
#         # =====================================================

#         if problem_type == "classification":
#             acc = accuracy_score(y_test, preds)
#             f1 = f1_score(y_test, preds, average="weighted")
#             st.write({"accuracy": acc, "f1": f1})
#         else:
#             r2 = r2_score(y_test, preds)
#             st.write({"r2": r2})

#         # Save experiment
#         st.session_state.experiments.append({
#             "id": experiment_id,
#             "model": type(model).__name__,
#             "problem_type": problem_type
#         })

#         # =====================================================
#         # SHAP GLOBAL
#         # =====================================================

#         st.subheader("🔬 SHAP Global Explanation")

#         sample_size = st.slider("SHAP sample size", 50, 500, 200)

#         try:
#             explainer = create_explainer(model, X_train)
#             shap_values, X_explain = compute_shap_values(
#                 explainer,
#                 X_test,
#                 sample_size
#             )

#             if len(shap_values.values.shape) == 3:
#                 shap_values = prepare_shap_for_plot(shap_values)

#             fig = plt.figure()
#             shap.plots.beeswarm(shap_values, show=False)
#             st.pyplot(fig)

#         except Exception as e:
#             st.error(f"SHAP explanation failed: {e}")

#         # =====================================================
#         # FEATURE INTERACTION
#         # =====================================================

#         st.subheader("🔗 Feature Interaction")

#         try:
#             fig2 = plt.figure()
#             shap.plots.scatter(shap_values[:, 0], show=False)
#             st.pyplot(fig2)
#         except:
#             st.info("Feature interaction not available for this model.")

#         # =====================================================
#         # LOCAL EXPLANATION
#         # =====================================================

#         st.subheader("🔍 Local Explanation")

#         idx = st.slider("Select instance", 0, len(X_test) - 1, 0)

#         try:
#             local_explainer = create_explainer(model, X_train)
#             local_shap, _ = compute_shap_values(
#                 local_explainer,
#                 X_test.iloc[[idx]],
#                 1
#             )

#             if len(local_shap.values.shape) == 3:
#                 local_shap = prepare_shap_for_plot(local_shap)

#             fig3 = plt.figure()
#             shap.plots.waterfall(local_shap[0], show=False)
#             st.pyplot(fig3)

#         except Exception as e:
#             st.error(f"Local SHAP failed: {e}")

# # =====================================================
# # EXPERIMENT HISTORY
# # =====================================================

# if st.session_state.experiments:
#     st.subheader("🧪 Experiment History")
#     st.write(st.session_state.experiments)


# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import shap

# from core.model_trainer import train_model, detect_problem_type
# from core.shap_engine import (
#     create_explainer,
#     compute_shap_values,
#     prepare_shap_for_plot
# )
# from core.insight_engine import InsightEngine
# from core.bias_detector import detect_class_imbalance
# from core.evaluation_engine import stability_analysis


# # =====================================================
# # PAGE CONFIG
# # =====================================================

# st.set_page_config(layout="wide")
# st.title("🔥 Unified Explainable Analytics Framework")


# # =====================================================
# # FORCE RECOMPUTE BUTTON
# # =====================================================

# if st.button("🔄 Force Recompute Everything"):
#     st.cache_data.clear()
#     st.cache_resource.clear()
#     st.success("All cached computations cleared.")


# # =====================================================
# # FILE UPLOAD
# # =====================================================

# uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

# if uploaded_file:

#     df = pd.read_csv(uploaded_file)

#     st.subheader("📂 Dataset Preview")
#     st.dataframe(df.head())

#     target_col = st.selectbox("Select Target Column", df.columns)

#     if target_col:

#         X = df.drop(columns=[target_col])
#         X = X.select_dtypes(include="number")
#         y = df[target_col]

#         if X.shape[1] == 0:
#             st.error("No numeric feature columns detected.")
#             st.stop()

#         # =====================================================
#         # PROBLEM TYPE DETECTION + OVERRIDE
#         # =====================================================

#         auto_type = detect_problem_type(y)

#         override = st.selectbox(
#             "Problem Type",
#             ["Auto Detect", "Classification", "Regression"]
#         )

#         if override == "Auto Detect":
#             problem_type = auto_type
#         else:
#             problem_type = override.lower()

#         st.info(f"Detected / Selected Problem Type: {problem_type}")

#         # =====================================================
#         # TRAIN MODEL (CACHED)
#         # =====================================================

#         model, X_train, X_test, y_train, y_test, metrics, detected_type = train_model(X, y)

#         st.subheader("📊 Model Metrics")
#         st.write(metrics)

#         # =====================================================
#         # MODEL INSIGHT
#         # =====================================================

#         engine = InsightEngine()
#         model_insight = engine.model_insight(problem_type, metrics)

#         st.subheader("🧠 Model Insight")
#         st.success(model_insight)

#         # =====================================================
#         # BIAS CHECK (CLASSIFICATION ONLY)
#         # =====================================================

#         if problem_type == "classification":
#             imbalance_note = detect_class_imbalance(y)
#             st.warning(imbalance_note)

#         # =====================================================
#         # STABILITY ANALYSIS
#         # =====================================================

#         stability = stability_analysis(model, X_train, y_train, problem_type)

#         st.subheader("📈 Stability Analysis")
#         st.write(f"Model score standard deviation across runs: {stability:.4f}")

#         # =====================================================
#         # SHAP EXPLANATION
#         # =====================================================

#         st.subheader("🔬 SHAP Global Explanation")

#         sample_size = st.slider("SHAP sample size", 50, 500, 200)

#         try:
#             explainer = create_explainer(model, X_train)

#             shap_values, X_explain = compute_shap_values(
#                 explainer,
#                 X_test,
#                 sample_size
#             )

#             # -------------------------------
#             # MULTI-CLASS HANDLING
#             # -------------------------------
#             if len(shap_values.values.shape) == 3:

#                 class_labels = model.classes_

#                 selected_class = st.selectbox(
#                     "Select class for SHAP explanation",
#                     range(len(class_labels)),
#                     format_func=lambda x: f"Class {class_labels[x]}"
#                 )

#                 shap_values = prepare_shap_for_plot(
#                     shap_values,
#                     class_index=selected_class
#                 )

#             else:
#                 shap_values = prepare_shap_for_plot(shap_values)

#             # -------------------------------
#             # BEESWARM PLOT
#             # -------------------------------
#             fig = plt.figure()
#             shap.plots.beeswarm(shap_values, show=False)
#             st.pyplot(fig)

#             shap_summary = engine.shap_insight(shap_values, X.columns)

#             st.subheader("📌 SHAP Insight")
#             st.info(shap_summary)

#         except Exception as e:
#             st.error(f"SHAP explanation failed: {e}")

#         # =====================================================
#         # LOCAL EXPLANATION (OPTIONAL)
#         # =====================================================

#         st.subheader("🔍 Local Explanation")

#         if len(X_test) > 0:

#             index = st.slider(
#                 "Select instance index",
#                 0,
#                 len(X_test) - 1,
#                 0
#             )

#             try:
#                 local_explainer = create_explainer(model, X_train)
#                 local_shap_values, local_X = compute_shap_values(
#                     local_explainer,
#                     X_test.iloc[[index]],
#                     sample_size=1
#                 )

#                 if len(local_shap_values.values.shape) == 3:
#                     local_shap_values = prepare_shap_for_plot(local_shap_values)

#                 fig2 = plt.figure()
#                 shap.plots.waterfall(local_shap_values[0], show=False)
#                 st.pyplot(fig2)

#             except Exception as e:
#                 st.error(f"Local SHAP failed: {e}")


# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import shap

# from core.model_trainer import train_model
# from core.shap_engine import (
#     create_explainer,
#     compute_shap_values,
#     prepare_shap_for_plot
# )
# from core.insight_engine import InsightEngine
# from core.bias_detector import detect_class_imbalance
# from core.evaluation_engine import stability_analysis

# st.set_page_config(layout="wide")
# st.title("🔥 Unified Explainable Analytics Framework")

# # Force refresh button
# if st.button("🔄 Force Recompute Everything"):
#     st.cache_data.clear()
#     st.cache_resource.clear()
#     st.success("Cache cleared!")

# uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

# if uploaded_file:

#     df = pd.read_csv(uploaded_file)
#     st.subheader("Dataset Preview")
#     st.dataframe(df.head())

#     target_col = st.selectbox("Select Target Column", df.columns)

#     if target_col:

#         X = df.drop(columns=[target_col])
#         X = X.select_dtypes(include="number")
#         y = df[target_col]

#         if X.shape[1] == 0:
#             st.error("No numeric feature columns detected.")
#             st.stop()

#         # ===============================
#         # TRAIN MODEL (CACHED)
#         # ===============================
#         model, X_train, X_test, y_train, y_test, metrics, problem_type = train_model(X, y)

#         st.success(f"Detected Problem Type: {problem_type}")

#         st.subheader("📊 Model Metrics")
#         st.write(metrics)

#         # ===============================
#         # INSIGHT ENGINE
#         # ===============================
#         engine = InsightEngine()
#         model_insight = engine.model_insight(problem_type, metrics)

#         st.subheader("🧠 Model Insight")
#         st.info(model_insight)

#         # ===============================
#         # BIAS CHECK
#         # ===============================
#         if problem_type == "classification":
#             imbalance_note = detect_class_imbalance(y)
#             st.warning(imbalance_note)

#         # ===============================
#         # STABILITY
#         # ===============================
#         stability = stability_analysis(model, X_train, y_train, problem_type)
#         st.subheader("📈 Stability Analysis")
#         st.write(f"Score standard deviation across runs: {stability:.4f}")

#         # ===============================
#         # SHAP
#         # ===============================
#         st.subheader("🔬 SHAP Global Explanation")

#         sample_size = st.slider("SHAP sample size", 50, 500, 200)

#         try:
#             explainer = create_explainer(model, X_train)
#             shap_values, X_explain = compute_shap_values(
#                 explainer,
#                 X_test,
#                 sample_size
#             )

#             if len(shap_values.values.shape) == 3:

#                 class_labels = model.classes_

#                 selected_class = st.selectbox(
#                     "Select class for SHAP explanation",
#                     range(len(class_labels)),
#                     format_func=lambda x: f"Class {class_labels[x]}"
#                 )

#                 shap_values = prepare_shap_for_plot(
#                     shap_values,
#                     class_index=selected_class
#                 )

#             else:
#                 shap_values = prepare_shap_for_plot(shap_values)

#             fig = plt.figure()
#             shap.plots.beeswarm(shap_values, show=False)
#             st.pyplot(fig)

#             shap_summary = engine.shap_insight(shap_values, X.columns)

#             st.subheader("📌 SHAP Insight")
#             st.success(shap_summary)

#         except Exception as e:
#             st.error(f"SHAP explanation failed: {e}")


# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import shap

# from core.model_trainer import train_model
# from core.shap_engine import compute_shap
# from core.shap_engine import prepare_shap_for_plot
# from core.insight_engine import InsightEngine
# from core.bias_detector import detect_class_imbalance
# from core.evaluation_engine import stability_analysis

# st.set_page_config(layout="wide")
# st.title("🔥 Unified Explainable Analytics Framework")

# uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

# if uploaded_file:

#     df = pd.read_csv(uploaded_file)
#     st.subheader("Dataset Preview")
#     st.dataframe(df.head())

#     target_col = st.selectbox("Select Target Column", df.columns)

#     if target_col:

#         X = df.drop(columns=[target_col])
#         X = X.select_dtypes(include=np.number)
#         y = df[target_col]

#         if X.shape[1] == 0:
#             st.error("No numeric feature columns detected.")
#             st.stop()

#         # ===============================
#         # TRAIN MODEL
#         # ===============================
#         model, X_train, X_test, y_train, y_test, metrics, problem_type = train_model(X, y)

#         st.success(f"Detected Problem Type: {problem_type}")

#         st.subheader("📊 Model Metrics")
#         st.write(metrics)

#         # ===============================
#         # INSIGHT ENGINE
#         # ===============================
#         engine = InsightEngine()
#         model_insight = engine.model_insight(problem_type, metrics)

#         st.subheader("🧠 Model Insight")
#         st.info(model_insight)

#         # ===============================
#         # BIAS CHECK (classification only)
#         # ===============================
#         if problem_type == "classification":
#             imbalance_note = detect_class_imbalance(y)
#             st.warning(imbalance_note)

#         # ===============================
#         # STABILITY ANALYSIS
#         # ===============================
#         stability = stability_analysis(model, X_train, y_train, problem_type)
#         st.subheader("📈 Stability Analysis")
#         st.write(f"Model score standard deviation across runs: {stability:.4f}")

#         # ===============================
#         # SHAP EXPLANATION
#         # ===============================
#         st.subheader("🔬 SHAP Global Explanation")

#         try:
#             shap_values, X_explain = compute_shap(model, X_train, X_test)

#             # Handle multi-class safely
#             if len(shap_values.values.shape) == 3:

#                 class_labels = model.classes_

#                 selected_class = st.selectbox(
#                     "Select class for SHAP explanation",
#                     range(len(class_labels)),
#                     format_func=lambda x: f"Class {class_labels[x]}"
#                 )

#                 shap_values = prepare_shap_for_plot(
#                     shap_values,
#                     class_index=selected_class
#                 )

#             else:
#                 shap_values = prepare_shap_for_plot(shap_values)

#             fig = plt.figure()
#             shap.plots.beeswarm(shap_values, show=False)
#             st.pyplot(fig)

#             shap_summary = engine.shap_insight(shap_values, X.columns)

#             st.subheader("📌 SHAP Insight")
#             st.success(shap_summary)

#         except Exception as e:
#             st.error(f"SHAP explanation failed: {e}")


# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import shap

# from core.data_processor import detect_problem_type, split_features_target
# from core.model_trainer import train_model
# from core.shap_engine import compute_shap
# from core.insight_engine import InsightEngine
# from core.evaluation_engine import stability_analysis
# from core.bias_detector import detect_class_imbalance
# from report.report_generator import generate_report

# st.set_page_config(layout="wide")
# st.title("🔥 Unified Explainable Analytics Framework")

# uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# if uploaded_file:

#     df = pd.read_csv(uploaded_file)
#     st.dataframe(df.head())

#     target_col = st.selectbox("Select Target Column", df.columns)

#     if target_col:

#         X, y = split_features_target(df, target_col)
#         problem_type = detect_problem_type(y)

#         model, X_train, X_test, y_test, metrics = train_model(X, y, problem_type)

#         st.subheader("📊 Metrics")
#         st.write(metrics)

#         engine = InsightEngine()

#         model_insight = engine.model_insight(problem_type, metrics)
#         st.subheader("🧠 Model Insight")
#         st.info(model_insight)

#         # SHAP
#         st.subheader("🔬 SHAP Explanation")
#         shap_values, X_explain = compute_shap(model, X_train, X_test)

#         fig = plt.figure()
#         shap.plots.beeswarm(shap_values, show=False)
#         st.pyplot(fig)

#         shap_summary = engine.shap_insight(shap_values, X.columns)

#         # Stability
#         stability = stability_analysis(model, X, y, problem_type)

#         # Bias
#         if problem_type == "classification":
#             bias_note = detect_class_imbalance(y)
#             st.warning(bias_note)

#         # Report
#         report = generate_report(
#             problem_type,
#             metrics,
#             model_insight,
#             shap_summary,
#             stability
#         )

#         st.subheader("📄 Full Structured Report")
#         st.code(report)


# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import shap

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# from explain.insight_engine import InsightEngine
# from explain.clustering_explainer import run_clustering
# from explain.model_explainer import (
#     train_classification_model,
#     compute_shap_values
# )

# st.set_page_config(layout="wide")
# st.title("🔍 Explainable Analytics Dashboard")

# engine = InsightEngine()

# uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# if uploaded_file:

#     df = pd.read_csv(uploaded_file)
#     st.write("### Dataset Preview")
#     st.dataframe(df.head())

#     numeric_df = df.select_dtypes(include=np.number)

#     tab1, tab2 = st.tabs(["📊 Clustering", "🤖 Classification"])

#     # =============================
#     # CLUSTERING
#     # =============================
#     with tab1:

#         st.header("KMeans Clustering")

#         if len(numeric_df.columns) < 2:
#             st.warning("Need at least 2 numeric columns for clustering.")
#         else:

#             k = st.slider("Number of clusters", 2, 10, 3)

#             scaler = StandardScaler()
#             X_scaled = scaler.fit_transform(numeric_df)

#             model, labels, sil, sizes = run_clustering(X_scaled, k)

#             fig, ax = plt.subplots()
#             ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels)
#             st.pyplot(fig)

#             st.subheader("🧠 Explanation Report")

#             report = engine.clustering_insight(sil, sizes)

#             st.info(report["summary"])
#             st.warning(report["distribution"])
#             st.success(report["suggestion"])

#             st.write(f"Silhouette Score: {sil:.3f}")
#             st.write(f"Cluster Sizes: {sizes}")

#     # =============================
#     # CLASSIFICATION
#     # =============================
#     with tab2:

#         st.header("RandomForest Classification + SHAP")

#         target_col = st.selectbox("Select Target Column", df.columns)

#         if target_col:

#             X = df.drop(columns=[target_col])
#             X = X.select_dtypes(include=np.number)
#             y = df[target_col]

#             if X.shape[1] == 0:
#                 st.error("No numeric features available.")
#             else:

#                 X_train, X_test, y_train, y_test = train_test_split(
#                     X, y, test_size=0.2, random_state=42
#                 )

#                 model, acc, f1 = train_classification_model(
#                     X_train, X_test, y_train, y_test
#                 )

#                 st.write("### Model Performance")
#                 st.write(f"Accuracy: {acc:.3f}")
#                 st.write(f"F1 Score: {f1:.3f}")

#                 report = engine.classification_insight(acc, f1)

#                 st.subheader("🧠 Insight Report")
#                 st.info(report["summary"])
#                 st.warning(report["risk"])
#                 st.success(report["confidence"])

#                 # =============================
#                 # SHAP (FIXED VERSION)
#                 # =============================

#                 st.subheader("🔬 SHAP Global Explanation")

#                 try:
#                     shap_values = compute_shap_values(
#                         model,
#                         X_train,
#                         X_test
#                     )

#                     fig2 = plt.figure()
#                     shap.plots.beeswarm(shap_values, show=False)
#                     st.pyplot(fig2)

#                 except Exception as e:
#                     st.error(f"SHAP failed: {e}")


# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import shap

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# from explain.insight_engine import InsightEngine
# from explain.clustering_explainer import run_clustering
# from explain.model_explainer import train_model, compute_shap

# st.set_page_config(layout="wide")
# st.title("🔍 Explainable Analytics Dashboard")

# engine = InsightEngine()

# uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# if uploaded_file:

#     df = pd.read_csv(uploaded_file)
#     st.write("### Dataset Preview")
#     st.dataframe(df.head())

#     numeric_df = df.select_dtypes(include=np.number)

#     tab1, tab2 = st.tabs(["📊 Clustering", "🤖 Classification"])

#     # =====================
#     # CLUSTERING TAB
#     # =====================
#     with tab1:
#         st.header("KMeans Clustering")

#         k = st.slider("Number of clusters", 2, 10, 3)

#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(numeric_df)

#         model, labels, sil, sizes = run_clustering(X_scaled, k)

#         fig, ax = plt.subplots()
#         ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels)
#         st.pyplot(fig)

#         st.subheader("🧠 Explanation Report")

#         report = engine.clustering_insight(sil, sizes)

#         st.info(f"Summary: {report['summary']}")
#         st.warning(f"Distribution: {report['distribution']}")
#         st.success(f"Suggestion: {report['suggestion']}")

#         st.write(f"Silhouette Score: {sil:.3f}")
#         st.write(f"Cluster Sizes: {sizes}")

#     # =====================
#     # CLASSIFICATION TAB
#     # =====================
#     with tab2:
#         st.header("RandomForest Classification")

#         target_col = st.selectbox("Select Target Column", df.columns)

#         if target_col:

#             X = df.drop(columns=[target_col])
#             X = X.select_dtypes(include=np.number)
#             y = df[target_col]

#             X_train, X_test, y_train, y_test = train_test_split(
#                 X, y, test_size=0.2, random_state=42
#             )

#             model, acc, f1 = train_model(X_train, X_test, y_train, y_test)

#             st.write("### Model Performance")
#             st.write(f"Accuracy: {acc:.3f}")
#             st.write(f"F1 Score: {f1:.3f}")

#             st.subheader("🧠 Insight Report")

#             report = engine.classification_insight(acc, f1)

#             st.info(f"Summary: {report['summary']}")
#             st.warning(f"Risk: {report['risk']}")
#             st.success(f"Confidence: {report['confidence']}")

#             # SHAP
#             st.subheader("🔬 SHAP Explanation")

#             shap_values = compute_shap(model, X_test)

#             fig2 = plt.figure()
#             shap.summary_plot(shap_values, X_test, show=False)
#             st.pyplot(fig2)