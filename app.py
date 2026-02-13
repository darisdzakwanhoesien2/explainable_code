import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from explain.insight_engine import InsightEngine
from explain.clustering_explainer import run_clustering
from explain.model_explainer import (
    train_classification_model,
    compute_shap_values
)

st.set_page_config(layout="wide")
st.title("🔍 Explainable Analytics Dashboard")

engine = InsightEngine()

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    numeric_df = df.select_dtypes(include=np.number)

    tab1, tab2 = st.tabs(["📊 Clustering", "🤖 Classification"])

    # =============================
    # CLUSTERING
    # =============================
    with tab1:

        st.header("KMeans Clustering")

        if len(numeric_df.columns) < 2:
            st.warning("Need at least 2 numeric columns for clustering.")
        else:

            k = st.slider("Number of clusters", 2, 10, 3)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(numeric_df)

            model, labels, sil, sizes = run_clustering(X_scaled, k)

            fig, ax = plt.subplots()
            ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels)
            st.pyplot(fig)

            st.subheader("🧠 Explanation Report")

            report = engine.clustering_insight(sil, sizes)

            st.info(report["summary"])
            st.warning(report["distribution"])
            st.success(report["suggestion"])

            st.write(f"Silhouette Score: {sil:.3f}")
            st.write(f"Cluster Sizes: {sizes}")

    # =============================
    # CLASSIFICATION
    # =============================
    with tab2:

        st.header("RandomForest Classification + SHAP")

        target_col = st.selectbox("Select Target Column", df.columns)

        if target_col:

            X = df.drop(columns=[target_col])
            X = X.select_dtypes(include=np.number)
            y = df[target_col]

            if X.shape[1] == 0:
                st.error("No numeric features available.")
            else:

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                model, acc, f1 = train_classification_model(
                    X_train, X_test, y_train, y_test
                )

                st.write("### Model Performance")
                st.write(f"Accuracy: {acc:.3f}")
                st.write(f"F1 Score: {f1:.3f}")

                report = engine.classification_insight(acc, f1)

                st.subheader("🧠 Insight Report")
                st.info(report["summary"])
                st.warning(report["risk"])
                st.success(report["confidence"])

                # =============================
                # SHAP (FIXED VERSION)
                # =============================

                st.subheader("🔬 SHAP Global Explanation")

                try:
                    shap_values = compute_shap_values(
                        model,
                        X_train,
                        X_test
                    )

                    fig2 = plt.figure()
                    shap.plots.beeswarm(shap_values, show=False)
                    st.pyplot(fig2)

                except Exception as e:
                    st.error(f"SHAP failed: {e}")


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