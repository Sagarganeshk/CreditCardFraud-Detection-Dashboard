import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# ---------------------------
# Utility: Get metrics
# ---------------------------
def get_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
    }

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("💳 Fraud Detection Dashboard")

mode = st.radio(
    "Choose Mode:",
    ["⚡ Use Pretrained Models (Fast)", "🛠️ Upload & Train Models (Custom)"]
)

# ---------------------------
# FAST MODE (Pretrained)
# ---------------------------
if mode == "⚡ Use Pretrained Models (Fast)":
    st.success("✅ Running with pretrained models!")

    # Load pretrained
    log_reg = joblib.load("log_reg.pkl")
    rf = joblib.load("random_forest.pkl")
    scaler = joblib.load("scaler.pkl")
    df_default = pd.read_csv("creditcard.csv")

    use_default = st.checkbox("Use Preloaded Credit Card Dataset", value=True)

    if use_default:
        df = df_default.copy()
        st.info("Using preloaded **creditcard.csv**")
    else:
        uploaded = st.file_uploader("Upload your CSV (must include 'Class')", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
        else:
            st.warning("Upload a CSV file to continue")
            st.stop()

    X = df.drop("Class", axis=1)
    y = df["Class"]
    X_scaled = scaler.transform(X)

    # Predictions
    y_pred_log = log_reg.predict(X_scaled)
    y_pred_rf = rf.predict(X_scaled)

    # Metrics
    perf = {
        "Logistic Regression": get_metrics(y, y_pred_log),
        "Random Forest": get_metrics(y, y_pred_rf),
    }
    perf_df = pd.DataFrame(perf).T

    # Fraud counts
    fraud_counts = {
        "Logistic Regression": np.sum(y_pred_log),
        "Random Forest": np.sum(y_pred_rf),
        "Actual": np.sum(y)
    }

# ---------------------------
# CUSTOM MODE (Train fresh)
# ---------------------------
else:
    uploaded = st.file_uploader("Upload your CSV (must include 'Class')", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.success("✅ File uploaded successfully!")

        X = df.drop("Class", axis=1)
        y = df["Class"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train Logistic Regression
        start = time.time()
        log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
        log_reg.fit(X_train, y_train)
        log_time = time.time() - start

        # Train Random Forest
        start = time.time()
        rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
        rf.fit(X_train, y_train)
        rf_time = time.time() - start

        # Predictions
        y_pred_log = log_reg.predict(X_test)
        y_pred_rf = rf.predict(X_test)

        # Metrics
        perf = {
            "Logistic Regression": get_metrics(y_test, y_pred_log),
            "Random Forest": get_metrics(y_test, y_pred_rf),
        }
        perf_df = pd.DataFrame(perf).T

        # Training times
        st.info(f"⏱️ Logistic Regression trained in {log_time:.2f} sec")
        st.info(f"⏱️ Random Forest trained in {rf_time:.2f} sec")

        # Fraud counts
        fraud_counts = {
            "Logistic Regression": np.sum(y_pred_log),
            "Random Forest": np.sum(y_pred_rf),
            "Actual": np.sum(y_test)
        }
    else:
        st.warning("Upload a CSV to train models.")
        st.stop()

# ---------------------------
# Dashboard Display
# ---------------------------
st.subheader("🏆 Model Performance Comparison")

cols = st.columns(len(perf_df))
for col, (model_name, row) in zip(cols, perf_df.iterrows()):
    with col:
        st.metric(
            label=model_name,
            value=f"Accuracy: {row['Accuracy']:.3f}",
            delta=f"F1: {row['F1-Score']:.3f}"
        )

# Chart
st.subheader("📊 Metrics Overview")
fig, ax = plt.subplots(figsize=(8,4))
perf_df[["Accuracy", "Precision", "Recall", "F1-Score"]].plot(kind="bar", ax=ax)
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0,1)
st.pyplot(fig)

# Fraud counts
st.subheader("🔎 Fraud Cases Detected")
st.write(pd.DataFrame([fraud_counts]))

# Preview predictions
st.subheader("📥 Predictions Preview")
if mode == "⚡ Use Pretrained Models (Fast)":
    preview_df = df.head(20).copy()
else:
    preview_df = X_test[:20].copy()
preview_df["LogReg_Pred"] = y_pred_log[:20]
preview_df["RF_Pred"] = y_pred_rf[:20]
st.dataframe(preview_df)
