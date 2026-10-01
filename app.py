import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

try:
    from imblearn.over_sampling import SMOTE
except ImportError:  # app still runs without SMOTE
    SMOTE = None

CSV_PATH = "loan_data.csv"
TARGET = "loan_status"
ID_COL = "applicant_id"

st.set_page_config(page_title="Loan Default Prediction", page_icon="🏦", layout="wide")


@st.cache_resource(show_spinner="Models train ho rahe hain...")
def build_models():
    """Notebook ka same pipeline: dropna -> one-hot -> label encode -> split -> SMOTE -> GNB + KNN."""
    df = pd.read_csv(CSV_PATH).dropna()
    if ID_COL in df.columns:
        df = df.drop(columns=ID_COL)

    cat_cols = list(df.select_dtypes(include=["object", "string"]).columns.drop(TARGET))
    raw_features = df.drop(columns=TARGET)

    # UI ke liye har raw column ki info
    ui_info = {}
    for c in raw_features.columns:
        if c in cat_cols:
            ui_info[c] = {"type": "cat", "options": sorted(raw_features[c].unique().tolist())}
        else:
            s = raw_features[c]
            is_int = pd.api.types.is_integer_dtype(s)
            ui_info[c] = {
                "type": "num", "is_int": is_int,
                "min": float(s.min()), "max": float(s.max()), "median": float(s.median()),
            }

    X = pd.get_dummies(raw_features, columns=cat_cols, dtype=int, drop_first=True)
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    if SMOTE is not None:
        X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

    gnb = GaussianNB().fit(X_train, y_train)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=3).fit(X_train_s, y_train)

    metrics = {}
    for name, pred in {
        "Naive Bayes": gnb.predict(X_test),
        "KNN": knn.predict(X_test_s),
    }.items():
        metrics[name] = {
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "cm": confusion_matrix(y_test, pred),
        }

    return {
        "df": df, "cat_cols": cat_cols, "ui_info": ui_info,
        "columns": X.columns.tolist(), "le": le, "scaler": scaler,
        "gnb": gnb, "knn": knn, "metrics": metrics, "smote": SMOTE is not None,
    }


def encode_input(raw_row: dict, art: dict) -> pd.DataFrame:
    """Form ki raw values ko training columns ke format me convert karo."""
    row = pd.DataFrame([raw_row])
    row = pd.get_dummies(row, columns=art["cat_cols"], dtype=int)
    return row.reindex(columns=art["columns"], fill_value=0)


# ---------------------------------------------------------------- UI
st.title("🏦 Loan Default Prediction")
st.caption("Naïve Bayes & KNN | Applicant ki details bharo aur default risk dekho")

if not os.path.exists(CSV_PATH):
    st.error(f"`{CSV_PATH}` nahi mili. Apna dataset is naam se repo me app.py ke saath rakho.")
    st.stop()

art = build_models()
ui_info = art["ui_info"]

tab_pred, tab_eval, tab_data = st.tabs(["🔮 Predict", "📊 Model Evaluation", "🗂️ Data"])

with tab_pred:
    model_name = st.radio("Model chuno", ["Naive Bayes", "KNN"], horizontal=True)
    st.subheader("Applicant details")

    raw = {}
    cols = st.columns(3)
    for i, (name, info) in enumerate(ui_info.items()):
        with cols[i % 3]:
            label = name.replace("_", " ").title()
            if info["type"] == "cat":
                raw[name] = st.selectbox(label, info["options"], key=name)
            elif info["is_int"]:
                raw[name] = st.number_input(
                    label, min_value=int(info["min"]), max_value=int(info["max"]),
                    value=int(info["median"]), step=1, key=name)
            else:
                raw[name] = st.number_input(
                    label, min_value=info["min"], max_value=info["max"],
                    value=info["median"], key=name)

    if st.button("Predict karo", type="primary"):
        x = encode_input(raw, art)
        if model_name == "Naive Bayes":
            model, x_in = art["gnb"], x
        else:
            model, x_in = art["knn"], art["scaler"].transform(x)

        pred = int(model.predict(x_in)[0])
        proba = model.predict_proba(x_in)[0]
        # Notebook ke hisaab se class 1 = default
        p_default = float(proba[list(model.classes_).index(1)]) if 1 in model.classes_ else 0.0

        if pred == 1:
            st.error(f"⚠️ Loan Default: **Yes**  (probability {p_default:.1%})")
        else:
            st.success(f"✅ Loan Default: **No**  (default probability {p_default:.1%})")
        st.progress(min(max(p_default, 0.0), 1.0))

with tab_eval:
    st.subheader("Test-set performance (33% hold-out)")
    if not art["smote"]:
        st.warning("imbalanced-learn install nahi hai, isliye SMOTE skip hua.")
    c1, c2 = st.columns(2)
    for col, (name, m) in zip((c1, c2), art["metrics"].items()):
        with col:
            st.markdown(f"**{name}**")
            st.table(pd.DataFrame(
                {k: [f"{m[k]:.3f}"] for k in ("Accuracy", "Precision", "Recall", "F1")}))
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(m["cm"], annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
    st.info("Note: dono models ka performance abhi weak hai (class 1 ka recall/precision kam). "
            "Ye demo ke liye hai, real lending decision ke liye nahi.")

with tab_data:
    df = art["df"]
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head(50), use_container_width=True)
    st.bar_chart(df[TARGET].value_counts())
