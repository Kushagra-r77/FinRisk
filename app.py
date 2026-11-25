
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "finrisk_random_forest.joblib"

st.set_page_config(
    page_title="FinRisk | Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# 2. DESIGN SYSTEM
# ==========================================

st.markdown("""
<style>
.stApp {
    background: #1c1d1c;
    color: #eeeeea;
}

[data-testid="stMainBlockContainer"] {
    max-width: 100%;
    padding: 1.5rem 1.5rem 2rem 1.5rem;
}

[data-testid="stSidebar"] {
    background: #171817;
    border-right: 1px solid #363836;
}

[data-testid="stSidebar"] > div {
    background: #171817;
}

h1, h2, h3 {
    color: #f1f1ed !important;
    letter-spacing: -0.5px;
}

p {
    color: #c3c5bf;
}

hr {
    border-color: #383a38;
}

.metric-card {
    background: #292a29;
    border: 1px solid #414341;
    border-radius: 13px;
    padding: 17px;
    min-height: 112px;
}

.metric-card.mint {
    background: #dff1dc;
    border-color: #dff1dc;
    color: #243d2c;
}

.metric-card.cyan {
    background: #d9eff0;
    border-color: #d9eff0;
    color: #203d40;
}

.metric-card.coral {
    background: #f2dddd;
    border-color: #f2dddd;
    color: #572d35;
}

.metric-label {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.8px;
    opacity: 0.8;
}

.metric-value {
    font-size: 29px;
    font-weight: 650;
    margin-top: 10px;
}

.metric-note {
    font-size: 11px;
    margin-top: 5px;
    opacity: 0.72;
}

.panel {
    background: #262726;
    border: 1px solid #414341;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 12px;
}

.panel-title {
    font-size: 16px;
    font-weight: 650;
    color: #eeeeea;
    margin-bottom: 12px;
}

.hero-title {
    font-size: 31px;
    font-weight: 700;
    color: #f4f4ed;
}

.hero-subtitle {
    color: #aeb4ac;
    font-size: 13px;
    margin-top: 4px;
}

.mint-text {
    color: #a9e5b6;
}

.coral-text {
    color: #f19ca5;
}

.activity-card {
    background: #292a29;
    border: 1px solid #404240;
    border-radius: 12px;
    padding: 15px;
    margin: 10px 0;
}

.activity-label {
    font-size: 10px;
    color: #bdbdb8;
    background: #454645;
    padding: 4px 7px;
    border-radius: 4px;
}

.activity-title {
    font-weight: 650;
    color: #f0f0ed;
    margin-top: 12px;
    font-size: 14px;
}

.activity-description {
    font-size: 12px;
    color: #aaaFAA;
    line-height: 1.5;
    margin-top: 6px;
}

.status-dot {
    color: #8ee4ac;
    font-size: 12px;
    font-weight: 700;
}

div[data-testid="stFileUploader"] {
    background: #292a29;
    border: 1px dashed #686b66;
    border-radius: 12px;
    padding: 12px;
}

.stButton button, .stDownloadButton button {
    border-radius: 8px;
}

[data-testid="stDataFrame"] {
    border: 1px solid #454745;
    border-radius: 10px;
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}
footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

/* Hide Streamlit's top header */
header[data-testid="stHeader"] {
    display: none !important;
    height: 0 !important;
}

/* Hide the top decoration */
[data-testid="stDecoration"] {
    display: none !important;
}

/* Hide remaining toolbar */
[data-testid="stToolbar"] {
    display: none !important;
}

/* Adjust dashboard spacing */
[data-testid="stMainBlockContainer"] {
    padding-top: 1.5rem !important;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. LOAD SAVED MODEL
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    bundle = load_model()

    model = bundle["model"]
    scaler = bundle["scaler"]
    threshold = bundle["threshold"]
    feature_columns = bundle["feature_columns"]

except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()

# ==========================================
# 3A. SESSION DATA RETENTION
# ==========================================

if "finrisk_results" not in st.session_state:
    st.session_state["finrisk_results"] = None

if "finrisk_uploaded_name" not in st.session_state:
    st.session_state["finrisk_uploaded_name"] = None

if "finrisk_data" not in st.session_state:
    st.session_state["finrisk_data"] = None

# ==========================================
# 4. SIDEBAR NAVIGATION
# ==========================================

st.sidebar.markdown(
    '<div style="font-size:27px;font-weight:750;'
    'color:#eeeeea;margin-top:10px">🛡 FINRISK</div>',
    unsafe_allow_html=True
)

st.sidebar.caption("FRAUD INTELLIGENCE PLATFORM")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    ["Dashboard", "Transactions", "Model Insights", "About"],
    label_visibility="visible"
)

st.sidebar.divider()

st.sidebar.markdown("### SYSTEM STATUS")

st.sidebar.markdown(
    '<span class="status-dot">● MODEL ONLINE</span>',
    unsafe_allow_html=True
)

st.sidebar.metric("Detection Threshold", f"{threshold:.2f}")

st.sidebar.caption("Random Forest Classifier")
st.sidebar.divider()

st.sidebar.caption("FINRISK · ACADEMIC ML PROJECT")


# ==========================================
# 5. MAIN CONTENT AND ACTIVITY PANEL
# ==========================================

main_col, activity_col = st.columns(
    [3.6, 1.25],
    gap="large"
)


# ==========================================
# 6. DASHBOARD PAGE
# ==========================================


# ==========================================
# 6. DASHBOARD PAGE
# ==========================================

if page == "Dashboard":

    with main_col:

        st.markdown(
            '<div class="hero-title">Dashboard Overview</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="hero-subtitle">'
            'Monitor transactions and identify suspicious activity.'
            '</div>',
            unsafe_allow_html=True
        )

        st.write("")

        uploaded_file = st.file_uploader(
            "UPLOAD TRANSACTION DATA",
            type=["csv"],
            help="CSV must contain all 30 numerical model features."
        )

        # ----------------------------------
        # CHECK FOR NEW UPLOAD
        # ----------------------------------

        analysis_results = None
        source_name = None

        if uploaded_file is not None:

            try:
                data = pd.read_csv(uploaded_file)

                if data.empty:
                    st.error("The uploaded CSV is empty.")

                else:

                    missing_columns = [
                        col for col in feature_columns
                        if col not in data.columns
                    ]

                    if missing_columns:
                        st.error(
                            "Missing required features: "
                            + ", ".join(missing_columns)
                        )

                    else:

                        X = data[feature_columns].copy()

                        for col in feature_columns:
                            X[col] = pd.to_numeric(
                                X[col],
                                errors="coerce"
                            )

                        X = X.replace(
                            [np.inf, -np.inf],
                            np.nan
                        )

                        if X.isnull().any().any():

                            st.error(
                                "The uploaded dataset contains "
                                "missing or invalid numerical values."
                            )

                        else:

                            # Apply saved preprocessing
                            X[["Time", "Amount"]] = scaler.transform(
                                X[["Time", "Amount"]]
                            )

                            # Generate predictions
                            probabilities = model.predict_proba(X)[:, 1]

                            predictions = (
                                probabilities >= threshold
                            ).astype(int)

                            total_count = len(data)

                            fraud_count = int(
                                (predictions == 1).sum()
                            )

                            legitimate_count = (
                                total_count - fraud_count
                            )

                            # Create results DataFrame
                            results = pd.DataFrame({
                                "Transaction ID": range(
                                    1, total_count + 1
                                ),
                                "Amount": data["Amount"].values,
                                "Fraud Probability": probabilities,
                                "Prediction": [
                                    "FRAUD ALERT" if p == 1
                                    else "LEGITIMATE"
                                    for p in predictions
                                ]
                            })

                            # SAVE RESULTS IN SESSION
                            st.session_state["finrisk_results"] = (
                                results.copy()
                            )

                            st.session_state["finrisk_data"] = (
                                data.copy()
                            )

                            st.session_state["finrisk_uploaded_name"] = (
                                uploaded_file.name
                            )

                            analysis_results = results.copy()
                            source_name = uploaded_file.name

            except Exception as error:
                st.error(
                    f"Unable to process the uploaded file: {error}"
                )

        # ----------------------------------
        # RESTORE PREVIOUS ANALYSIS
        # ----------------------------------

        elif st.session_state.get("finrisk_results") is not None:

            analysis_results = (
                st.session_state["finrisk_results"].copy()
            )

            source_name = st.session_state.get(
                "finrisk_uploaded_name"
            )

        # ----------------------------------
        # DISPLAY UPLOADED OR SAVED ANALYSIS
        # ----------------------------------

        if analysis_results is not None:

            total_count = len(analysis_results)

            fraud_count = int(
                (
                    analysis_results["Prediction"]
                    == "FRAUD ALERT"
                ).sum()
            )

            legitimate_count = total_count - fraud_count

            st.success(
                f"Analysis available: {source_name}"
            )

            st.markdown(
                '<div class="panel-title">TRANSACTION SUMMARY</div>',
                unsafe_allow_html=True
            )

            cards = st.columns(3)

            card_data = [
                ("TOTAL TRANSACTIONS", f"{total_count:,}", "mint"),
                ("FRAUD ALERTS", f"{fraud_count:,}", "coral"),
                ("LEGITIMATE", f"{legitimate_count:,}", "cyan")
            ]

            for column, item in zip(cards, card_data):

                with column:

                    label, value, style = item

                    st.markdown(f"""
                    <div class="metric-card {style}">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{value}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.write("")

            # ----------------------------------
            # PREDICTION CHART
            # ----------------------------------

            left, right = st.columns([1.5, 1])

            with left:

                st.markdown(
                    '<div class="panel-title">'
                    'PREDICTION DISTRIBUTION</div>',
                    unsafe_allow_html=True
                )

                fig, ax = plt.subplots(figsize=(7, 3))

                fig.patch.set_facecolor("#262726")
                ax.set_facecolor("#262726")

                labels = ["Legitimate", "Fraud Alerts"]

                values = [
                    legitimate_count,
                    fraud_count
                ]

                bars = ax.barh(
                    labels,
                    values,
                    color=["#b9e7c0", "#e99ca5"]
                )

                ax.invert_yaxis()
                ax.tick_params(colors="white")

                ax.set_xlabel(
                    "Transactions",
                    color="white"
                )

                for bar, value in zip(bars, values):

                    ax.text(
                        bar.get_width(),
                        bar.get_y() + bar.get_height() / 2,
                        f" {value:,}",
                        va="center",
                        color="white"
                    )

                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_color("#555555")
                ax.spines["left"].set_color("#555555")

                plt.tight_layout()

                st.pyplot(
                    fig,
                    use_container_width=True
                )

                plt.close(fig)

            with right:

                st.markdown(f"""
                <div class="panel">
                    <div class="panel-title">RISK SUMMARY</div>
                    <div class="metric-label">
                        FRAUD RATE IN UPLOAD
                    </div>
                    <div class="metric-value coral-text">
                        {fraud_count / total_count * 100:.2f}%
                    </div>
                    <hr>
                    <p>Detection threshold: {threshold:.2f}</p>
                    <p>Transactions requiring review: {fraud_count}</p>
                </div>
                """, unsafe_allow_html=True)

            # ----------------------------------
            # TRANSACTION TABLE
            # ----------------------------------

            st.markdown(
                '<div class="panel-title">'
                'TRANSACTION RESULTS</div>',
                unsafe_allow_html=True
            )

            display_results = analysis_results.sort_values(
                "Fraud Probability",
                ascending=False
            ).copy()

            display_results["Fraud Probability"] = (
                display_results["Fraud Probability"]
                .map(lambda x: f"{x:.2%}")
            )

            st.dataframe(
                display_results,
                use_container_width=True,
                height=380,
                hide_index=True
            )

            # Download report
            st.download_button(
                "Download Prediction Report",
                data=analysis_results.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="finrisk_predictions.csv",
                mime="text/csv",
                key="dashboard_prediction_download"
            )

            st.caption(
                "Fraud alerts require further investigation "
                "and are not definitive proof of fraud."
            )

        # ----------------------------------
        # DEFAULT DASHBOARD (NO UPLOAD)
        # ----------------------------------

        elif uploaded_file is None:

            st.markdown(
                '<div class="panel-title">'
                'MODEL PERFORMANCE SNAPSHOT</div>',
                unsafe_allow_html=True
            )

            cards = st.columns(4)

            metric_data = [
                ("TEST PRECISION", "88.30%", "mint"),
                ("TEST RECALL", "84.69%", "cyan"),
                ("F1 SCORE", "0.8646", "mint"),
                ("ROC-AUC", "0.9529", "coral")
            ]

            for column, item in zip(cards, metric_data):

                with column:

                    label, value, style = item

                    st.markdown(f"""
                    <div class="metric-card {style}">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{value}</div>
                        <div class="metric-note">
                            Held-out test results
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.write("")

            left, right = st.columns([1.5, 1])

            with left:

                st.markdown(
                    '<div class="panel-title">'
                    'FINAL MODEL CONFUSION MATRIX</div>',
                    unsafe_allow_html=True
                )

                fig, ax = plt.subplots(figsize=(7, 4))

                fig.patch.set_facecolor("#262726")
                ax.set_facecolor("#262726")

                cm = np.array([
                    [56853, 11],
                    [15, 83]
                ])

                ax.imshow(cm, cmap="GnBu")

                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])

                ax.set_xticklabels(
                    ["Legitimate", "Fraudulent"],
                    color="white"
                )

                ax.set_yticklabels(
                    ["Legitimate", "Fraudulent"],
                    color="white"
                )

                ax.set_xlabel(
                    "Predicted label",
                    color="white"
                )

                ax.set_ylabel(
                    "Actual label",
                    color="white"
                )

                for i in range(2):
                    for j in range(2):

                        ax.text(
                            j, i, f"{cm[i, j]:,}",
                            ha="center",
                            va="center",
                            color="white",
                            fontsize=11
                        )

                ax.tick_params(colors="white")

                plt.tight_layout()

                st.pyplot(
                    fig,
                    use_container_width=True
                )

                plt.close(fig)

            with right:

                st.markdown("""
                <div class="panel">
                    <div class="panel-title">MODEL PROFILE</div>
                    <p class="mint-text">● Random Forest</p>
                    <p>30 input features</p>
                    <p>100 decision trees</p>
                    <p>Classification threshold: 0.25</p>
                    <hr>
                    <div class="panel-title">DATASET</div>
                    <p>284,807 transactions</p>
                    <p>492 fraudulent transactions</p>
                    <p>0.1727% fraud rate</p>
                </div>
                """, unsafe_allow_html=True)

            st.info(
                "Upload a transaction CSV to start fraud screening."
            )



# ==========================================
# 7. TRANSACTIONS PAGE
# ==========================================

elif page == "Transactions":

    with main_col:

        st.markdown(
            '<div class="hero-title">Transaction Review</div>',
            unsafe_allow_html=True
        )

        st.caption(
            "Explore, search and review previously analyzed transactions."
        )

        # Retrieve previously uploaded analysis
        saved_results = st.session_state.get("finrisk_results")
        uploaded_name = st.session_state.get("finrisk_uploaded_name")

        if saved_results is None:

            st.info(
                "No transaction analysis available yet. "
                "Upload a CSV from the Dashboard first."
            )

        else:

            results = saved_results.copy()

            st.success(
                f"Analysis retained: {uploaded_name}"
            )

            # Summary metrics
            total = len(results)

            fraud_count = int(
                (results["Prediction"] == "FRAUD ALERT").sum()
            )

            legitimate_count = total - fraud_count

            c1, c2, c3 = st.columns(3)

            c1.metric("Total Transactions", f"{total:,}")
            c2.metric("Fraud Alerts", f"{fraud_count:,}")
            c3.metric("Legitimate", f"{legitimate_count:,}")

            st.divider()

            # Search and filter controls
            st.markdown("### Review Queue")

            filter_col, search_col = st.columns([1, 2])

            with filter_col:
                status_filter = st.selectbox(
                    "Filter by prediction",
                    [
                        "All Transactions",
                        "Fraud Alerts",
                        "Legitimate"
                    ]
                )

            with search_col:
                search_query = st.text_input(
                    "Search transaction ID",
                    placeholder="Enter transaction ID..."
                )

            filtered = results.copy()

            if status_filter == "Fraud Alerts":
                filtered = filtered[
                    filtered["Prediction"] == "FRAUD ALERT"
                ]

            elif status_filter == "Legitimate":
                filtered = filtered[
                    filtered["Prediction"] == "LEGITIMATE"
                ]

            if search_query.strip():
                filtered = filtered[
                    filtered["Transaction ID"]
                    .astype(str)
                    .str.contains(search_query.strip(), na=False)
                ]

            # Display highest-risk transactions first
            filtered = filtered.sort_values(
                "Fraud Probability",
                ascending=False
            )

            display_data = filtered.copy()

            display_data["Fraud Probability"] = (
                display_data["Fraud Probability"]
                .map(lambda x: f"{x:.2%}")
            )

            st.caption(
                f"Showing {len(filtered):,} of {total:,} transactions"
            )

            st.dataframe(
                display_data,
                use_container_width=True,
                height=450,
                hide_index=True
            )

            # Download complete report
            st.download_button(
                "Download Complete Prediction Report",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="finrisk_transaction_report.csv",
                mime="text/csv"
            )

            st.divider()

            # Clear retained analysis
            if st.button(
                "Clear Uploaded Analysis",
                type="secondary"
            ):

                st.session_state["finrisk_results"] = None
                st.session_state["finrisk_data"] = None
                st.session_state["finrisk_uploaded_name"] = None

                st.rerun()


# ==========================================
# 8. MODEL INSIGHTS PAGE
# ==========================================

elif page == "Model Insights":

    with main_col:

        st.markdown(
            '<div class="hero-title">Model Intelligence</div>',
            unsafe_allow_html=True
        )

        st.caption("Evaluation on the held-out test dataset.")

        comparison = pd.DataFrame({
            "Model": [
                "Logistic Regression",
                "Random Forest (0.50)",
                "Random Forest (0.25)"
            ],
            "Precision": [0.0610, 0.9600, 0.8830],
            "Recall": [0.9184, 0.7347, 0.8469],
            "F1-Score": [0.1144, 0.8324, 0.8646],
            "ROC-AUC": [0.9722, 0.9529, 0.9529]
        })

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### Selected model: Random Forest")

        st.write("""
        Random Forest was selected as the leading model based on
        its balance of precision and recall after threshold tuning.

        The threshold was selected using a separate validation set
        rather than the final test set.
        """)

        st.markdown("### Final test confusion matrix")

        st.dataframe(
            pd.DataFrame(
                [[56853, 11], [15, 83]],
                index=["Actual Legitimate", "Actual Fraudulent"],
                columns=["Predicted Legitimate", "Predicted Fraudulent"]
            ),
            use_container_width=True
        )


# ==========================================
# 9. ABOUT PAGE
# ==========================================

elif page == "About":

    with main_col:

        st.markdown(
            '<div class="hero-title">About FinRisk</div>',
            unsafe_allow_html=True
        )

        st.markdown("""
        <div class="panel">
            <div class="panel-title">PROJECT OVERVIEW</div>
            FinRisk is an academic machine learning project
            designed to identify potentially fraudulent
            credit card transactions using historical data.
        </div>

        <div class="panel">
            <div class="panel-title">TECHNOLOGIES</div>
            Python · Pandas · NumPy · Scikit-learn ·
            Matplotlib · Joblib · Streamlit
        </div>

        <div class="panel">
            <div class="panel-title">MACHINE LEARNING</div>
            Logistic Regression and Random Forest classifiers
            are trained and compared. The Random Forest
            decision threshold is optimized using validation data.
        </div>

        <div class="panel">
            <div class="panel-title">DATASET</div>
            284,807 transactions and 30 input features.
            V1 to V28 are anonymized numerical features.
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# 10. RIGHT ACTIVITY PANEL
# ==========================================

with activity_col:

    st.markdown("### ACTIVITIES")
    st.caption("MODEL AND RISK MONITOR")

    st.markdown("""
    <div class="activity-card">
        <span class="activity-label">MODEL STATUS</span>
        <div class="activity-title">Random Forest active</div>
        <div class="activity-description">
            Saved fraud detection model loaded successfully.
            Classification threshold: 0.25.
        </div>
    </div>

    <div class="activity-card">
        <span class="activity-label">MODEL INSIGHT</span>
        <div class="activity-title">High fraud recall</div>
        <div class="activity-description">
            The selected model detected 84.69% of fraudulent
            transactions in the final test evaluation.
        </div>
    </div>

    <div class="activity-card">
        <span class="activity-label">DATASET INSIGHT</span>
        <div class="activity-title">Class imbalance detected</div>
        <div class="activity-description">
            Only 492 out of 284,807 original transactions
            were fraudulent.
        </div>
    </div>

    <div class="activity-card">
        <span class="activity-label">PERFORMANCE</span>
        <div class="activity-title">F1-score: 0.8646</div>
        <div class="activity-description">
            Final threshold-tuned Random Forest performance
            on the held-out test data.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.caption("FINRISK · FRAUD INTELLIGENCE")
    st.caption("Academic prototype · Not a live banking system")


# ==========================================
# 11. FOOTER
# ==========================================

st.divider()

st.caption(
    "FINRISK  |  CREDIT CARD FRAUD DETECTION  |  "
    "MACHINE LEARNING PROJECT"
)
