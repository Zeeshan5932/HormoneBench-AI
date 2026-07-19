import streamlit as st
import pandas as pd
from pathlib import Path

from services.prediction_service import PredictionService

st.title("🧠 Hormonal Risk Prediction")

processed_file = Path("data/processed/processed_data.csv")

if not processed_file.exists():

    st.warning("Please upload and process a dataset first.")

    st.stop()

df = pd.read_csv(processed_file)

st.subheader("Processed Dataset")

st.dataframe(
    df.head(),
    use_container_width=True
)

st.markdown("---")

prediction = PredictionService()

if st.button("🚀 Run Prediction"):

    with st.spinner("Running Prediction..."):

        results = prediction.predict(df)

    st.success("Prediction Completed")

    st.markdown("---")

    risk = results["risk"]

    confidence = results["confidence"]

    score = results["score"]

    reasons = results["reasons"]

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Risk Level",
            risk
        )

    with c2:

        st.metric(
            "Confidence",
            f"{confidence}%"
        )

    with c3:

        st.metric(
            "Risk Score",
            score
        )

    st.markdown("---")

    st.subheader("Important Factors")

    for reason in reasons:

        st.write("✅", reason)

    st.markdown("---")

    st.subheader("Recommendation")

    if risk == "Low Risk":

        st.success(
            """
Healthy hormonal profile detected.

Maintain:

• Proper Sleep

• Balanced Diet

• Regular Exercise
            """
        )

    elif risk == "Medium Risk":

        st.warning(
            """
Moderate hormonal imbalance detected.

Recommendations:

• Improve Sleep

• Reduce Stress

• Drink More Water
            """
        )

    else:

        st.error(
            """
High hormonal risk detected.

Recommendations:

• Consult Healthcare Professional

• Improve Lifestyle

• Monitor Hormonal Health
            """
        )

    st.markdown("---")

    st.subheader("Prediction Summary")

    summary = pd.DataFrame({

        "Metric":[
            "Risk",
            "Confidence",
            "Score"
        ],

        "Value":[
            risk,
            confidence,
            score
        ]

    })

    st.table(summary)