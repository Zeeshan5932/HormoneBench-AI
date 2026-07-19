import streamlit as st
import pandas as pd
from pathlib import Path

from utils.constants import APP_TITLE, APP_SUBTITLE

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🩺",
    layout="wide"
)

st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

st.markdown("---")

# ===============================
# Project Overview
# ===============================

st.header("📌 Project Overview")

st.write("""
HormoneBench AI is an AI-powered benchmark platform for women's hormonal health.

This MVP demonstrates:

- Dataset Upload
- Data Cleaning
- Unified Schema Mapping
- Risk Prediction
- Report Generation
""")

st.markdown("---")

# ===============================
# Load Processed Dataset
# ===============================

processed_file = Path("data/processed/processed_data.csv")

if processed_file.exists():

    df = pd.read_csv(processed_file)

    total_records = len(df)
    total_columns = len(df.columns)
    missing = int(df.isnull().sum().sum())

else:

    total_records = 0
    total_columns = 0
    missing = 0

# ===============================
# Metrics
# ===============================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Uploaded Records",
        total_records
    )

with col2:
    st.metric(
        "Features",
        total_columns
    )

with col3:
    st.metric(
        "Missing Values",
        missing
    )

st.markdown("---")

# ===============================
# Dataset Preview
# ===============================

st.subheader("Dataset Preview")

if processed_file.exists():

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

else:

    st.info("No processed dataset available.")

st.markdown("---")

# ===============================
# Workflow
# ===============================

st.subheader("System Workflow")

st.markdown("""
```text
CSV Upload
      │
      ▼
Cleaning
      │
      ▼
Schema Mapping
      │
      ▼
Prediction
      │
      ▼
PDF Report

""")

#================================Footer
st.markdown("HormoneBench AI MVP Ready")