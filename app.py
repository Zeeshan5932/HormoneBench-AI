import streamlit as st
from pathlib import Path
from PIL import Image

logo = Image.open("assets/logo.png")

st.image(logo,width=120)
# ===========================
# Page Configuration
# ===========================

st.set_page_config(
    page_title="HormoneBench AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================
# Custom CSS
# ===========================

css_file = Path("assets/style.css")

if css_file.exists():
    with open(css_file) as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

# ===========================
# Sidebar
# ===========================

with st.sidebar:

    st.title("🩺 HormoneBench AI")

    st.markdown("---")

    st.success("Women's Hormonal Health Benchmark")

    st.markdown("### Navigation")

    st.info("""
📊 Dashboard

📂 Upload Dataset

🧠 Prediction

📄 Report

ℹ About
""")

    st.markdown("---")

    st.caption("Version 1.0 MVP")

# ===========================
# Main Page
# ===========================

st.title("🩺 HormoneBench AI")

st.subheader(
    "Open Benchmark Infrastructure for Women's Hormonal Health"
)

st.markdown("---")

st.markdown("""
### 🎯 Project Objective

HormoneBench AI is a research-oriented platform designed to create a unified benchmark
for women's hormonal health datasets.

The system provides:

- CSV Dataset Upload
- Data Cleaning
- Unified Schema Mapping
- Hormonal Risk Prediction
- PDF Report Generation
""")

st.markdown("---")

# ===========================
# Workflow
# ===========================

st.subheader("🔄 Workflow")

st.code(
"""
CSV Upload
      │
      ▼
Data Cleaning
      │
      ▼
Schema Mapping
      │
      ▼
Prediction
      │
      ▼
PDF Report
"""
)

# ===========================
# Features
# ===========================

st.subheader("🚀 Features")

col1, col2 = st.columns(2)

with col1:

    st.success("✔ Upload CSV Dataset")

    st.success("✔ Clean Data")

    st.success("✔ Unified Schema")

with col2:

    st.success("✔ Risk Prediction")

    st.success("✔ PDF Report")

    st.success("✔ Research Ready")

st.markdown("---")

# ===========================
# Technology Stack
# ===========================

st.subheader("💻 Technology Stack")

st.write("""
- Python
- Streamlit
- Pandas
- NumPy
- ReportLab
- Scikit-learn
""")
def load_css():

    with open("assets/style.css") as f:

        st.markdown(

            f"<style>{f.read()}</style>",

            unsafe_allow_html=True

        )

load_css()

st.markdown("---")

# ===========================
# Footer
# ===========================

st.info(
    "Developed as an MVP for HormoneBench AI Research Project"
)