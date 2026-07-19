import streamlit as st
from pathlib import Path

from services.report_service import ReportService

st.title("📄 Prediction Report")

processed_file = Path("data/processed/processed_data.csv")

if not processed_file.exists():

    st.warning("Please upload and process a dataset first.")

    st.stop()

st.success("Processed dataset found.")

st.markdown("---")

st.write("""
Generate a professional PDF report containing:

- Dataset Summary
- Hormonal Risk
- Confidence Score
- Recommendations
- Disclaimer
""")

if st.button("📥 Generate Report"):

    report = ReportService()

    pdf_path = report.generate()

    st.success("PDF Report Generated Successfully")

    with open(pdf_path, "rb") as file:

        st.download_button(

            label="⬇ Download Report",

            data=file,

            file_name="HormoneBench_Report.pdf",

            mime="application/pdf"
        )

st.markdown("---")

st.info("HormoneBench AI Research Report")