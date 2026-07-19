import streamlit as st
import pandas as pd

from services.storage_service import StorageService
from services.cleaning_service import CleaningService

st.title("📂 Upload Dataset")

st.write(
    "Upload a hormone dataset in CSV format."
)

uploaded_file = st.file_uploader(
    "Choose CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        storage = StorageService()

        cleaning = CleaningService()

        # Save Uploaded File
        file_path = storage.save_uploaded_file(uploaded_file)

        st.success("✅ File Uploaded Successfully")

        # Preview Uploaded Dataset
        df = pd.read_csv(file_path)

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(),
            use_container_width=True
        )

        st.markdown("---")

        if st.button("🚀 Process Dataset"):

            with st.spinner("Cleaning Dataset..."):

                processed_df = cleaning.process(file_path)

            st.success("Dataset Processed Successfully")

            st.subheader("Processed Dataset")

            st.dataframe(
                processed_df.head(),
                use_container_width=True
            )

            st.info(
                f"Total Records : {len(processed_df)}"
            )

            st.info(
                f"Columns : {len(processed_df.columns)}"
            )

    except Exception as e:

        st.error(str(e))