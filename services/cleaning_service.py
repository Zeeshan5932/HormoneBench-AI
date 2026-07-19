"""
cleaning_service.py

Responsibilities
----------------
1. Load uploaded CSV
2. Validate dataset
3. Clean column names
4. Remove duplicate rows
5. Handle missing values
6. Convert dataset into unified schema
7. Save cleaned dataset
"""

import pandas as pd

from services.storage_service import StorageService

from utils.helpers import (
    validate_csv,
    clean_column_names,
    dataframe_info
)

from utils.schema_mapper import (
    map_schema,
    ensure_required_columns,
    reorder_columns
)

from utils.constants import (
    REQUIRED_COLUMNS,
    DEFAULT_PROCESSED_FILE
)


class CleaningService:

    def __init__(self):

        self.storage = StorageService()

    def clean_dataset(self, uploaded_file):

        # -------------------------
        # Save Uploaded File
        # -------------------------

        file_path = self.storage.save_uploaded_file(uploaded_file)

        # -------------------------
        # Load CSV
        # -------------------------

        df = self.storage.load_csv(file_path)

        # -------------------------
        # Validate Dataset
        # -------------------------

        validate_csv(df)

        # -------------------------
        # Standardize Column Names
        # -------------------------

        df = clean_column_names(df)

        # -------------------------
        # Remove Duplicate Rows
        # -------------------------

        before_duplicates = len(df)

        df = df.drop_duplicates()

        duplicates_removed = before_duplicates - len(df)

        # -------------------------
        # Schema Mapping
        # -------------------------

        df = map_schema(df)

        # -------------------------
        # Add Missing Columns
        # -------------------------

        df = ensure_required_columns(
            df,
            REQUIRED_COLUMNS
        )

        # -------------------------
        # Keep Required Columns Only
        # -------------------------

        df = reorder_columns(
            df,
            REQUIRED_COLUMNS
        )

        # -------------------------
        # Handle Missing Values
        # -------------------------

        numeric_columns = df.select_dtypes(
            include=["number"]
        ).columns

        for column in numeric_columns:

            df[column] = df[column].fillna(
                df[column].median()
            )

        object_columns = df.select_dtypes(
            include=["object"]
        ).columns

        for column in object_columns:

            df[column] = df[column].fillna(
                "Unknown"
            )

        # -------------------------
        # Remove Negative Values
        # -------------------------

        for column in numeric_columns:

            df[column] = df[column].clip(lower=0)

        # -------------------------
        # Dataset Summary
        # -------------------------

        summary = dataframe_info(df)

        summary["Duplicates Removed"] = duplicates_removed

        # -------------------------
        # Save Processed Dataset
        # -------------------------

        output_path = self.storage.save_processed_data(
            df,
            DEFAULT_PROCESSED_FILE
        )

        return {

            "dataframe": df,

            "summary": summary,

            "processed_file": output_path

        }

    def preview_dataset(self, uploaded_file, rows=10):

        """
        Returns first N rows for Streamlit Preview.
        """

        file_path = self.storage.save_uploaded_file(uploaded_file)

        df = self.storage.load_csv(file_path)

        df = clean_column_names(df)

        return df.head(rows)