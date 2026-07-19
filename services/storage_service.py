"""
storage_service.py

Responsible for all file operations.

Responsibilities:
---------------
1. Save uploaded CSV files
2. Read CSV files
3. Save processed datasets
4. Save generated reports
5. Create folders automatically
"""

from pathlib import Path
import pandas as pd
import shutil


class StorageService:

    def __init__(self):

        self.base_dir = Path(__file__).resolve().parent.parent

        self.raw_data = self.base_dir / "data" / "raw"
        self.processed_data = self.base_dir / "data" / "processed"
        self.reports = self.base_dir / "data" / "reports"

        self._create_directories()

    def _create_directories(self):
        """
        Automatically create required folders
        """

        self.raw_data.mkdir(parents=True, exist_ok=True)
        self.processed_data.mkdir(parents=True, exist_ok=True)
        self.reports.mkdir(parents=True, exist_ok=True)

    def save_uploaded_file(self, uploaded_file):

        """
        Save uploaded CSV into raw folder
        """

        file_path = self.raw_data / uploaded_file.name

        with open(file_path, "wb") as f:
            shutil.copyfileobj(uploaded_file, f)

        return file_path

    def load_csv(self, file_path):

        """
        Read CSV
        """

        return pd.read_csv(file_path)

    def save_processed_data(self, dataframe, filename="processed_data.csv"):

        """
        Save cleaned dataframe
        """

        output_path = self.processed_data / filename

        dataframe.to_csv(output_path, index=False)

        return output_path

    def save_report(self, report_bytes, filename="prediction_report.pdf"):

        """
        Save generated PDF
        """

        report_path = self.reports / filename

        with open(report_path, "wb") as f:
            f.write(report_bytes)

        return report_path