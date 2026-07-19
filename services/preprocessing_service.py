"""
preprocessing_service.py

Data Processing & Feature Engineering (Module 2)
Handles missing values, creates longitudinal sequences, and engineers 
physiological features to generate benchmark-ready datasets.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class PreprocessingService:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def preprocess(self, dataframe):
        """
        Standardized multimodal preprocessing pipeline.
        """
        df = dataframe.copy()
        df = self.convert_types(df)
        df = self.fill_missing(df)
        df = self.engineer_longitudinal_features(df)
        df = self.normalize(df)
        return df

    def convert_types(self, df):
        numeric_columns = ["age", "sleep", "heart_rate", "temperature", "stress", "cycle_day", "bmi"]
        for column in numeric_columns:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")
        return df

    def fill_missing(self, df):
        numeric_columns = df.select_dtypes(include="number").columns
        for column in numeric_columns:
            df[column] = df[column].fillna(df[column].median())
        return df
        
    def engineer_longitudinal_features(self, df):
        """
        Creates temporal features for Time-Series Foundation Models.
        """
        if "temperature" in df.columns:
            df["temp_rolling_avg_3d"] = df["temperature"].rolling(window=3, min_periods=1).mean()
        return df

    def normalize(self, df):
        numeric_columns = df.select_dtypes(include="number").columns
        df[numeric_columns] = self.scaler.fit_transform(df[numeric_columns])
        return df

    def statistics(self, df):
        return {
            "Rows": df.shape[0],
            "Columns": df.shape[1],
            "Missing Values": int(df.isnull().sum().sum()),
            "Numeric Columns": len(df.select_dtypes(include="number").columns)
        }