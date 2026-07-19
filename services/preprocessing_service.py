"""
preprocessing_service.py

Responsible for preprocessing dataset before prediction
or ML model training.

Current Version:
----------------
1. Fill Missing Values
2. Convert Data Types
3. Normalize Numeric Columns

Future Version:
---------------
1. One Hot Encoding
2. Feature Engineering
3. Feature Selection
4. SMOTE
5. ML Pipeline
"""

import pandas as pd

from sklearn.preprocessing import MinMaxScaler


class PreprocessingService:

    def __init__(self):

        self.scaler = MinMaxScaler()

    def preprocess(self, dataframe):

        """
        Complete preprocessing pipeline.
        """

        df = dataframe.copy()

        df = self.convert_types(df)

        df = self.fill_missing(df)

        df = self.normalize(df)

        return df

    def convert_types(self, df):

        """
        Convert numeric columns.
        """

        numeric_columns = [

            "age",
            "sleep",
            "heart_rate",
            "temperature",
            "stress",
            "cycle_day",
            "bmi"

        ]

        for column in numeric_columns:

            if column in df.columns:

                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce"
                )

        return df

    def fill_missing(self, df):

        """
        Fill missing numeric values.
        """

        numeric_columns = df.select_dtypes(
            include="number"
        ).columns

        for column in numeric_columns:

            df[column] = df[column].fillna(
                df[column].median()
            )

        return df

    def normalize(self, df):

        """
        Normalize all numeric features.
        """

        numeric_columns = df.select_dtypes(
            include="number"
        ).columns

        df[numeric_columns] = self.scaler.fit_transform(
            df[numeric_columns]
        )

        return df

    def statistics(self, df):

        """
        Dataset statistics.
        """

        return {

            "Rows": df.shape[0],

            "Columns": df.shape[1],

            "Missing Values": int(
                df.isnull().sum().sum()
            ),

            "Numeric Columns": len(
                df.select_dtypes(
                    include="number"
                ).columns
            )

        }