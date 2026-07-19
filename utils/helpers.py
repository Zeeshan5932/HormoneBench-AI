"""
helpers.py

Common helper functions used across the project.
"""

import pandas as pd
import random


def validate_csv(df: pd.DataFrame):
    """
    Check dataframe is empty or not.
    """

    if df.empty:
        raise ValueError("Uploaded CSV is empty.")

    return True


def missing_values(df: pd.DataFrame):
    """
    Return missing values count.
    """

    return df.isnull().sum()


def dataframe_info(df: pd.DataFrame):
    """
    Returns dataset summary.
    """

    return {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Missing Values": int(df.isnull().sum().sum())
    }


def clean_column_names(df: pd.DataFrame):
    """
    Standardize column names.
    """

    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(" ", "_")
    )

    return df


def calculate_risk_score(row):
    """
    Simple Rule Based Risk Score.
    """

    score = 0

    if row["sleep"] < 5:
        score += 30

    if row["stress"] > 8:
        score += 30

    if row["temperature"] > 37.5:
        score += 20

    if row["heart_rate"] > 100:
        score += 20

    return score


def random_confidence():
    """
    Temporary confidence score.

    Later replace using ML probability.
    """

    return random.randint(80, 98)