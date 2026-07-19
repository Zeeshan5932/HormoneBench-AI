"""
schema_mapper.py

Convert different datasets into one Unified Schema.
"""

import pandas as pd


COLUMN_MAPPING = {

    # Age
    "age": "age",
    "patient_age": "age",

    # Sleep
    "sleep": "sleep",
    "sleep_hours": "sleep",
    "sleephour": "sleep",

    # Heart Rate
    "heart_rate": "heart_rate",
    "heartrate": "heart_rate",
    "pulse": "heart_rate",

    # Temperature
    "temperature": "temperature",
    "temp": "temperature",
    "body_temp": "temperature",

    # Stress
    "stress": "stress",
    "stress_level": "stress",

    # Cycle Day
    "cycle_day": "cycle_day",
    "cycleday": "cycle_day",

    # BMI
    "bmi": "bmi",
    "body_mass_index": "bmi"
}


def map_schema(df: pd.DataFrame):
    """
    Rename dataframe columns to unified schema.
    """

    rename_dict = {}

    for column in df.columns:

        key = column.lower().strip()

        if key in COLUMN_MAPPING:
            rename_dict[column] = COLUMN_MAPPING[key]

    df = df.rename(columns=rename_dict)

    return df


def ensure_required_columns(df, required_columns):
    """
    Create missing columns with None values.
    """

    for column in required_columns:

        if column not in df.columns:
            df[column] = None

    return df


def reorder_columns(df, required_columns):
    """
    Arrange dataframe columns.
    """

    return df[required_columns]