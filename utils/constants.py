"""
constants.py

Contains all project constants.
Changing values here updates the whole project.
"""

# ==============================
# Supported File Types
# ==============================

ALLOWED_FILE_TYPES = [
    "csv"
]

# ==============================
# Required Unified Schema
# ==============================

REQUIRED_COLUMNS = [
    "age",
    "sleep",
    "heart_rate",
    "temperature",
    "stress",
    "cycle_day",
    "bmi"
]

# ==============================
# Prediction Labels
# ==============================

LOW_RISK = "Low Risk"

MEDIUM_RISK = "Medium Risk"

HIGH_RISK = "High Risk"

# ==============================
# Risk Thresholds
# ==============================

LOW_THRESHOLD = 30

MEDIUM_THRESHOLD = 60

HIGH_THRESHOLD = 100

# ==============================
# Report Name
# ==============================

REPORT_NAME = "prediction_report.pdf"

# ==============================
# Default Dataset Name
# ==============================

DEFAULT_PROCESSED_FILE = "processed_data.csv"

# ==============================
# Streamlit App Title
# ==============================

APP_TITLE = "HormoneBench AI"

APP_SUBTITLE = "Open Benchmark Infrastructure for Women's Hormonal Health"