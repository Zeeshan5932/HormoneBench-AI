"""
prediction.py

Rule Based Prediction Model

Later Replace With:
-------------------
XGBoost
Random Forest
LightGBM
"""

from utils.constants import (
    LOW_RISK,
    MEDIUM_RISK,
    HIGH_RISK
)


class HormonePredictionModel:

    def __init__(self):
        pass

    def predict(self, dataframe):

        score = 0
        reasons = []

        # ==========================
        # Average Values
        # ==========================

        avg_sleep = dataframe["sleep"].mean()

        avg_stress = dataframe["stress"].mean()

        avg_temp = dataframe["temperature"].mean()

        avg_hr = dataframe["heart_rate"].mean()

        avg_bmi = dataframe["bmi"].mean()

        # ==========================
        # Sleep
        # ==========================

        if avg_sleep < 5:

            score += 25

            reasons.append("Low Sleep Duration")

        # ==========================
        # Stress
        # ==========================

        if avg_stress > 8:

            score += 25

            reasons.append("High Stress Level")

        # ==========================
        # Temperature
        # ==========================

        if avg_temp > 37.5:

            score += 20

            reasons.append("High Body Temperature")

        # ==========================
        # Heart Rate
        # ==========================

        if avg_hr > 100:

            score += 15

            reasons.append("High Heart Rate")

        # ==========================
        # BMI
        # ==========================

        if avg_bmi > 30:

            score += 15

            reasons.append("High BMI")

        # ==========================
        # Final Prediction
        # ==========================

        if score < 30:

            risk = LOW_RISK

            confidence = 96

        elif score < 60:

            risk = MEDIUM_RISK

            confidence = 90

        else:

            risk = HIGH_RISK

            confidence = 86

        return {

            "risk": risk,

            "confidence": confidence,

            "score": score,

            "reasons": reasons

        }