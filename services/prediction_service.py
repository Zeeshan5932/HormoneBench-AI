"""
prediction_service.py

Prediction Service

This service connects the frontend with the prediction model.

Flow

CSV
 ↓
PredictionService
 ↓
HormonePredictionModel
 ↓
Prediction Result
"""

import pandas as pd

from models.prediction import HormonePredictionModel


class PredictionService:

    def __init__(self):

        self.model = HormonePredictionModel()

    def predict(self, dataframe: pd.DataFrame):

        """
        Predict hormonal risk.

        Returns
        -------
        dict
        """

        return self.model.predict(dataframe)

    def predict_dataset(self, dataframe: pd.DataFrame):

        """
        Optional

        Predict every row separately.

        Useful in future for benchmark evaluation.
        """

        predictions = []

        for _, row in dataframe.iterrows():

            temp_df = pd.DataFrame([row])

            result = self.model.predict(temp_df)

            predictions.append({

                "Risk": result["risk"],

                "Confidence": result["confidence"],

                "Score": result["score"]

            })

        return pd.DataFrame(predictions)

    def summary(self, dataframe: pd.DataFrame):

        """
        Dashboard Summary
        """

        result = self.model.predict(dataframe)

        return {

            "Risk": result["risk"],

            "Confidence": result["confidence"],

            "Score": result["score"],

            "Reasons": result["reasons"]

        }