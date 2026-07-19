"""
report_service.py

Generate PDF Report using ReportLab
"""

from io import BytesIO
from datetime import datetime

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

from reportlab.lib import colors

from services.storage_service import StorageService


class ReportService:

    def __init__(self):

        self.storage = StorageService()

        self.styles = getSampleStyleSheet()

    def generate_report(
        self,
        summary,
        prediction_df
    ):

        """
        Generate PDF report.
        """

        buffer = BytesIO()

        document = SimpleDocTemplate(buffer)

        elements = []

        # ------------------------
        # Title
        # ------------------------

        title = Paragraph(
            "HormoneBench AI Report",
            self.styles["Title"]
        )

        elements.append(title)

        elements.append(Spacer(1, 20))

        # ------------------------
        # Date
        # ------------------------

        date = Paragraph(

            f"Generated : {datetime.now()}",

            self.styles["Normal"]

        )

        elements.append(date)

        elements.append(Spacer(1, 20))

        # ------------------------
        # Summary
        # ------------------------

        elements.append(

            Paragraph(

                "<b>Dataset Summary</b>",

                self.styles["Heading2"]

            )

        )

        summary_table = [

            ["Metric", "Value"]

        ]

        for key, value in summary.items():

            summary_table.append(

                [key, str(value)]

            )

        table = Table(summary_table)

        table.setStyle(

            TableStyle([

                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),

                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),

                ("GRID", (0, 0), (-1, -1), 1, colors.black),

                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),

                ("BOTTOMPADDING", (0, 0), (-1, 0), 10)

            ])

        )

        elements.append(table)

        elements.append(Spacer(1, 20))

        # ------------------------
        # Prediction Table
        # ------------------------

        elements.append(

            Paragraph(

                "<b>Prediction Results</b>",

                self.styles["Heading2"]

            )

        )

        prediction_table = [

            list(prediction_df.columns)

        ]

        for _, row in prediction_df.iterrows():

            prediction_table.append(

                list(row.astype(str))

            )

        prediction_table = Table(prediction_table)

        prediction_table.setStyle(

            TableStyle([

                ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),

                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),

                ("GRID", (0, 0), (-1, -1), 1, colors.black),

                ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke)

            ])

        )

        elements.append(prediction_table)

        elements.append(Spacer(1, 20))

        # ------------------------
        # Disclaimer
        # ------------------------

        disclaimer = Paragraph(

            "<b>Disclaimer:</b><br/>"

            "This report is generated for research "

            "and educational purposes only. "

            "It is not a medical diagnosis.",

            self.styles["BodyText"]

        )

        elements.append(disclaimer)

        # ------------------------
        # Build PDF
        # ------------------------

        document.build(elements)

        pdf = buffer.getvalue()

        buffer.close()

        path = self.storage.save_report(pdf)

        return path