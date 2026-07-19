from __future__ import annotations

from html import escape
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Optional
import textwrap

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# OPTIONAL PROJECT IMPORTS
# ============================================================
# Kisi service me import error aaye to poora app crash nahi hoga.
# App built-in fallback pipeline use karega.

try:
    from services.cleaning_service import CleaningService
except Exception as error:
    CleaningService = None
    CLEANING_IMPORT_ERROR = str(error)
else:
    CLEANING_IMPORT_ERROR = None


try:
    from services.preprocessing_service import PreprocessingService
except Exception as error:
    PreprocessingService = None
    PREPROCESSING_IMPORT_ERROR = str(error)
else:
    PREPROCESSING_IMPORT_ERROR = None


try:
    from services.prediction_service import PredictionService
except Exception as error:
    PredictionService = None
    PREDICTION_IMPORT_ERROR = str(error)
else:
    PREDICTION_IMPORT_ERROR = None


try:
    from services.report_service import ReportService
except Exception as error:
    ReportService = None
    REPORT_IMPORT_ERROR = str(error)
else:
    REPORT_IMPORT_ERROR = None


try:
    from services.rag_service import RAGService
except Exception as error:
    RAGService = None
    RAG_IMPORT_ERROR = str(error)
else:
    RAG_IMPORT_ERROR = None


try:
    from utils.constants import APP_TITLE, REQUIRED_COLUMNS
except Exception:
    APP_TITLE = "HormoneBench AI"

    REQUIRED_COLUMNS = [
        "age",
        "sleep",
        "heart_rate",
        "temperature",
        "stress",
        "cycle_day",
        "bmi",
    ]


try:
    from utils.schema_mapper import (
        ensure_required_columns,
        map_schema,
    )
except Exception:
    ensure_required_columns = None
    map_schema = None


# ============================================================
# PAGE CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS LOADER
# ============================================================

def local_css(file_path: Path) -> None:
    """
    Load CSS file safely.
    """

    if file_path.exists():

        css_content = file_path.read_text(
            encoding="utf-8"
        )

        st.markdown(
            f"<style>{css_content}</style>",
            unsafe_allow_html=True,
        )


local_css(
    BASE_DIR / "assets" / "style.css"
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {

    "analysis_ready": False,

    "raw_df": None,

    "cleaned_df": None,

    "processed_df": None,

    "prediction_df": None,

    "pipeline_metadata": {},

    "report_bytes": None,

    "report_name": "hormonebench_report.html",

    "report_mime": "text/html",

    "rag_history": [],
}


for key, default_value in DEFAULT_STATE.items():

    if key not in st.session_state:

        st.session_state[key] = default_value


# ============================================================
# CSV READER
# ============================================================

@st.cache_data(show_spinner=False)
def read_csv_bytes(
    file_bytes: bytes,
) -> pd.DataFrame:
    """
    Read uploaded CSV bytes.
    """

    if not file_bytes:

        raise ValueError(
            "Uploaded CSV file is empty."
        )

    try:

        dataframe = pd.read_csv(
            BytesIO(file_bytes)
        )

    except UnicodeDecodeError:

        dataframe = pd.read_csv(
            BytesIO(file_bytes),
            encoding="latin-1",
        )

    if dataframe.empty:

        raise ValueError(
            "Uploaded CSV contains no records."
        )

    return dataframe


# ============================================================
# RAG SERVICE CACHE
# ============================================================

@st.cache_resource(
    show_spinner="Loading AI Research Copilot..."
)
def get_rag_service() -> Any:
    """
    Initialize RAGService only once.

    It will only run when user clicks
    Retrieve Evidence.
    """

    if RAGService is None:

        raise RuntimeError(

            "RAGService could not be imported. "

            f"Error: {RAG_IMPORT_ERROR}"
        )

    return RAGService()


# ============================================================
# GENERAL HELPERS
# ============================================================

def extract_dataframe(
    result: Any,
) -> Optional[pd.DataFrame]:
    """
    Extract dataframe from different service outputs.

    Supported outputs:

    DataFrame

    Dictionary

    Tuple

    List
    """

    if isinstance(
        result,
        pd.DataFrame,
    ):

        return result.copy()


    if isinstance(
        result,
        pd.Series,
    ):

        return result.to_frame()


    if isinstance(
        result,
        dict,
    ):

        preferred_keys = [

            "data",

            "dataframe",

            "cleaned_data",

            "cleaned_df",

            "processed_data",

            "processed_df",

            "predictions",

            "prediction_df",

            "results",
        ]

        for key in preferred_keys:

            if key in result:

                extracted = extract_dataframe(
                    result[key]
                )

                if extracted is not None:

                    return extracted


        for value in result.values():

            extracted = extract_dataframe(
                value
            )

            if extracted is not None:

                return extracted


    if isinstance(
        result,
        (tuple, list),
    ):

        for value in result:

            extracted = extract_dataframe(
                value
            )

            if extracted is not None:

                return extracted


    return None


def call_dataframe_service(
    service_class: Any,
    method_names: Iterable[str],
    dataframe: pd.DataFrame,
) -> tuple[Optional[pd.DataFrame], str]:
    """
    Call a service using common method names.

    Example supported method names:

    clean_data

    clean_dataframe

    preprocess_data

    predict
    """

    if service_class is None:

        return None, "Service unavailable"


    try:

        service = service_class()

    except Exception as error:

        return None, (
            "Service initialization failed: "
            f"{type(error).__name__}: {error}"
        )


    last_error = None


    for method_name in method_names:

        method = getattr(
            service,
            method_name,
            None,
        )

        if not callable(method):

            continue


        try:

            result = method(
                dataframe.copy()
            )

            extracted = extract_dataframe(
                result
            )

            if (
                extracted is not None
                and not extracted.empty
            ):

                return (
                    extracted.reset_index(
                        drop=True
                    ),
                    (
                        f"{service_class.__name__}."
                        f"{method_name}"
                    ),
                )

        except Exception as error:

            last_error = error


    if last_error is not None:

        return None, (

            "Service method failed: "

            f"{type(last_error).__name__}: "

            f"{last_error}"
        )


    return None, (
        "No compatible method found"
    )


def reset_analysis() -> None:
    """
    Reset dataset output.

    RAG chat history is preserved.
    """

    keys = [

        "analysis_ready",

        "raw_df",

        "cleaned_df",

        "processed_df",

        "prediction_df",

        "pipeline_metadata",

        "report_bytes",

        "report_name",

        "report_mime",
    ]

    for key in keys:

        st.session_state[key] = (
            DEFAULT_STATE[key]
        )


def find_column(
    dataframe: pd.DataFrame,
    candidates: Iterable[str],
) -> Optional[str]:
    """
    Return first matching column.
    """

    for column in candidates:

        if column in dataframe.columns:

            return column

    return None


# ============================================================
# SCHEMA MAPPING
# ============================================================

COLUMN_ALIASES = {

    "patient_age": "age",

    "years": "age",

    "sleep_hours": "sleep",

    "sleep_hour": "sleep",

    "sleephour": "sleep",

    "heartrate": "heart_rate",

    "heart_rate_bpm": "heart_rate",

    "pulse": "heart_rate",

    "temp": "temperature",

    "body_temp": "temperature",

    "body_temperature": "temperature",

    "stress_level": "stress",

    "stress_score": "stress",

    "cycleday": "cycle_day",

    "day_of_cycle": "cycle_day",

    "body_mass_index": "bmi",
}


def normalize_column_name(
    column: Any,
) -> str:
    """
    Convert column name into snake_case.
    """

    column_name = str(
        column
    ).strip().lower()

    column_name = column_name.replace(
        "-",
        " ",
    )

    column_name = "_".join(
        column_name.split()
    )

    return COLUMN_ALIASES.get(
        column_name,
        column_name,
    )


def unify_schema(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert uploaded dataframe into unified schema.
    """

    unified = dataframe.copy()


    unified.columns = [

        normalize_column_name(column)

        for column in unified.columns
    ]


    if callable(map_schema):

        unified = map_schema(
            unified
        )


    # Temp and Temperature may map to same column.
    unified = unified.loc[
        :,
        ~unified.columns.duplicated(
            keep="first"
        ),
    ]


    if callable(ensure_required_columns):

        unified = ensure_required_columns(

            unified,

            REQUIRED_COLUMNS,
        )

    else:

        for column in REQUIRED_COLUMNS:

            if column not in unified.columns:

                unified[column] = np.nan


    for column in REQUIRED_COLUMNS:

        unified[column] = pd.to_numeric(

            unified[column],

            errors="coerce",
        )


    unified = unified.dropna(

        subset=REQUIRED_COLUMNS,

        how="all",
    )


    if unified.empty:

        raise ValueError(

            "Dataset contains no supported "

            "HormoneBench features."
        )


    for column in REQUIRED_COLUMNS:

        median = unified[column].median()

        fill_value = (

            float(median)

            if pd.notna(median)

            else 0.0
        )

        unified[column] = unified[
            column
        ].fillna(fill_value)


    ordered_columns = (

        REQUIRED_COLUMNS

        + [

            column

            for column in unified.columns

            if column not in REQUIRED_COLUMNS
        ]
    )


    return unified[
        ordered_columns
    ].reset_index(drop=True)


# ============================================================
# CLEANING STAGE
# ============================================================

def run_cleaning_stage(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    str,
    dict[str, int],
]:
    """
    Run CleaningService.

    If service fails, use built-in cleaning.
    """

    original_rows = len(
        dataframe
    )

    missing_before = int(
        dataframe.isna().sum().sum()
    )


    service_dataframe, source = (
        call_dataframe_service(

            CleaningService,

            [

                "clean_dataframe",

                "clean_data",

                "clean_dataset",

                "clean",

                "process",
            ],

            dataframe,
        )
    )


    if service_dataframe is None:

        cleaned = dataframe.copy()

        cleaned = cleaned.dropna(

            axis=0,

            how="all",
        )

        cleaned = cleaned.dropna(

            axis=1,

            how="all",
        )

        cleaned = cleaned.drop_duplicates()

        source = (

            "Built-in cleaning fallback "

            f"({source})"
        )

    else:

        cleaned = service_dataframe


    cleaned = unify_schema(
        cleaned
    )


    metadata = {

        "original_rows": original_rows,

        "cleaned_rows": len(cleaned),

        "duplicates_removed": max(

            original_rows - len(cleaned),

            0,
        ),

        "missing_before": missing_before,

        "missing_after": int(

            cleaned.isna().sum().sum()
        ),
    }


    return cleaned, source, metadata


# ============================================================
# PREPROCESSING STAGE
# ============================================================

def run_preprocessing_stage(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """
    Run PreprocessingService.

    If unavailable, create basic engineered features.
    """

    service_dataframe, source = (
        call_dataframe_service(

            PreprocessingService,

            [

                "preprocess_dataframe",

                "preprocess_data",

                "preprocess_dataset",

                "preprocess",

                "transform",

                "process",
            ],

            dataframe,
        )
    )


    if service_dataframe is not None:

        return service_dataframe, source


    processed = dataframe.copy()


    for column in REQUIRED_COLUMNS:

        processed[column] = pd.to_numeric(

            processed[column],

            errors="coerce",
        ).fillna(0.0)


    processed["sleep_deficit"] = (

        7.0 - processed["sleep"]

    ).clip(lower=0.0)


    processed["high_stress_flag"] = (

        processed["stress"] >= 7.0

    ).astype(int)


    processed[
        "elevated_heart_rate_flag"
    ] = (

        processed["heart_rate"] > 100.0

    ).astype(int)


    processed[
        "elevated_temperature_flag"
    ] = (

        processed["temperature"] > 37.5

    ).astype(int)


    return (

        processed,

        f"Built-in preprocessing ({source})",
    )


# ============================================================
# PREDICTION STAGE
# ============================================================

def fallback_prediction(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Transparent rule-based MVP prediction.

    This is not a trained medical model.
    """

    result = dataframe.copy().reset_index(
        drop=True
    )


    result["sleep_impact"] = np.select(

        [

            result["sleep"] < 5,

            result["sleep"] < 6,
        ],

        [

            30.0,

            15.0,
        ],

        default=0.0,
    )


    result["stress_impact"] = np.select(

        [

            result["stress"] > 8,

            result["stress"] >= 7,
        ],

        [

            30.0,

            15.0,
        ],

        default=0.0,
    )


    result[
        "temperature_impact"
    ] = np.where(

        result["temperature"] > 37.5,

        20.0,

        0.0,
    )


    result[
        "heart_rate_impact"
    ] = np.where(

        result["heart_rate"] > 100,

        20.0,

        0.0,
    )


    result["bmi_impact"] = np.select(

        [

            result["bmi"] >= 30,

            result["bmi"] < 18.5,
        ],

        [

            15.0,

            10.0,
        ],

        default=0.0,
    )


    impact_columns = [

        "sleep_impact",

        "stress_impact",

        "temperature_impact",

        "heart_rate_impact",

        "bmi_impact",
    ]


    result["risk_score"] = (

        result[impact_columns]

        .sum(axis=1)

        .clip(0, 100)
    )


    result["risk_level"] = pd.cut(

        result["risk_score"],

        bins=[

            -1,

            29,

            59,

            100,
        ],

        labels=[

            "Low Risk",

            "Medium Risk",

            "High Risk",
        ],
    ).astype(str)


    result["confidence"] = (

        70

        + (

            result["risk_score"] - 50

        ).abs() * 0.4

    ).clip(70, 90).round(1)


    factor_mapping = {

        "sleep_impact": "low sleep",

        "stress_impact": "high stress",

        "temperature_impact": (
            "elevated temperature"
        ),

        "heart_rate_impact": (
            "elevated heart rate"
        ),

        "bmi_impact": (
            "BMI outside reference range"
        ),
    }


    def important_factors(
        row: pd.Series,
    ) -> str:

        factors = [

            factor_name

            for column, factor_name
            in factor_mapping.items()

            if float(row[column]) > 0
        ]

        return (

            ", ".join(factors)

            if factors

            else "No active demo rules"
        )


    result["important_factors"] = (

        result.apply(

            important_factors,

            axis=1,
        )
    )


    return result


def prediction_is_valid(
    dataframe: Optional[pd.DataFrame],
) -> bool:
    """
    Check service prediction output.
    """

    if dataframe is None:

        return False


    valid_columns = {

        "risk_score",

        "risk_level",

        "prediction",

        "predicted_label",

        "risk",

        "label",
    }


    return bool(

        valid_columns.intersection(

            dataframe.columns
        )
    )


def run_prediction_stage(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """
    Run PredictionService or fallback.
    """

    predicted, source = call_dataframe_service(

        PredictionService,

        [

            "predict_dataframe",

            "predict_dataset",

            "run_prediction",

            "generate_predictions",

            "predict",
        ],

        dataframe,
    )


    if prediction_is_valid(
        predicted
    ):

        predicted = predicted.reset_index(
            drop=True
        )


        if len(predicted) == len(dataframe):

            duplicate_columns = [

                column

                for column in dataframe.columns

                if column in predicted.columns
            ]


            predicted = pd.concat(

                [

                    dataframe.reset_index(
                        drop=True
                    ),

                    predicted.drop(

                        columns=duplicate_columns,

                        errors="ignore",
                    ),
                ],

                axis=1,
            )


        return predicted, source


    return (

        fallback_prediction(dataframe),

        f"Built-in rule engine ({source})",
    )


# ============================================================
# REPORT STAGE
# ============================================================

def create_html_report(
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    metadata: dict[str, Any],
) -> bytes:
    """
    Create HTML report without extra dependencies.
    """

    if "risk_level" in prediction_df.columns:

        distribution = (

            prediction_df["risk_level"]

            .astype(str)

            .value_counts()

            .rename_axis("Risk Level")

            .reset_index(name="Records")
        )

    else:

        distribution = pd.DataFrame(

            {

                "Risk Level": ["Unavailable"],

                "Records": [

                    len(prediction_df)
                ],
            }
        )


    average_score = 0.0


    if "risk_score" in prediction_df.columns:

        average_score = float(

            prediction_df[

                "risk_score"

            ].mean()
        )


    report_html = f"""
<!doctype html>

<html lang="en">

<head>

<meta charset="utf-8">

<title>HormoneBench AI Report</title>

<style>

body {{
    font-family: Arial, sans-serif;
    margin: 40px;
    color: #222222;
}}

h1, h2 {{
    color: #6b2d5c;
}}

.card {{
    border: 1px solid #dddddd;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 16px;
}}

table {{
    border-collapse: collapse;
    width: 100%;
}}

th, td {{
    border: 1px solid #dddddd;
    padding: 8px;
    text-align: left;
}}

th {{
    background: #f4f4f4;
}}

.notice {{
    background: #fff4e5;
    padding: 14px;
    border-radius: 8px;
}}

</style>

</head>

<body>

<h1>HormoneBench AI Research Report</h1>

<div class="card">

<h2>Pipeline Summary</h2>

<p>
<strong>Uploaded records:</strong>
{len(raw_df)}
</p>

<p>
<strong>Processed records:</strong>
{len(cleaned_df)}
</p>

<p>
<strong>Average score:</strong>
{average_score:.1f}/100
</p>

<p>
<strong>Cleaning:</strong>
{escape(str(metadata.get("cleaning_source", "Unknown")))}
</p>

<p>
<strong>Preprocessing:</strong>
{escape(str(metadata.get("preprocessing_source", "Unknown")))}
</p>

<p>
<strong>Prediction:</strong>
{escape(str(metadata.get("prediction_source", "Unknown")))}
</p>

</div>

<h2>Prediction Distribution</h2>

{distribution.to_html(index=False)}

<h2>Prediction Preview</h2>

{prediction_df.head(100).to_html(index=False)}

<p class="notice">

<strong>Disclaimer:</strong>

HormoneBench AI is a research MVP.

This report is not a medical diagnosis,

clinical probability, or treatment recommendation.

</p>

</body>

</html>
"""


    return report_html.encode(
        "utf-8"
    )


def run_report_stage(
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    metadata: dict[str, Any],
) -> tuple[bytes, str, str, str]:
    """
    Try ReportService first.

    Built-in HTML report is fallback.
    """

    if ReportService is not None:

        try:

            report_service = ReportService()


            method_names = [

                "generate_pdf",

                "generate_report",

                "create_report",

                "build_report",
            ]


            for method_name in method_names:

                method = getattr(

                    report_service,

                    method_name,

                    None,
                )


                if not callable(method):

                    continue


                candidate_calls = [

                    lambda: method(
                        prediction_df
                    ),

                    lambda: method(

                        processed_df,

                        prediction_df,
                    ),

                    lambda: method(

                        dataframe=processed_df,

                        predictions=prediction_df,

                        metadata=metadata,
                    ),
                ]


                for candidate_call in candidate_calls:

                    try:

                        result = candidate_call()


                        if isinstance(result, bytes):

                            if result.startswith(
                                b"%PDF"
                            ):

                                return (

                                    result,

                                    "hormonebench_report.pdf",

                                    "application/pdf",

                                    (
                                        "ReportService."
                                        f"{method_name}"
                                    ),
                                )


                            return (

                                result,

                                "hormonebench_report.bin",

                                (
                                    "application/"
                                    "octet-stream"
                                ),

                                (
                                    "ReportService."
                                    f"{method_name}"
                                ),
                            )


                        if isinstance(
                            result,
                            (str, Path),
                        ):

                            report_path = Path(
                                result
                            )


                            if report_path.exists():

                                mime = (

                                    "application/pdf"

                                    if (
                                        report_path.suffix
                                        .lower()
                                        == ".pdf"
                                    )

                                    else (
                                        "application/"
                                        "octet-stream"
                                    )
                                )


                                return (

                                    report_path
                                    .read_bytes(),

                                    report_path.name,

                                    mime,

                                    (
                                        "ReportService."
                                        f"{method_name}"
                                    ),
                                )


                    except TypeError:

                        continue

                    except Exception:

                        break


        except Exception:

            pass


    html_report = create_html_report(

        raw_df,

        cleaned_df,

        prediction_df,

        metadata,
    )


    return (

        html_report,

        "hormonebench_report.html",

        "text/html",

        "Built-in HTML report",
    )


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def run_complete_pipeline(
    uploaded_file: Any,
) -> dict[str, Any]:
    """
    Complete pipeline:

    CSV

    Cleaning

    Unified Schema

    Preprocessing

    Prediction

    Report
    """

    raw_df = read_csv_bytes(

        uploaded_file.getvalue()
    )


    (
        cleaned_df,

        cleaning_source,

        cleaning_metadata,

    ) = run_cleaning_stage(
        raw_df
    )


    (
        processed_df,

        preprocessing_source,

    ) = run_preprocessing_stage(
        cleaned_df
    )


    (
        prediction_df,

        prediction_source,

    ) = run_prediction_stage(
        processed_df
    )


    metadata = {

        **cleaning_metadata,

        "uploaded_file": uploaded_file.name,

        "cleaning_source": cleaning_source,

        "preprocessing_source": (
            preprocessing_source
        ),

        "prediction_source": prediction_source,
    }


    (
        report_bytes,

        report_name,

        report_mime,

        report_source,

    ) = run_report_stage(

        raw_df,

        cleaned_df,

        processed_df,

        prediction_df,

        metadata,
    )


    metadata["report_source"] = (
        report_source
    )


    return {

        "raw_df": raw_df,

        "cleaned_df": cleaned_df,

        "processed_df": processed_df,

        "prediction_df": prediction_df,

        "metadata": metadata,

        "report_bytes": report_bytes,

        "report_name": report_name,

        "report_mime": report_mime,
    }


# ============================================================
# DATASET TAB
# ============================================================

def render_dataset_tab() -> None:

    raw_df = st.session_state.raw_df

    cleaned_df = st.session_state.cleaned_df

    processed_df = (
        st.session_state.processed_df
    )

    metadata = (
        st.session_state.pipeline_metadata
    )


    column1, column2, column3, column4 = (

        st.columns(4)
    )


    column1.metric(

        "Uploaded Records",

        len(raw_df),
    )


    column2.metric(

        "Unified Records",

        len(cleaned_df),
    )


    column3.metric(

        "Unified Features",

        len(REQUIRED_COLUMNS),
    )


    column4.metric(

        "Missing Values",

        metadata.get(
            "missing_after",
            0,
        ),
    )


    st.subheader(
        "Unified Benchmark Dataset"
    )


    st.dataframe(

        cleaned_df,

        width="True",

        hide_index=True,
    )


    st.download_button(

        "Download Unified CSV",

        data=cleaned_df.to_csv(

            index=False

        ).encode("utf-8"),

        file_name=(

            "unified_hormonebench_data.csv"
        ),

        mime="text/csv",

        width="True",

        key="download_unified_csv",
    )


    with st.expander(
        "Processed Data Preview"
    ):

        st.dataframe(

            processed_df.head(100),

            width="True",

            hide_index=True,
        )


    with st.expander(
        "Original Uploaded Dataset"
    ):

        st.dataframe(

            raw_df.head(100),

            width="Trues",

            hide_index=True,
        )


    with st.expander(
        "Pipeline Details"
    ):

        st.write(

            "**Cleaning:** "

            f"{metadata.get('cleaning_source')}"
        )

        st.write(

            "**Preprocessing:** "

            f"{metadata.get('preprocessing_source')}"
        )

        st.write(

            "**Prediction:** "

            f"{metadata.get('prediction_source')}"
        )

        st.write(

            "**Report:** "

            f"{metadata.get('report_source')}"
        )


# ============================================================
# PREDICTION TAB
# ============================================================

def render_prediction_tab() -> None:

    prediction_df = (
        st.session_state.prediction_df
    )

    metadata = (
        st.session_state.pipeline_metadata
    )


    risk_column = find_column(

        prediction_df,

        [

            "risk_level",

            "prediction",

            "predicted_label",

            "label",

            "risk",
        ],
    )


    score_column = find_column(

        prediction_df,

        [

            "risk_score",

            "score",

            "probability",

            "confidence",
        ],
    )


    average_score = 0.0


    if score_column:

        average_score = float(

            pd.to_numeric(

                prediction_df[score_column],

                errors="coerce",

            ).mean()
        )


    high_risk_count = 0


    if risk_column:

        high_risk_count = int(

            prediction_df[risk_column]

            .astype(str)

            .str.contains(

                "high",

                case=False,

                na=False,

            ).sum()
        )


    column1, column2, column3 = (

        st.columns(3)
    )


    column1.metric(

        "Analyzed Records",

        len(prediction_df),
    )


    column2.metric(

        "Average Score",

        f"{average_score:.1f}",
    )


    column3.metric(

        "High Risk Flags",

        high_risk_count,
    )


    prediction_source = str(

        metadata.get(

            "prediction_source",

            "",
        )
    )


    if "Built-in rule engine" in prediction_source:

        st.warning(

            "Current output uses a transparent "

            "rule-based MVP engine. It is not "

            "a trained or clinically validated "

            "foundation model."
        )

    else:

        st.success(

            f"Prediction source: "

            f"{prediction_source}"
        )


    if risk_column:

        distribution = (

            prediction_df[risk_column]

            .astype(str)

            .value_counts()

            .rename_axis("Risk Category")

            .to_frame("Records")
        )


        st.subheader(
            "Prediction Distribution"
        )


        st.bar_chart(
            distribution
        )


    display_columns = [

        column

        for column in [

            "age",

            "sleep",

            "heart_rate",

            "temperature",

            "stress",

            "cycle_day",

            "bmi",

            score_column,

            risk_column,

            "confidence",

            "important_factors",
        ]

        if (

            column

            and column
            in prediction_df.columns
        )
    ]


    st.subheader(
        "Record-Level Results"
    )


    st.dataframe(

        (

            prediction_df[display_columns]

            if display_columns

            else prediction_df
        ),

        width="True",

        hide_index=True,
    )


    st.download_button(

        "Download Prediction Results",

        data=prediction_df.to_csv(

            index=False

        ).encode("utf-8"),

        file_name=(

            "hormonebench_predictions.csv"
        ),

        mime="text/csv",

        width="True",

        key="download_predictions",
    )


    st.caption(

        "Research-use only. This output is "

        "not a medical diagnosis or treatment "

        "recommendation."
    )


# ============================================================
# EXPLAINABILITY TAB
# ============================================================

def render_explainability_tab() -> None:

    prediction_df = (
        st.session_state.prediction_df
    )


    impact_columns = [

        column

        for column in prediction_df.columns

        if column.endswith("_impact")
    ]


    if not impact_columns:

        st.info(

            "PredictionService did not return "

            "feature contribution or SHAP values."
        )

        return


    st.warning(

        "These are rule contributions for the "

        "current MVP. They are not genuine SHAP "

        "values from a trained machine-learning "

        "model."
    )


    average_impacts = (

        prediction_df[impact_columns]

        .apply(

            pd.to_numeric,

            errors="coerce",
        )

        .abs()

        .mean()

        .sort_values(
            ascending=False
        )

        .rename(

            index=lambda value: (

                value

                .replace("_impact", "")

                .replace("_", " ")

                .title()
            )
        )

        .to_frame(
            "Average Contribution"
        )
    )


    st.subheader(
        "Global Feature Contribution"
    )


    st.bar_chart(
        average_impacts
    )


    record_number = st.number_input(

        "Inspect Record",

        min_value=1,

        max_value=max(

            len(prediction_df),

            1,
        ),

        value=1,

        step=1,
    )


    selected_row = prediction_df.iloc[

        int(record_number) - 1
    ]


    record_impacts = (

        selected_row[impact_columns]

        .astype(float)

        .sort_values(
            ascending=False
        )

        .rename(

            index=lambda value: (

                value

                .replace("_impact", "")

                .replace("_", " ")

                .title()
            )
        )

        .to_frame(
            "Contribution"
        )
    )


    st.subheader(

        f"Record {record_number} Explanation"
    )


    st.dataframe(

        record_impacts,

        width="True",
    )


    if (

        "important_factors"

        in prediction_df.columns
    ):

        st.info(

            "Important Factors: "

            f"{selected_row['important_factors']}"
        )


# ============================================================
# RAG COPILOT TAB
# ============================================================

def render_rag_tab() -> None:

    st.markdown(

        "### Evidence-Grounded "

        "AI Research Copilot"
    )


    st.caption(

        "Searches configured local documents, "

        "ArXiv papers, and public web sources."
    )


    if RAGService is None:

        st.error(

            "RAGService could not be imported.\n\n"

            f"Error: {RAG_IMPORT_ERROR}"
        )

        return


    with st.form(
        "rag_question_form"
    ):

        user_query = st.text_area(

            "Ask a Research Question",

            value=(

                "How does resting heart rate "

                "change during the luteal phase?"
            ),

            height=100,
        )


        retrieve_button = (

            st.form_submit_button(

                "Retrieve Evidence",

                type="primary",
            )
        )


    if retrieve_button:

        if not user_query.strip():

            st.warning(

                "Please enter a question."
            )

        else:

            try:

                rag_service = (
                    get_rag_service()
                )


                with st.spinner(

                    "Searching local evidence, "

                    "ArXiv, and the web..."
                ):

                    response = (

                        rag_service
                        .query_copilot(

                            user_query.strip()
                        )
                    )


                st.session_state[
                    "rag_history"
                ].append(

                    {

                        "question": (

                            user_query.strip()
                        ),

                        "answer": response,
                    }
                )


            except Exception as error:

                st.error(

                    "Research Copilot Error: "

                    f"{type(error).__name__}: "

                    f"{error}"
                )


    if st.session_state["rag_history"]:

        st.subheader(
            "Copilot Conversation"
        )


        history = st.session_state[

            "rag_history"

        ][-5:]


        for conversation in reversed(
            history
        ):

            with st.chat_message("user"):

                st.markdown(

                    conversation["question"]
                )


            with st.chat_message(
                "assistant"
            ):

                st.markdown(

                    conversation["answer"]
                )


        if st.button(

            "Clear Copilot History",

            key="clear_rag_history",
        ):

            st.session_state[
                "rag_history"
            ] = []

            st.rerun()


# ============================================================
# REPORT TAB
# ============================================================

def render_report_tab() -> None:

    if not st.session_state[
        "report_bytes"
    ]:

        st.info(

            "Run the pipeline to generate report."
        )

        return


    st.subheader(
        "Research Report"
    )


    st.write(

        "The generated report includes pipeline "

        "summary, prediction distribution, "

        "record preview, and disclaimer."
    )


    st.download_button(

        "Download Research Report",

        data=st.session_state[
            "report_bytes"
        ],

        file_name=st.session_state[
            "report_name"
        ],

        mime=st.session_state[
            "report_mime"
        ],

        type="primary",

        width="True",

        key="download_report",
    )


    st.caption(

        "Generated by: "

        f"{st.session_state['pipeline_metadata'].get('report_source')}"
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "🩺 HormoneBench AI"
    )


    st.caption(
        "Open Research Infrastructure"
    )


    menu = st.radio(

        "Navigation",

        [

            "Dashboard",

            "About Us",
        ],

        label_visibility="collapsed",
    )


    st.divider()


    st.success(

        "Core System: Operational"
    )


    if RAGService is None:

        st.warning(

            "Research Copilot: Import Error"
        )

    else:

        st.info(

            "Research Copilot: Loads On Demand"
        )


    if st.session_state[
        "analysis_ready"
    ]:

        st.divider()


        metadata = st.session_state[

            "pipeline_metadata"
        ]


        st.write(

            "**Dataset:** "

            f"{metadata.get('uploaded_file')}"
        )


        st.write(

            "**Processed Records:** "

            f"{len(st.session_state['cleaned_df'])}"
        )


        if st.button(

            "Reset Dataset Analysis",

            width="True",
        ):

            reset_analysis()

            st.rerun()


# ============================================================
# DASHBOARD PAGE
# ============================================================

if menu == "Dashboard":

    st.title(

        "🩺 HormoneBench AI: "

        "Integrated Research Platform"
    )


    st.caption(

        "Open AI Infrastructure for "

        "Women's Hormonal Health"
    )


    with st.container(
        border=True
    ):

        st.subheader(

            "Scientific Validation Pipeline"
        )


        st.write(

            "Upload a CSV dataset. The platform "

            "will clean the data, map it to a "

            "unified schema, preprocess features, "

            "run predictions, and generate a report."
        )


        with st.form(

            "dataset_upload_form",

            clear_on_submit=False,
        ):

            uploaded_file = st.file_uploader(

                "Upload Benchmark Dataset",

                type=["csv"],

                help="Only CSV files are supported.",
            )


            run_pipeline_button = (

                st.form_submit_button(

                    "Run Scientific Validation Pipeline",

                    type="primary",
                )
            )


        if run_pipeline_button:

            if uploaded_file is None:

                st.warning(

                    "Please upload a CSV file first."
                )

            else:

                reset_analysis()


                try:

                    with st.spinner(

                        "Running cleaning, schema "

                        "mapping, preprocessing, "

                        "prediction, and reporting..."
                    ):

                        pipeline_result = (

                            run_complete_pipeline(

                                uploaded_file
                            )
                        )


                    st.session_state[
                        "raw_df"
                    ] = pipeline_result[
                        "raw_df"
                    ]


                    st.session_state[
                        "cleaned_df"
                    ] = pipeline_result[
                        "cleaned_df"
                    ]


                    st.session_state[
                        "processed_df"
                    ] = pipeline_result[
                        "processed_df"
                    ]


                    st.session_state[
                        "prediction_df"
                    ] = pipeline_result[
                        "prediction_df"
                    ]


                    st.session_state[
                        "pipeline_metadata"
                    ] = pipeline_result[
                        "metadata"
                    ]


                    st.session_state[
                        "report_bytes"
                    ] = pipeline_result[
                        "report_bytes"
                    ]


                    st.session_state[
                        "report_name"
                    ] = pipeline_result[
                        "report_name"
                    ]


                    st.session_state[
                        "report_mime"
                    ] = pipeline_result[
                        "report_mime"
                    ]


                    st.session_state[
                        "analysis_ready"
                    ] = True


                    st.success(

                        "Scientific validation "

                        "pipeline completed."
                    )


                except Exception as error:

                    st.session_state[
                        "analysis_ready"
                    ] = False


                    st.error(

                        "Pipeline Failed: "

                        f"{type(error).__name__}: "

                        f"{error}"
                    )


    if st.session_state[
        "analysis_ready"
    ]:

        (

            dataset_tab,

            prediction_tab,

            explainability_tab,

            rag_tab,

            report_tab,

        ) = st.tabs(

            [

                "📊 Benchmark Dataset",

                "🧠 Foundation Model",

                "🔍 Explainability",

                "🤖 AI Research Copilot",

                "📄 Report",
            ]
        )


        with dataset_tab:

            render_dataset_tab()


        with prediction_tab:

            render_prediction_tab()


        with explainability_tab:

            render_explainability_tab()


        with rag_tab:

            render_rag_tab()


        with report_tab:

            render_report_tab()


    else:

        st.info(

            "Upload a dataset and run the pipeline "

            "to unlock benchmark analysis, "

            "prediction, explainability, Copilot, "

            "and report sections."
        )


# ============================================================
# ABOUT PAGE
# ============================================================

# ============================================================
# ABOUT US PAGE
# ============================================================

elif menu == "About Us":

    st.title("Meet Our Team")

    # --------------------------------------------------------
    # ABOUT PAGE CSS
    # --------------------------------------------------------

    st.markdown(
        """
        <style>

        .team-card {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 18px;
            padding: 25px;
            margin-bottom: 22px;
            min-height: 390px;
            text-align: center;
            transition: 0.3s ease;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
        }

        .team-card:hover {
            transform: translateY(-5px);
            border-color: #c44ba0;
            box-shadow: 0 14px 30px rgba(196, 75, 160, 0.18);
        }

        .team-image {
            width: 135px;
            height: 135px;
            border-radius: 50%;
            object-fit: cover;
            border: 4px solid #c44ba0;
            margin-bottom: 16px;
            background: white;
        }

        .team-name {
            font-size: 24px;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .team-role {
            font-size: 16px;
            color: #e775c2;
            font-weight: 600;
            margin-bottom: 15px;
        }

        .team-description {
            font-size: 14px;
            line-height: 1.7;
            min-height: 105px;
            color: #d7d7d7;
            margin-bottom: 22px;
        }

        .linkedin-button {
            display: inline-block;
            background: #0a66c2;
            color: white !important;
            padding: 10px 22px;
            border-radius: 8px;
            text-decoration: none !important;
            font-weight: 600;
            transition: 0.3s ease;
        }

        .linkedin-button:hover {
            background: #084d93;
            transform: scale(1.03);
        }

        .linkedin-unavailable {
            display: inline-block;
            background: #555555;
            color: white;
            padding: 10px 22px;
            border-radius: 8px;
            font-weight: 600;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # TEAM DATA
    # --------------------------------------------------------

    team_members = [

        {
            "name": "Tayyab Nisar",

            "role": "Chief Executive Officer — CEO",

            "image": (
                "https://api.dicebear.com/9.x/"
                "initials/svg?seed=Tayyab%20Nisar"
                "&backgroundColor=0a66c2"
                "&fontFamily=Arial"
            ),

            "linkedin": (
                "https://www.linkedin.com/"
                "in/tayyabnisar"
            ),

            "description": (
                "Tayyab leads the overall vision and strategic "
                "direction of HormoneBench AI. He manages project "
                "planning, team coordination, research priorities, "
                "product decisions, and presentation of the platform."
            ),
        },

        {
            "name": "Zeeshan Younas",

            "role": "Chief Technology Officer — CTO",

            "image": (
                "https://api.dicebear.com/9.x/"
                "initials/svg?seed=Zeeshan%20Younas"
                "&backgroundColor=6b2d5c"
                "&fontFamily=Arial"
            ),

            # Zeeshan ka LinkedIn link milnay par
            # empty string ki jagah link paste kar dein.
            "linkedin": "",

            "description": (
                "Zeeshan manages the technical architecture of "
                "HormoneBench AI. He is responsible for the Streamlit "
                "application, backend services, Gemini RAG integration, "
                "system reliability, and deployment infrastructure."
            ),
        },

        {
            "name": "Beenish Mehar",

            "role": "Research and Data Lead",

            "image": (
                "https://api.dicebear.com/9.x/"
                "initials/svg?seed=Beenish%20Mehar"
                "&backgroundColor=b94b8c"
                "&fontFamily=Arial"
            ),

            "linkedin": (
                "https://www.linkedin.com/"
                "in/beenish-m-32b64916b/"
            ),

            "description": (
                "Beenish works on hormonal-health research, dataset "
                "analysis, evidence collection, schema validation, "
                "benchmark documentation, and evaluation of the "
                "research results produced by the platform."
            ),
        },

        {
            "name": "Mahnoor M. Akram",

            "role": "UI/UX and Documentation Lead",

            "image": (
                "https://api.dicebear.com/9.x/"
                "initials/svg?seed=Mahnoor%20M%20Akram"
                "&backgroundColor=8b3d72"
                "&fontFamily=Arial"
            ),

            "linkedin": (
                "https://www.linkedin.com/"
                "in/mahnoor-m-akram-998a86283"
            ),

            "description": (
                "Mahnoor designs the user experience and visual "
                "presentation of HormoneBench AI. She works on the "
                "Streamlit interface, project documentation, report "
                "layout, demo preparation, and user-friendly research "
                "communication."
            ),
        },
    ]

    # --------------------------------------------------------
    # TEAM MEMBER CARD FUNCTION
    # --------------------------------------------------------

    def display_team_member(member: dict) -> None:

        linkedin_url = member.get(
            "linkedin",
            "",
        )

        if linkedin_url:

            linkedin_element = f"""
            <a
                class="linkedin-button"
                href="{linkedin_url}"
                target="_blank"
                rel="noopener noreferrer"
            >
                View LinkedIn Profile
            </a>
            """

        else:

            linkedin_element = """
            <span class="linkedin-unavailable">
                LinkedIn Profile Coming Soon
            </span>
            """

        team_card_html = textwrap.dedent(
            f"""
            <div class="team-card">
                <img
                    class="team-image"
                    src="{member['image']}"
                    alt="{member['name']}"
                >
                <div class="team-name">{member['name']}</div>
                <div class="team-role">{member['role']}</div>
                <div class="team-description">{member['description']}</div>
                {linkedin_element}
            </div>
            """
        ).strip()

        st.markdown(
            team_card_html,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # FIRST ROW
    # --------------------------------------------------------

    first_column, second_column = st.columns(
        2,
        gap="large",
    )

    with first_column:

        display_team_member(
            team_members[0]
        )

    with second_column:

        display_team_member(
            team_members[1]
        )

    # --------------------------------------------------------
    # SECOND ROW
    # --------------------------------------------------------

    third_column, fourth_column = st.columns(
        2,
        gap="large",
    )

    with third_column:

        display_team_member(
            team_members[2]
        )

    with fourth_column:

        display_team_member(
            team_members[3]
        )