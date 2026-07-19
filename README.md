# 🩺 HormoneBench AI

HormoneBench AI is a Streamlit-based MVP for benchmarking women's hormonal health datasets.

The project allows researchers to upload hormone datasets, clean them, convert them into a unified schema, generate a simple hormonal risk prediction, and download a professional PDF report.

---

# Features

- CSV Dataset Upload
- Automatic Data Cleaning
- Unified Schema Mapping
- Rule-Based Hormonal Risk Prediction
- Interactive Dashboard
- PDF Report Generation
- Streamlit Web Interface

---

# Project Structure

```
HormoneBench-AI/
│
├── app.py
├── requirements.txt
├── README.md
│
├── assets/
│   ├── logo.png
│   └── style.css
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── reports/
│
├── models/
│   └── prediction.py
│
├── pages/
│   ├── 1_Dashboard.py
│   ├── 2_Upload_Data.py
│   ├── 3_Prediction.py
│   ├── 4_Report.py
│   └── 5_About.py
│
├── services/
│   ├── storage_service.py
│   ├── cleaning_service.py
│   ├── preprocessing_service.py
│   ├── prediction_service.py
│   └── report_service.py
│
└── utils/
    ├── constants.py
    ├── helpers.py
    └── schema_mapper.py
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/your-username/HormoneBench-AI.git
```

Go to project folder

```bash
cd HormoneBench-AI
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Run Project

```bash
streamlit run app.py
```

Application will open at

```
http://localhost:8501
```

---

# Workflow

```
Upload CSV

↓

Data Cleaning

↓

Schema Mapping

↓

Prediction

↓

Generate Report
```

---

# Prediction Logic

Current MVP uses a simple rule-based prediction.

Example:

- Low Sleep
- High Stress
- High Temperature
- High Heart Rate

↓

High Hormonal Risk

Future versions will replace this with Machine Learning models.

---

# Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-Learn
- ReportLab

---

# Future Improvements

- XGBoost Model
- LightGBM
- SHAP Explainability
- Research Benchmark Leaderboard
- Multi-Dataset Support
- REST API
- Docker Deployment

---

# Author

HormoneBench AI

MVP for Women's Hormonal Health Benchmark Research