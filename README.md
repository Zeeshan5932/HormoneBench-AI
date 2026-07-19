HormoneBench AI

Open AI Research Infrastructure for Women's Hormonal Health

HormoneBench AI is an open-source research platform that standardizes, analyzes, and benchmarks women's hormonal health datasets through an integrated AI-powered scientific validation pipeline.

The platform enables researchers, students, and healthcare innovators to transform heterogeneous datasets into a unified benchmark, generate transparent AI predictions, explain model outputs, retrieve evidence from scientific literature, and produce downloadable research reports.

Disclaimer: HormoneBench AI is a research and educational project. It is not a medical device and should not be used for diagnosis, treatment, or clinical decision-making.

Features
Dataset Standardization
Upload CSV datasets
Automatic schema mapping
Unified HormoneBench benchmark format
Missing value handling
Duplicate removal
Data validation
Data Processing Pipeline
Automated data cleaning
Feature preprocessing
Feature engineering
Standardized benchmark generation
AI Risk Prediction
AI-powered hormonal health risk assessment
Record-level predictions
Risk scoring
Confidence estimation
Important contributing factors
Explainability
Feature contribution analysis
Global feature importance
Record-level explanations
Transparent prediction pipeline
AI Research Copilot

Powered by Retrieval-Augmented Generation (RAG).

Supports evidence retrieval from:

Local knowledge base
Scientific literature
ArXiv
Public web sources

Provides evidence-grounded answers for women's hormonal health research.

Research Reporting

Automatically generates downloadable reports containing:

Dataset summary
Pipeline metadata
Prediction statistics
Risk distribution
Prediction preview
Research disclaimer
Project Workflow
CSV Upload
      │
      ▼
Dataset Validation
      │
      ▼
Schema Mapping
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Preprocessing
      │
      ▼
AI Risk Prediction
      │
      ▼
Explainability
      │
      ▼
AI Research Copilot (RAG)
      │
      ▼
Research Report Generation
      │
      ▼
Download Results
Tech Stack
Frontend
Streamlit
Backend
Python 3.11+
Data Processing
Pandas
NumPy
Machine Learning
Scikit-learn
AI & LLM
Google Gemini
LangChain
Vector Retrieval
FAISS
Research Sources
ArXiv
DuckDuckGo Search
Local Knowledge Base
Reporting
HTML Reports
PDF Reports (when supported)
Project Structure
HormoneBench-AI/
│
├── app.py
├── requirements.txt
├── assets/
│   └── style.css
│
├── services/
│   ├── cleaning_service.py
│   ├── preprocessing_service.py
│   ├── prediction_service.py
│   ├── rag_service.py
│   ├── report_service.py
│   └── __init__.py
│
├── utils/
│   ├── constants.py
│   ├── schema_mapper.py
│   └── __init__.py
│
├── knowledge_base/
├── models/
├── data/
└── README.md
Installation

Clone the repository

git clone https://github.com/yourusername/HormoneBench-AI.git

cd HormoneBench-AI

Create a virtual environment

python -m venv .venv

Activate the environment

Windows

.venv\Scripts\activate

Linux/macOS

source .venv/bin/activate

Install dependencies

pip install -r requirements.txt
Environment Variables

Create a .env file.

GOOGLE_API_KEY=YOUR_GEMINI_API_KEY
Run Locally
streamlit run app.py
Deployment

The application can be deployed on:

Streamlit Community Cloud
Railway
Render

For Streamlit Community Cloud:

Push the project to GitHub.
Create a new app on Streamlit Community Cloud.
Select the repository and app.py as the entry point.
Add your GOOGLE_API_KEY under App Settings → Secrets.
Deploy.
Example Dataset

The application expects a CSV containing hormonal health features.

Required standardized columns:

age
sleep
heart_rate
temperature
stress
cycle_day
bmi

The platform automatically maps common alternative column names into the unified schema whenever possible.

Research Components
Dataset Standardization
Scientific Validation Pipeline
AI Risk Prediction
Explainable AI
Retrieval-Augmented Generation (RAG)
Automated Research Reporting
Limitations
Research prototype
Not clinically validated
Predictions should not be interpreted as medical advice
Requires a Gemini API key for AI Research Copilot functionality
Team
Role	Member
AI Architect	Tayyab Nisar
Lead Data Scientist	Beenish Mehar
UX Researcher	Mahnoor Akram
Technical Lead (CTO)	Zeeshan
License

This project is intended for research, education, and non-commercial innovation unless otherwise specified.

Citation

If you use HormoneBench AI in your research or academic work, please cite the project appropriately.

Contact

Tayyab Nisar

LinkedIn: https://www.linkedin.com/in/tayyabnisar

Acknowledgements

Special thanks to the open-source community and the developers of:

Streamlit
Pandas
NumPy
Scikit-learn
LangChain
Google Gemini
FAISS
ArXiv
DuckDuckGo Search

for enabling accessible AI research and innovation in women's health.
