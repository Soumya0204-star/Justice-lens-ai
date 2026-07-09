# ⚖️ JusticeLens AI

> **AI-Powered Decision Support System for Tele-Law Disparity Analysis**
>
> **IBM SkillsBuild Internship | Problem Statement 37**

JusticeLens AI is an AI-powered analytics platform designed to help policymakers identify regional and demographic disparities in India's Tele-Law initiative. By combining Machine Learning, Explainable AI (SHAP), and IBM watsonx Granite models, the system transforms raw Tele-Law registration data into actionable policy insights.

---

## 📌 Table of Contents

* [Project Overview](#-project-overview)
* [Problem Statement](#-problem-statement)
* [Solution](#-solution)
* [Key Features](#-key-features)
* [System Architecture](#-system-architecture)
* [Machine Learning Pipeline](#-machine-learning-pipeline)
* [Explainable AI](#-explainable-ai)
* [IBM watsonx Integration](#-ibm-watsonx-integration)
* [Technology Stack](#-technology-stack)
* [Project Structure](#-project-structure)
* [Installation](#-installation)
* [Configuration](#-configuration)
* [Running the Project](#-running-the-project)
* [Deployment](#-deployment)
* [Screenshots](#-screenshots)
* [Evaluation Highlights](#-evaluation-highlights)
* [Future Enhancements](#-future-enhancements)
* [License](#-license)
* [Acknowledgements](#-acknowledgements)
* [Author](#-author)

---

# 📖 Project Overview

Tele-Law is a Government of India initiative that provides free legal assistance to citizens through Common Service Centres (CSCs). While the initiative has significantly improved legal accessibility, registration rates differ considerably across districts, states, and demographic groups.

JusticeLens AI enables policymakers and administrators to understand these disparities using data-driven insights instead of manual reporting.

The platform integrates predictive analytics, explainable machine learning, and Large Language Models to identify underserved regions and generate policy recommendations that support informed decision-making.

---

# ❗ Problem Statement

Government agencies currently face several challenges:

* Difficulty identifying underserved districts
* Lack of explainability behind disparity trends
* Time-consuming manual analysis of large datasets
* Limited decision-support tools for policymakers
* No automated policy recommendation system

These limitations delay interventions and reduce the effectiveness of welfare programs.

---

# 💡 Solution

JusticeLens AI addresses these challenges by combining:

* Machine Learning for disparity prediction
* SHAP Explainable AI for transparent predictions
* IBM Granite (watsonx.ai) for AI-generated narratives
* Interactive Streamlit dashboards
* Automated policy recommendations
* Natural language question answering

The result is an intelligent decision-support system that assists policymakers in making faster, evidence-based decisions.

---

# ✨ Key Features

### 📊 Interactive Dashboard

* Executive Dashboard
* District Explorer
* State Comparison
* Predictive Intelligence
* Explainable AI
* IBM watsonx Policy Room
* About Page

### 🤖 Machine Learning

* Multiple classification models
* Automatic model comparison
* Cross-validation
* Best model selection
* Performance visualization

### 🔍 Explainable AI

* Global Feature Importance
* Local SHAP Explanations
* Prediction Transparency

### 💬 IBM Granite Integration

* Executive summaries
* Policy recommendations
* AI-generated reports
* Natural language Q&A

### 📈 Analytics

* District-level insights
* State-level comparisons
* Demographic analysis
* Registration trend analysis

### ⚙️ Enterprise Features

* Modular architecture
* FastAPI backend
* Streamlit frontend
* Docker support
* Cloud deployment ready

---

# 🏗️ System Architecture

```text
Government Tele-Law Dataset
            │
            ▼
Data Cleaning & Harmonization
            │
            ▼
Feature Engineering
            │
            ▼
Machine Learning Pipeline
            │
            ▼
Best Model Selection
            │
            ▼
SHAP Explainability
            │
            ▼
IBM Granite (watsonx.ai)
            │
            ▼
Interactive Dashboard
```

---

# 🧠 Machine Learning Pipeline

The application trains and evaluates multiple classification algorithms.

| Model               | Purpose                            |
| ------------------- | ---------------------------------- |
| Logistic Regression | Baseline model                     |
| Decision Tree       | Rule-based classification          |
| Random Forest       | Ensemble learning                  |
| Gradient Boosting   | Sequential boosting                |
| XGBoost             | High-performance gradient boosting |

### Model Evaluation

* 5-Fold Cross Validation
* ROC-AUC Score
* Accuracy
* Precision
* Recall
* F1 Score

The highest-performing model is automatically selected for prediction.

---

# 🔍 Explainable AI

JusticeLens AI incorporates SHAP (SHapley Additive Explanations) to ensure model transparency.

### Global Explanations

Identify the most influential features affecting Tele-Law disparities across all districts.

### Local Explanations

Explain why an individual district is classified as underserved.

This helps policymakers understand not only *what* the model predicts, but also *why*.

---

# 💬 IBM watsonx Integration

IBM Granite Foundation Models generate human-readable insights including:

* Executive summaries
* District analysis
* Policy recommendations
* AI-generated reports
* Question answering interface

This converts complex analytical outputs into reports that can be directly understood by administrators and policymakers.

---

# 🛠 Technology Stack

| Category             | Technologies                    |
| -------------------- | ------------------------------- |
| Programming Language | Python                          |
| Machine Learning     | Scikit-learn, XGBoost           |
| Explainability       | SHAP                            |
| AI Foundation Model  | IBM Granite                     |
| Cloud Platform       | IBM Cloud (watsonx.ai)          |
| Frontend             | Streamlit                       |
| Backend              | FastAPI                         |
| Visualization        | Plotly                          |
| API Server           | Uvicorn                         |
| Deployment           | Docker, Render, Streamlit Cloud |
| Version Control      | Git & GitHub                    |

---

# 📂 Project Structure

```text
JusticeLensAI/
│
├── app/
│   ├── Home.py
│   ├── common/
│   └── pages/
│
├── api/
│   └── main.py
│
├── justicelens/
│   ├── data_loader.py
│   ├── data_engineering.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── shap_explainability.py
│   ├── watsonx_integration.py
│   ├── policy_recommendation_engine.py
│   ├── qa_engine.py
│   └── ai_report_generator.py
│
├── orchestrator/
│   └── run_pipeline.py
│
├── data/
├── models/
├── outputs/
├── docs/
│
├── README.md
├── requirements.txt
├── Dockerfile
├── render.yaml
└── .env.example
```

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/Soumya0204-star/Justice-lens-ai.git

cd Justice-lens-ai
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ⚙️ Configuration

Create a `.env` file using the provided template.

```env
WATSONX_API_KEY=your_api_key
WATSONX_PROJECT_ID=your_project_id
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_MODEL_ID=ibm/granite-13b-instruct-v2
```

---

# ▶️ Running the Project

### Train the Machine Learning Pipeline

```bash
python orchestrator/run_pipeline.py
```

### Start the FastAPI Backend

```bash
uvicorn api.main:app --reload
```

### Launch the Streamlit Dashboard

```bash
streamlit run app/Home.py
```

---

# 🌐 Deployment

### Live Application

https://justice-lens-ai-pxby8xqnatrbbsbcr4tnv6.streamlit.app/

### Backend API

https://justice-lens-api.onrender.com/docs

### GitHub Repository

https://github.com/Soumya0204-star/Justice-lens-ai

---

# 📷 Screenshots

Add screenshots of:

* Executive Dashboard
* District Explorer
* State Comparison
* Predictive Intelligence
* Explainable AI
* IBM watsonx Policy Room

---

# 📊 Evaluation Highlights

| Criteria              | Status |
| --------------------- | ------ |
| IBM Cloud Integration | ✅      |
| AI Innovation         | ✅      |
| Explainable AI        | ✅      |
| Scalable Architecture | ✅      |
| Deployment Ready      | ✅      |
| Social Impact         | ✅      |

---

# 🚀 Future Enhancements

* Real-time Tele-Law data integration
* GIS-based district heatmaps
* Time-series forecasting
* Multi-language AI reports
* Role-based authentication
* Policy impact simulation
* Automated monitoring dashboards
* Mobile application support

---

# 📜 License

This project was developed as part of the **IBM SkillsBuild Internship Program** conducted by **Edunet Foundation** in collaboration with **AICTE**.

This repository is intended for educational and demonstration purposes.

---

# 🙏 Acknowledgements

Special thanks to:

* IBM SkillsBuild
* IBM Cloud
* IBM watsonx.ai
* IBM Granite Foundation Models
* IBM Bob
* Edunet Foundation
* AICTE
* data.gov.in for the Tele-Law dataset

---

# 👩‍💻 Author

**Soumya Bhat**

**GitHub:** https://github.com/Soumya0204-star

**LinkedIn:** *Add your LinkedIn profile here*

---

# ⭐ Support

If you found this project useful, consider giving the repository a **⭐ Star** on GitHub.

---

> **"Justice delayed is justice denied. JusticeLens AI helps ensure that no district is left behind in access to legal services."**
