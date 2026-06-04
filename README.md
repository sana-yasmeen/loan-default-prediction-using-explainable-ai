# 🏦 LoanAI — AI-Powered Loan Default Prediction System

> **B.Tech Final Year Major Project | IEEE Paper Implementation | Hackathon Project**

A complete, production-grade AI-powered loan management platform with Explainable AI (XAI), real-time EMI calculation, SHAP-based predictions, and interactive analytics dashboards.

---

## 🎯 Features

| Feature | Description |
|---|---|
| 📋 Loan Application | Full form with 7 loan types, 6 banks |
| 🧮 EMI Calculator | Real-time with amortization charts |
| ✅ Eligibility Checker | Rule-based + scoring system |
| 🤖 ML Prediction | XGBoost, RF, LR, DT models |
| 🔍 XAI Dashboard | SHAP waterfall, force, summary plots |
| 📊 Analytics | Plotly interactive dashboards |
| ⚙️ Admin Panel | Application management, model stats |
| 💬 AI Assistant | Rule-based chatbot for loan queries |
| 📥 Report Download | PDF + CSV report generation |

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit + Plotly
- **ML:** Scikit-learn, XGBoost, Random Forest
- **XAI:** SHAP (SHapley Additive exPlanations)
- **Database:** SQLite
- **Reports:** FPDF2

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd Loan_Default_Prediction_Project
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run app.py
```

### 3. First Launch

- App auto-trains ML models on first run (~30 seconds)
- Default admin: `admin` / `admin123`

---

## 📁 Project Structure

```
Loan_Default_Prediction_Project/
├── app.py                  # Main Streamlit application (all pages)
├── train_model.py          # ML training (RF, XGBoost, LR, DT)
├── shap_explainer.py       # SHAP explanations module
├── emi_calculator.py       # EMI formula + bank rates
├── database.py             # SQLite operations
├── report_generator.py     # PDF report generation
├── requirements.txt        # Dependencies
├── README.md               # This file
├── dataset/                # Generated loan dataset (CSV)
├── models/                 # Trained model .pkl files
└── database/               # SQLite database file
```

---

## 🧠 Machine Learning Models

| Model | Notes |
|---|---|
| Random Forest | 200 estimators, depth 15 |
| XGBoost | 200 estimators, lr=0.1 |
| Logistic Regression | Scaled features |
| Decision Tree | Depth 10 |

Best model auto-selected by ROC-AUC score.

---

## 🔍 Explainable AI (SHAP)

SHAP values explain individual predictions:

- **Waterfall Plot:** Feature-by-feature impact on prediction
- **Force Plot:** Risk vs protective factor visualization
- **Summary Plot:** All feature SHAP values ranked
- **Global Feature Importance:** Trained model's feature weights

---

## 🏛️ Supported Banks & Loans

Banks: SBI, HDFC, ICICI, Axis Bank, Canara Bank, Union Bank

Loans: Personal, Home, Education, Vehicle, Gold, Business, Agricultural

---

## ☁️ Deployment

### Streamlit Cloud
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set `app.py` as main file

### Render
```bash
# Build command
pip install -r requirements.txt
# Start command
streamlit run app.py --server.port $PORT
```

### Railway
```bash
railway init
railway add
railway deploy
```

---

## 📊 CIBIL Score Classification

| Range | Category | Approval Chance |
|---|---|---|
| 750–900 | Excellent | High |
| 650–750 | Good | Moderate |
| 550–650 | Fair | Low |
| 300–550 | Poor | Very Low |

---

## 👥 Author

**B.Tech Final Year Project**  
AI-Powered Loan Default Prediction Using Explainable AI Techniques

---

## 📄 License

MIT License — Free to use for academic and educational purposes.
