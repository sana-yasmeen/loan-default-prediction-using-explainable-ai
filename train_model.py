"""
ML Model Training Module
Trains Random Forest, XGBoost, Logistic Regression, and Decision Tree
for Loan Default Prediction. Uses synthetic realistic dataset.
"""

import numpy as np
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, roc_auc_score)
from xgboost import XGBClassifier

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

FEATURE_COLS = [
    "age", "dependents", "work_experience",
    "monthly_income", "annual_income", "existing_loans",
    "existing_emi", "savings", "assets", "cibil_score",
    "loan_amount", "interest_rate", "loan_tenure",
    "dti_ratio", "loan_to_income_ratio",
    "gender_enc", "marital_enc", "education_enc",
    "occupation_enc", "loan_type_enc"
]


def generate_dataset(n=5000):
    np.random.seed(42)
    ages = np.random.randint(21, 65, n)
    monthly_income = np.random.lognormal(10.5, 0.6, n).clip(10000, 500000)
    cibil = np.random.randint(300, 900, n)
    loan_amount = np.random.lognormal(13, 0.8, n).clip(50000, 5000000)
    tenure = np.random.choice([12, 24, 36, 48, 60, 84, 120, 180, 240], n)
    interest = np.random.uniform(7.0, 14.0, n)
    existing_emi = monthly_income * np.random.uniform(0, 0.5, n)
    savings = monthly_income * np.random.uniform(1, 24, n)
    assets = savings * np.random.uniform(1, 5, n)
    existing_loans = np.random.randint(0, 5, n)
    dependents = np.random.randint(0, 5, n)
    work_exp = np.random.uniform(0, 30, n)
    annual_income = monthly_income * 12

    dti = existing_emi / monthly_income
    loan_to_inc = loan_amount / annual_income

    genders = np.random.choice(["Male", "Female"], n)
    marital = np.random.choice(["Single", "Married", "Divorced"], n)
    educations = np.random.choice(["High School", "Graduate", "Post Graduate", "PhD"], n)
    occupations = np.random.choice(["Salaried", "Business", "Self-employed", "Government", "Student"], n)
    loan_types = np.random.choice(["Personal Loan","Home Loan","Education Loan",
                                   "Vehicle Loan","Gold Loan","Business Loan","Agricultural Loan"], n)

    # Build default probability based on risk factors
    risk = (
        (cibil < 600).astype(float) * 0.35 +
        (dti > 0.4).astype(float) * 0.20 +
        (loan_to_inc > 5).astype(float) * 0.15 +
        (existing_loans > 2).astype(float) * 0.10 +
        (monthly_income < 25000).astype(float) * 0.10 +
        (ages < 25).astype(float) * 0.05 +
        np.random.uniform(0, 0.15, n)
    )
    default = (risk > 0.45).astype(int)

    df = pd.DataFrame({
        "age": ages, "dependents": dependents, "work_experience": work_exp,
        "monthly_income": monthly_income.round(2),
        "annual_income": annual_income.round(2),
        "existing_loans": existing_loans,
        "existing_emi": existing_emi.round(2),
        "savings": savings.round(2),
        "assets": assets.round(2),
        "cibil_score": cibil,
        "loan_amount": loan_amount.round(2),
        "interest_rate": interest.round(2),
        "loan_tenure": tenure,
        "dti_ratio": dti.round(4),
        "loan_to_income_ratio": loan_to_inc.round(4),
        "gender": genders, "marital_status": marital,
        "education": educations, "occupation": occupations,
        "loan_type": loan_types,
        "default": default
    })
    return df


def encode_features(df):
    le = {}
    col_map = {"gender": "gender", "marital_status": "marital",
               "education": "education", "occupation": "occupation", "loan_type": "loan_type"}
    for col, short in col_map.items():
        enc = LabelEncoder()
        df[short + "_enc"] = enc.fit_transform(df[col].astype(str))
        le[col] = enc
    return df, le


def train_all_models():
    print("Generating dataset...")
    df = generate_dataset(5000)
    df, encoders = encode_features(df)
    df.to_csv(os.path.join(DATASET_DIR, "loan_data.csv"), index=False)

    X = df[FEATURE_COLS]
    y = df["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=15,
                                                random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                 use_label_encoder=False, eval_metric="logloss",
                                 random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    }

    results = {}
    best_model_name = None
    best_score = 0

    for name, model in models.items():
        print(f"Training {name}...")
        if name == "Logistic Regression":
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            y_prob = model.predict_proba(X_test_s)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
        }
        print(f"  {name}: Acc={acc:.4f} AUC={auc:.4f}")

        if auc > best_score:
            best_score = auc
            best_model_name = name

        joblib.dump(model, os.path.join(MODELS_DIR, f"{name.replace(' ','_').lower()}.pkl"))

    # Save best model separately
    best_model = joblib.load(os.path.join(MODELS_DIR, f"{best_model_name.replace(' ','_').lower()}.pkl"))
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(encoders, os.path.join(MODELS_DIR, "encoders.pkl"))
    joblib.dump(FEATURE_COLS, os.path.join(MODELS_DIR, "feature_cols.pkl"))
    joblib.dump(results, os.path.join(MODELS_DIR, "model_results.pkl"))
    joblib.dump(best_model_name, os.path.join(MODELS_DIR, "best_model_name.pkl"))

    print(f"\nBest Model: {best_model_name} (AUC={best_score:.4f})")
    print("All models saved!")
    return results, best_model_name


if __name__ == "__main__":
    train_all_models()
