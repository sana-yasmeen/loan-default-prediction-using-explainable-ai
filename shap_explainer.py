"""
SHAP Explainer Module
Generates SHAP explanations for loan default predictions.
"""

import numpy as np
import pandas as pd
import shap
import joblib
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

FEATURE_LABELS = {
    "age": "Age",
    "dependents": "Dependents",
    "work_experience": "Work Experience (yrs)",
    "monthly_income": "Monthly Income (₹)",
    "annual_income": "Annual Income (₹)",
    "existing_loans": "Existing Loans",
    "existing_emi": "Existing EMI (₹)",
    "savings": "Savings (₹)",
    "assets": "Assets (₹)",
    "cibil_score": "CIBIL Score",
    "loan_amount": "Loan Amount (₹)",
    "interest_rate": "Interest Rate (%)",
    "loan_tenure": "Loan Tenure (months)",
    "dti_ratio": "Debt-to-Income Ratio",
    "loan_to_income_ratio": "Loan-to-Income Ratio",
    "gender_enc": "Gender",
    "marital_enc": "Marital Status",
    "education_enc": "Education Level",
    "occupation_enc": "Occupation",
    "loan_type_enc": "Loan Type",
}

_explainer_cache = {}


def _load_explainer(model_name="best_model"):
    if model_name in _explainer_cache:
        return _explainer_cache[model_name]
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    if not os.path.exists(model_path):
        return None
    model = joblib.load(model_path)
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(
            pd.DataFrame(np.zeros((50, 20)), columns=list(FEATURE_LABELS.keys())), 10))
    _explainer_cache[model_name] = (model, explainer)
    return model, explainer


def predict_and_explain(features_dict: dict) -> dict:
    """
    Given a features dict, returns prediction + SHAP values.
    """
    loaded = _load_explainer()
    if loaded is None:
        return {"error": "Model not trained yet."}

    model, explainer = loaded
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))

    X = pd.DataFrame([features_dict])[feature_cols]

    prob = model.predict_proba(X)[0][1]
    pred = int(prob > 0.5)

    try:
        shap_vals = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            sv = shap_vals[1][0]
        else:
            sv = shap_vals[0]

        shap_dict = {col: float(sv[i]) for i, col in enumerate(feature_cols)}
        expected_value = float(explainer.expected_value[1] if isinstance(
            explainer.expected_value, (list, np.ndarray)) else explainer.expected_value)
    except Exception as e:
        shap_dict = {col: 0.0 for col in feature_cols}
        expected_value = 0.5

    # Risk level
    if prob < 0.3:
        risk_level = "Low Risk"
    elif prob < 0.6:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"

    # Top factors
    sorted_feats = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    top_positive = [(FEATURE_LABELS.get(k, k), v) for k, v in sorted_feats if v > 0][:5]
    top_negative = [(FEATURE_LABELS.get(k, k), v) for k, v in sorted_feats if v < 0][:5]

    return {
        "prediction": pred,
        "probability": round(float(prob), 4),
        "risk_level": risk_level,
        "shap_values": shap_dict,
        "expected_value": expected_value,
        "top_positive_factors": top_positive,
        "top_negative_factors": top_negative,
        "feature_values": features_dict,
    }


def get_shap_waterfall_data(result: dict) -> dict:
    sv = result["shap_values"]
    labels = [FEATURE_LABELS.get(k, k) for k in sv.keys()]
    values = list(sv.values())
    return {"labels": labels, "values": values, "base_value": result["expected_value"]}


def get_feature_importance_data() -> dict:
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    if not os.path.exists(model_path):
        return {}
    model = joblib.load(model_path)
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        return {FEATURE_LABELS.get(f, f): float(fi[i]) for i, f in enumerate(feature_cols)}
    return {}
