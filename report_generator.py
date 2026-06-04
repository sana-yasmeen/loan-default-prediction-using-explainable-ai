"""
PDF Report Generator using fpdf2
"""

import os
from datetime import datetime

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False


def generate_pdf_report(data: dict) -> bytes:
    if not FPDF_AVAILABLE:
        return b""

    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_fill_color(15, 52, 96)
    pdf.rect(0, 0, 210, 30, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_y(8)
    pdf.cell(0, 10, "LoanAI - Loan Assessment Report", align="C", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%d %B %Y %H:%M')}", align="C", ln=True)

    pdf.set_text_color(0, 0, 0)
    pdf.ln(8)

    def section_header(title):
        pdf.set_fill_color(230, 240, 255)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, f"  {title}", fill=True, ln=True)
        pdf.ln(2)

    def row(label, value, color=False):
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_fill_color(248, 248, 248)
        pdf.cell(80, 7, f"  {label}:", fill=color, border=0)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, str(value), ln=True)

    # Personal Details
    section_header("Personal Details")
    row("Full Name", data.get("full_name", "N/A"))
    row("Age", data.get("age", "N/A"))
    row("Gender", data.get("gender", "N/A"))
    row("Phone", data.get("phone", "N/A"))
    row("Email", data.get("email", "N/A"))
    pdf.ln(3)

    # Financial Details
    section_header("Financial Details")
    row("Monthly Income", f"Rs {data.get('monthly_income', 0):,.2f}")
    row("CIBIL Score", data.get("cibil_score", "N/A"))
    row("Existing Loans", data.get("existing_loans", 0))
    row("Existing EMI", f"Rs {data.get('existing_emi', 0):,.2f}")
    pdf.ln(3)

    # Loan Details
    section_header("Loan Details")
    row("Loan Type", data.get("loan_type", "N/A"))
    row("Loan Amount", f"Rs {data.get('loan_amount', 0):,.2f}")
    row("Bank", data.get("bank_name", "N/A"))
    row("Interest Rate", f"{data.get('interest_rate', 0):.2f}% per annum")
    row("Tenure", f"{data.get('loan_tenure', 0)} months")
    pdf.ln(3)

    # EMI Summary
    section_header("EMI Summary")
    row("Monthly EMI", f"Rs {data.get('emi', 0):,.2f}")
    row("Total Interest", f"Rs {data.get('total_interest', 0):,.2f}")
    row("Total Payment", f"Rs {data.get('total_payment', 0):,.2f}")
    pdf.ln(3)

    # Prediction Result
    section_header("AI Prediction Result")
    pred = data.get("prediction", "N/A")
    risk = data.get("risk_level", "N/A")
    prob = data.get("approval_probability", 0)

    risk_color = {"Low Risk": (0, 150, 50), "Medium Risk": (200, 130, 0), "High Risk": (200, 30, 30)}
    rc = risk_color.get(risk, (100, 100, 100))
    pdf.set_text_color(*rc)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, f"  Result: {pred}  |  Risk: {risk}  |  Confidence: {prob*100:.1f}%", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

    # Eligibility
    section_header("Eligibility Status")
    elig = data.get("eligibility", "N/A")
    pdf.set_font("Helvetica", "B", 11)
    elig_color = (0, 150, 50) if elig == "Eligible" else (200, 30, 30)
    pdf.set_text_color(*elig_color)
    pdf.cell(0, 8, f"  Eligibility: {elig}", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

    # Top SHAP factors
    section_header("Key Factors (Explainable AI)")
    pdf.set_font("Helvetica", "", 10)
    top_pos = data.get("top_positive_factors", [])
    top_neg = data.get("top_negative_factors", [])

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "  Risk-Increasing Factors:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for label, val in (top_pos or [])[:3]:
        pdf.cell(0, 6, f"    + {label}: {val:+.4f}", ln=True)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "  Risk-Decreasing Factors:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for label, val in (top_neg or [])[:3]:
        pdf.cell(0, 6, f"    - {label}: {val:+.4f}", ln=True)

    # Footer
    pdf.set_y(-20)
    pdf.set_fill_color(15, 52, 96)
    pdf.rect(0, pdf.get_y(), 210, 20, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(0, 10, "LoanAI - AI-Powered Loan Management System | Confidential Report", align="C")

    return bytes(pdf.output())
