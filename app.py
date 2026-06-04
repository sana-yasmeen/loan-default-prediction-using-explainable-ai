"""
LoanAI - AI-Powered Loan Default Prediction System
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import sys
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import database as db
from emi_calculator import calculate_emi, get_rate, BANKS, LOAN_TYPES, BANK_RATES

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LoanAI - Intelligent Loan Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS Styling ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
  --navy: #0F3460;
  --navy-light: #16213E;
  --gold: #E8B86D;
  --gold-dark: #C9982A;
  --success: #00B894;
  --danger: #E17055;
  --warning: #FDCB6E;
  --text: #2D3748;
  --muted: #718096;
  --bg-card: #FFFFFF;
  --bg-page: #F0F4F8;
  --border: #E2E8F0;
}

html, body, [data-testid="stAppViewContainer"] {
  font-family: 'DM Sans', sans-serif;
  background-color: var(--bg-page);
  color: var(--text);
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0F3460 0%, #16213E 100%) !important;
  border-right: 3px solid #E8B86D;
}

[data-testid="stSidebar"] * {
  color: #F8F9FA !important;
}

[data-testid="stSidebar"] .stRadio label {
  padding: 8px 12px;
  border-radius: 8px;
  transition: background 0.2s;
}

.stRadio [data-testid="stMarkdownContainer"] p {
  font-size: 0.95rem !important;
}

h1, h2, h3 {
  font-family: 'Playfair Display', serif;
  color: var(--navy);
}

.hero-card {
  background: linear-gradient(135deg, #0F3460 0%, #1A4A80 50%, #0F3460 100%);
  border-radius: 20px;
  padding: 40px;
  color: white;
  margin-bottom: 24px;
  position: relative;
  overflow: hidden;
}
.hero-card::before {
  content: '';
  position: absolute;
  top: -50px; right: -50px;
  width: 200px; height: 200px;
  background: rgba(232, 184, 109, 0.15);
  border-radius: 50%;
}
.hero-card::after {
  content: '';
  position: absolute;
  bottom: -30px; left: -30px;
  width: 150px; height: 150px;
  background: rgba(232, 184, 109, 0.1);
  border-radius: 50%;
}
.hero-title {
  font-family: 'Playfair Display', serif;
  font-size: 2.4rem;
  font-weight: 700;
  margin-bottom: 8px;
  position: relative;
  z-index: 1;
}
.hero-subtitle {
  font-size: 1.1rem;
  opacity: 0.85;
  position: relative;
  z-index: 1;
}
.hero-gold {
  color: #E8B86D;
}

.metric-card {
  background: white;
  border-radius: 16px;
  padding: 24px;
  box-shadow: 0 4px 20px rgba(15, 52, 96, 0.08);
  border-left: 4px solid var(--gold);
  margin-bottom: 16px;
  transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 30px rgba(15, 52, 96, 0.15);
}
.metric-value {
  font-size: 2rem;
  font-weight: 700;
  color: var(--navy);
  font-family: 'Playfair Display', serif;
}
.metric-label {
  font-size: 0.85rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-weight: 500;
}

.section-card {
  background: white;
  border-radius: 16px;
  padding: 28px;
  box-shadow: 0 4px 20px rgba(15, 52, 96, 0.06);
  margin-bottom: 20px;
  border: 1px solid var(--border);
}

.risk-badge-low {
  background: #D4EDDA; color: #155724;
  padding: 4px 14px; border-radius: 20px;
  font-weight: 600; font-size: 0.9rem; display: inline-block;
}
.risk-badge-medium {
  background: #FFF3CD; color: #856404;
  padding: 4px 14px; border-radius: 20px;
  font-weight: 600; font-size: 0.9rem; display: inline-block;
}
.risk-badge-high {
  background: #F8D7DA; color: #721C24;
  padding: 4px 14px; border-radius: 20px;
  font-weight: 600; font-size: 0.9rem; display: inline-block;
}

.approved-banner {
  background: linear-gradient(135deg, #00B894, #00A381);
  color: white; padding: 20px 28px; border-radius: 16px;
  text-align: center; font-size: 1.5rem; font-weight: 700;
  font-family: 'Playfair Display', serif;
  box-shadow: 0 8px 25px rgba(0,184,148,0.35);
  margin: 16px 0;
}
.rejected-banner {
  background: linear-gradient(135deg, #E17055, #C0392B);
  color: white; padding: 20px 28px; border-radius: 16px;
  text-align: center; font-size: 1.5rem; font-weight: 700;
  font-family: 'Playfair Display', serif;
  box-shadow: 0 8px 25px rgba(225,112,85,0.35);
  margin: 16px 0;
}

.cibil-poor   { color: #E17055; font-weight: 700; }
.cibil-bad    { color: #FDCB6E; font-weight: 700; }
.cibil-avg    { color: #74B9FF; font-weight: 700; }
.cibil-good   { color: #00B894; font-weight: 700; }

.stButton>button {
  background: linear-gradient(135deg, #0F3460, #1A4A80) !important;
  color: white !important;
  border: none !important;
  border-radius: 10px !important;
  padding: 10px 28px !important;
  font-weight: 600 !important;
  font-family: 'DM Sans', sans-serif !important;
  letter-spacing: 0.03em !important;
  transition: all 0.2s !important;
  box-shadow: 0 4px 14px rgba(15,52,96,0.3) !important;
}
.stButton>button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 20px rgba(15,52,96,0.4) !important;
  background: linear-gradient(135deg, #1A4A80, #0F3460) !important;
}

.info-box {
  background: linear-gradient(135deg, #EEF2FF, #E0E7FF);
  border-left: 4px solid #6366F1;
  border-radius: 0 12px 12px 0;
  padding: 16px 20px;
  margin: 12px 0;
}

.sidebar-logo {
  text-align: center;
  padding: 20px 10px;
  border-bottom: 1px solid rgba(232,184,109,0.3);
  margin-bottom: 20px;
}
.sidebar-logo h2 {
  color: #E8B86D !important;
  font-family: 'Playfair Display', serif;
  font-size: 1.6rem;
  margin: 0;
}
.sidebar-logo p {
  color: rgba(255,255,255,0.6) !important;
  font-size: 0.75rem;
  margin: 4px 0 0 0;
}

/* Hide default streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─── Init ────────────────────────────────────────────────────────────────────
db.init_db()

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

def models_trained():
    return os.path.exists(os.path.join(MODELS_DIR, "best_model.pkl"))

def auto_train():
    """Train models if not already trained."""
    if not models_trained():
        with st.spinner("🤖 First-time setup: Training AI models (this takes ~30 seconds)..."):
            from train_model import train_all_models
            train_all_models()
        st.success("✅ Models trained successfully!")
        st.rerun()


# ─── Session State ────────────────────────────────────────────────────────────
if "user" not in st.session_state:
    st.session_state.user = None
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "last_app_data" not in st.session_state:
    st.session_state.last_app_data = None


# ─── Sidebar Navigation ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <h2>🏦 LoanAI</h2>
        <p>Intelligent Loan Platform</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.user:
        role = st.session_state.user.get("role", "user")
        st.markdown(f"<div style='padding:10px;background:rgba(232,184,109,0.15);border-radius:10px;margin-bottom:16px;'>"
                    f"<b style='color:#E8B86D'>👤 {st.session_state.user['full_name']}</b><br>"
                    f"<small style='opacity:0.7'>{role.title()} Account</small></div>",
                    unsafe_allow_html=True)

    pages_user = [
        "🏠 Home",
        "🔐 Login / Register",
        "📋 Loan Application",
        "🧮 EMI Calculator",
        "✅ Eligibility Checker",
        "🤖 Default Prediction",
        "🔍 XAI Dashboard",
        "📊 Analytics",
        "📁 My Applications",
        "💬 AI Assistant",
        "📥 Reports",
    ]
    pages_admin = pages_user + ["⚙️ Admin Panel"]

    pages = pages_admin if (st.session_state.user and
                            st.session_state.user.get("role") == "admin") else pages_user

    page = st.radio("Navigation", pages, index=pages.index(st.session_state.page)
                    if st.session_state.page in pages else 0, label_visibility="collapsed")
    st.session_state.page = page

    st.markdown("---")
    if st.session_state.user:
        if st.button("🚪 Logout"):
            st.session_state.user = None
            st.session_state.page = "🏠 Home"
            st.rerun()
    else:
        st.info("Login to access all features")

    if not models_trained():
        if st.button("🚀 Train AI Models"):
            auto_train()
    else:
        st.success("✅ AI Models Ready")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div class="hero-card">
        <div class="hero-title">🏦 LoanAI — <span class="hero-gold">Intelligent</span> Loan Platform</div>
        <div class="hero-subtitle">Advanced Machine Learning • Explainable AI • Real-time Analytics</div>
        <div style="margin-top:20px;font-size:0.9rem;opacity:0.75;z-index:1;position:relative;">
            B.Tech Final Year Major Project | IEEE Paper Implementation | AI-Powered Banking System
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("7", "Loan Types", "💳"),
        ("6", "Bank Partners", "🏛️"),
        ("4", "ML Models", "🤖"),
        ("SHAP", "Explainable AI", "🔍"),
    ]
    for col, (val, label, icon) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:2rem;margin-bottom:4px">{icon}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 🎯 Platform Features")
        features = [
            ("📋", "Smart Loan Application", "Apply for 7 types of loans with intelligent form validation"),
            ("🧮", "EMI Calculator", "Real-time EMI with amortization schedule and visualizations"),
            ("✅", "Eligibility Checker", "Instant eligibility based on 10+ financial parameters"),
            ("🤖", "Default Prediction", "XGBoost + Random Forest ensemble for accurate risk scoring"),
            ("🔍", "Explainable AI", "SHAP force plots, waterfall charts to explain every prediction"),
            ("📊", "Analytics Dashboard", "Interactive Plotly dashboards with loan portfolio insights"),
        ]
        for icon, title, desc in features:
            st.markdown(f"""
            <div style="display:flex;gap:14px;margin-bottom:14px;padding:14px;
                        background:white;border-radius:12px;
                        box-shadow:0 2px 12px rgba(15,52,96,0.06);">
                <span style="font-size:1.6rem;min-width:36px">{icon}</span>
                <div>
                    <div style="font-weight:600;color:#0F3460">{title}</div>
                    <div style="font-size:0.85rem;color:#718096">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_b:
        st.markdown("### 🏛️ Supported Banks & Loan Types")
        bank_data = {"Bank": list(BANK_RATES.keys())}
        for lt in ["Personal Loan", "Home Loan", "Education Loan"]:
            bank_data[lt] = [f"{BANK_RATES[b][lt]}%" for b in BANK_RATES]
        df_banks = pd.DataFrame(bank_data)
        st.dataframe(df_banks, use_container_width=True, hide_index=True)

        st.markdown("### 📈 CIBIL Score Guide")
        cibil_df = pd.DataFrame({
            "Range": ["300–550", "550–650", "650–750", "750–900"],
            "Category": ["Poor", "Bad", "Average", "Good"],
            "Approval Chance": ["Very Low", "Low", "Moderate", "High"],
        })
        st.dataframe(cibil_df, use_container_width=True, hide_index=True)

    # Tech stack
    st.markdown("---")
    st.markdown("### 🛠️ Technology Stack")
    techs = ["Python", "Streamlit", "XGBoost", "Random Forest", "SHAP", "Plotly", "SQLite", "Scikit-learn"]
    cols = st.columns(len(techs))
    colors = ["#0F3460","#1A4A80","#2E6DA4","#E8B86D","#C9982A","#00B894","#636E72","#74B9FF"]
    for col, tech, color in zip(cols, techs, colors):
        with col:
            st.markdown(f"""
            <div style="background:{color};color:white;text-align:center;padding:10px 6px;
                        border-radius:10px;font-weight:600;font-size:0.8rem;">{tech}</div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: LOGIN / REGISTER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔐 Login / Register":
    st.markdown("## 🔐 User Authentication")
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        with st.form("login_form"):
            st.markdown("### Welcome Back")
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submitted = st.form_submit_button("Login →", use_container_width=True)
            if submitted:
                user = db.authenticate_user(username, password)
                if user:
                    st.session_state.user = user
                    st.success(f"Welcome back, {user['full_name']}!")
                    st.rerun()
                else:
                    st.error("Invalid credentials. Try admin / admin123")
        st.markdown("""
        <div class="info-box">
            <b>Demo Credentials:</b> Username: <code>admin</code> | Password: <code>admin123</code>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        with st.form("reg_form"):
            st.markdown("### Create Account")
            c1, c2 = st.columns(2)
            with c1:
                reg_fname = st.text_input("Full Name")
                reg_user = st.text_input("Username")
                reg_pass = st.text_input("Password", type="password")
            with c2:
                reg_email = st.text_input("Email")
                reg_phone = st.text_input("Phone")
                reg_pass2 = st.text_input("Confirm Password", type="password")
            reg_submit = st.form_submit_button("Create Account →", use_container_width=True)
            if reg_submit:
                if reg_pass != reg_pass2:
                    st.error("Passwords do not match!")
                elif not all([reg_fname, reg_user, reg_pass, reg_email]):
                    st.error("Please fill all required fields.")
                else:
                    ok, msg = db.register_user(reg_user, reg_pass, reg_fname, reg_email, reg_phone)
                    if ok:
                        st.success(msg + " Please login.")
                    else:
                        st.error(msg)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EMI CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧮 EMI Calculator":
    st.markdown("## 🧮 EMI Calculator")
    st.markdown("Instant EMI computation with amortization schedule")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Loan Parameters")
        bank = st.selectbox("Select Bank", BANKS)
        loan_type = st.selectbox("Loan Type", LOAN_TYPES)
        auto_rate = get_rate(bank, loan_type)

        principal = st.number_input("Loan Amount (₹)", min_value=10000.0,
                                     max_value=10000000.0, value=500000.0, step=10000.0)
        rate_input = st.number_input("Interest Rate (% per annum)",
                                      min_value=1.0, max_value=30.0, value=float(auto_rate), step=0.05)
        tenure_yrs = st.slider("Loan Tenure (Years)", 1, 30, 5)
        tenure_months = tenure_yrs * 12

        st.markdown(f"""
        <div class="info-box">
            <b>{bank}</b> offers <b>{auto_rate}% p.a.</b> for {loan_type}
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    result = calculate_emi(principal, rate_input, tenure_months)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### EMI Breakdown")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Monthly EMI", f"₹{result['emi']:,.0f}")
        mc2.metric("Total Interest", f"₹{result['total_interest']:,.0f}")
        mc3.metric("Total Payment", f"₹{result['total_payment']:,.0f}")

        fig_pie = go.Figure(data=[go.Pie(
            labels=["Principal", "Total Interest"],
            values=[result["principal"], result["total_interest"]],
            hole=0.55,
            marker_colors=["#0F3460", "#E8B86D"],
            textfont_size=13,
        )])
        fig_pie.update_layout(
            title_text="Principal vs Interest Split",
            showlegend=True, height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans"),
            margin=dict(t=40, b=0, l=0, r=0),
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Amortization chart
    schedule = result["schedule"]
    df_sched = pd.DataFrame(schedule)

    st.markdown("#### 📈 Loan Repayment Schedule")
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(name="Principal", x=df_sched["month"],
                              y=df_sched["principal_component"], marker_color="#0F3460"))
    fig_bar.add_trace(go.Bar(name="Interest", x=df_sched["month"],
                              y=df_sched["interest_component"], marker_color="#E8B86D"))
    fig_bar.add_trace(go.Scatter(name="Balance", x=df_sched["month"],
                                  y=df_sched["balance"], mode="lines",
                                  line=dict(color="#00B894", width=2), yaxis="y2"))
    fig_bar.update_layout(
        barmode="stack", xaxis_title="Month", yaxis_title="Amount (₹)",
        height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Outstanding Balance (₹)"),
        font=dict(family="DM Sans"), legend=dict(orientation="h", y=1.05),
        xaxis=dict(gridcolor="#E2E8F0"), yaxis=dict(gridcolor="#E2E8F0"),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    with st.expander("📋 Full Amortization Table"):
        st.dataframe(df_sched, use_container_width=True, hide_index=True)

    # Bank comparison
    st.markdown("#### 🏛️ Bank-wise EMI Comparison")
    comp_data = []
    for b in BANKS:
        r = get_rate(b, loan_type)
        res = calculate_emi(principal, r, tenure_months)
        comp_data.append({"Bank": b, "Rate (%)": r, "Monthly EMI (₹)": res["emi"],
                          "Total Interest (₹)": res["total_interest"]})
    df_comp = pd.DataFrame(comp_data)
    fig_comp = px.bar(df_comp, x="Bank", y="Monthly EMI (₹)", color="Rate (%)",
                      title=f"EMI Comparison — {loan_type} | ₹{principal:,.0f} | {tenure_yrs}yr",
                      color_continuous_scale="Blues", text_auto=True)
    fig_comp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(family="DM Sans"), height=380,
                            xaxis=dict(gridcolor="#E2E8F0"), yaxis=dict(gridcolor="#E2E8F0"))
    st.plotly_chart(fig_comp, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ELIGIBILITY CHECKER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "✅ Eligibility Checker":
    st.markdown("## ✅ Loan Eligibility Checker")

    col1, col2 = st.columns(2)
    with col1:
        monthly_income = st.number_input("Monthly Income (₹)", 0.0, 1e7, 50000.0, 1000.0)
        cibil = st.slider("CIBIL Score", 300, 900, 700)
        loan_amount = st.number_input("Loan Amount (₹)", 10000.0, 1e7, 500000.0, 10000.0)
        age = st.slider("Age", 18, 70, 30)

    with col2:
        existing_emi = st.number_input("Existing EMI (₹/month)", 0.0, 1e6, 0.0, 500.0)
        existing_loans = st.number_input("Number of Existing Loans", 0, 10, 0)
        occupation = st.selectbox("Occupation", ["Salaried", "Business", "Self-employed",
                                                  "Government Employee", "Student"])
        dependents = st.slider("Dependents", 0, 6, 1)

    if st.button("Check Eligibility →", use_container_width=True):
        # Rule-based + scoring
        dti = (existing_emi / monthly_income) if monthly_income > 0 else 1
        lti = (loan_amount / (monthly_income * 12)) if monthly_income > 0 else 99
        max_emi = monthly_income * 0.50 - existing_emi
        new_emi = calculate_emi(loan_amount, 10.5, 60)["emi"]

        score = 0
        reasons_ok = []
        reasons_fail = []

        # CIBIL check
        if cibil >= 750: score += 30; reasons_ok.append("Excellent CIBIL score (750+)")
        elif cibil >= 650: score += 20; reasons_ok.append("Average CIBIL score")
        elif cibil >= 550: score += 5; reasons_fail.append("Low CIBIL score (550–650)")
        else: score -= 10; reasons_fail.append("Poor CIBIL score (<550)")

        # Income check
        if monthly_income >= 50000: score += 25; reasons_ok.append("Good monthly income")
        elif monthly_income >= 25000: score += 15; reasons_ok.append("Adequate monthly income")
        else: score += 5; reasons_fail.append("Low monthly income")

        # DTI check
        if dti < 0.3: score += 20; reasons_ok.append("Low debt-to-income ratio")
        elif dti < 0.5: score += 10
        else: score -= 10; reasons_fail.append("High debt-to-income ratio")

        # Age
        if 25 <= age <= 55: score += 10; reasons_ok.append("Ideal age bracket")
        elif age < 21: score -= 5; reasons_fail.append("Too young for most loans")

        # Occupation
        occ_scores = {"Government Employee": 15, "Salaried": 12, "Business": 10,
                      "Self-employed": 8, "Student": 5}
        score += occ_scores.get(occupation, 8)

        # EMI check
        if new_emi <= max_emi: score += 10; reasons_ok.append("EMI within income limits")
        else: score -= 15; reasons_fail.append("EMI exceeds 50% of income")

        # Existing loans
        if existing_loans == 0: score += 5; reasons_ok.append("No existing loans")
        elif existing_loans > 3: reasons_fail.append("Multiple existing loans")

        score = max(0, min(100, score))
        eligible = score >= 50
        prob = score / 100

        # Display
        st.markdown("---")
        col_r1, col_r2 = st.columns([1, 1])
        with col_r1:
            if eligible:
                st.markdown('<div class="approved-banner">✅ ELIGIBLE FOR LOAN</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="rejected-banner">❌ NOT ELIGIBLE</div>',
                            unsafe_allow_html=True)

            # Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=score,
                number={"suffix": "/100", "font": {"size": 36, "family": "Playfair Display"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": "#0F3460", "thickness": 0.3},
                    "steps": [
                        {"range": [0, 33], "color": "#FED7D7"},
                        {"range": [33, 66], "color": "#FEF3C7"},
                        {"range": [66, 100], "color": "#D1FAE5"},
                    ],
                    "threshold": {"value": 50, "line": {"color": "#E8B86D", "width": 4}},
                },
                title={"text": "Eligibility Score", "font": {"size": 18}},
            ))
            fig_gauge.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)",
                                    font=dict(family="DM Sans"))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_r2:
            st.metric("Approval Probability", f"{prob*100:.1f}%")
            st.metric("Debt-to-Income Ratio", f"{dti*100:.1f}%")
            st.metric("Loan-to-Income Ratio", f"{lti:.2f}x")
            st.metric("Suggested Max EMI", f"₹{max_emi:,.0f}/month")

            if reasons_ok:
                st.markdown("**✅ Positive Factors:**")
                for r in reasons_ok:
                    st.markdown(f"&nbsp;&nbsp;• {r}")
            if reasons_fail:
                st.markdown("**⚠️ Risk Factors:**")
                for r in reasons_fail:
                    st.markdown(f"&nbsp;&nbsp;• {r}")

        # CIBIL analysis
        st.markdown("---")
        st.markdown("#### 📊 CIBIL Score Analysis")
        if cibil < 550:
            cibil_cat, cibil_class, cibil_msg = "Poor", "cibil-poor", "Very difficult to get loan approval."
        elif cibil < 650:
            cibil_cat, cibil_class, cibil_msg = "Bad", "cibil-bad", "Approval possible with high interest."
        elif cibil < 750:
            cibil_cat, cibil_class, cibil_msg = "Average", "cibil-avg", "Moderate approval chances."
        else:
            cibil_cat, cibil_class, cibil_msg = "Good", "cibil-good", "High chances of loan approval."

        st.markdown(f"""
        <div class="section-card">
            <b>CIBIL Score:</b> <span class="{cibil_class}">{cibil} — {cibil_cat}</span><br>
            <span style="color:#718096">{cibil_msg}</span>
        </div>
        """, unsafe_allow_html=True)

        fig_cibil = go.Figure(go.Indicator(
            mode="gauge+number",
            value=cibil,
            gauge={
                "axis": {"range": [300, 900]},
                "bar": {"color": "#0F3460"},
                "steps": [
                    {"range": [300, 550], "color": "#FCA5A5"},
                    {"range": [550, 650], "color": "#FCD34D"},
                    {"range": [650, 750], "color": "#93C5FD"},
                    {"range": [750, 900], "color": "#6EE7B7"},
                ],
            },
            title={"text": "CIBIL Score Meter"},
        ))
        fig_cibil.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)", font=dict(family="DM Sans"))
        st.plotly_chart(fig_cibil, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: LOAN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Loan Application":
    st.markdown("## 📋 Loan Application Form")

    auto_train()
    if not models_trained():
        st.warning("Please train the models first.")
        st.stop()

    with st.form("loan_form", clear_on_submit=False):
        # Personal
        st.markdown("#### 👤 Personal Details")
        c1, c2, c3 = st.columns(3)
        with c1:
            full_name = st.text_input("Full Name *", placeholder="Your full name")
            age = st.number_input("Age *", 18, 75, 30)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        with c2:
            marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
            dependents = st.number_input("Dependents", 0, 10, 0)
            phone = st.text_input("Phone *", placeholder="10-digit mobile number")
        with c3:
            email = st.text_input("Email *", placeholder="email@example.com")
            education = st.selectbox("Highest Education",
                                      ["High School", "Diploma", "Graduate", "Post Graduate", "PhD"])
            occupation = st.selectbox("Occupation",
                                       ["Salaried", "Business", "Self-employed",
                                        "Government Employee", "Student"])

        st.markdown("#### 💼 Employment & Financial Details")
        c4, c5, c6 = st.columns(3)
        with c4:
            work_exp = st.number_input("Work Experience (years)", 0.0, 50.0, 3.0, 0.5)
            company = st.text_input("Company / Business Name", placeholder="Optional")
            monthly_income = st.number_input("Monthly Income (₹) *", 0.0, 5000000.0, 50000.0, 1000.0)
        with c5:
            annual_income = monthly_income * 12
            st.metric("Annual Income (₹)", f"₹{annual_income:,.0f}")
            existing_loans = st.number_input("No. of Existing Loans", 0, 10, 0)
            existing_emi = st.number_input("Existing EMI (₹/month)", 0.0, 500000.0, 0.0, 500.0)
        with c6:
            savings = st.number_input("Savings (₹)", 0.0, 10000000.0, 100000.0, 5000.0)
            assets = st.number_input("Total Assets (₹)", 0.0, 50000000.0, 500000.0, 10000.0)
            cibil = st.slider("CIBIL Score *", 300, 900, 700)

        st.markdown("#### 🏦 Bank & Loan Details")
        c7, c8, c9 = st.columns(3)
        with c7:
            bank = st.selectbox("Bank *", BANKS)
            loan_type = st.selectbox("Loan Type *", LOAN_TYPES)
            auto_rate = get_rate(bank, loan_type)
        with c8:
            loan_amount = st.number_input("Loan Amount (₹) *", 10000.0, 10000000.0, 500000.0, 10000.0)
            rate = st.number_input("Interest Rate (% p.a.)", 1.0, 30.0, float(auto_rate), 0.05)
            tenure = st.slider("Loan Tenure (months)", 6, 360, 60)
        with c9:
            purpose = st.text_area("Purpose of Loan", placeholder="Brief purpose...", height=80)
            emi_result = calculate_emi(loan_amount, rate, tenure)
            st.metric("Computed EMI", f"₹{emi_result['emi']:,.0f}/mo")
            st.metric("Total Payment", f"₹{emi_result['total_payment']:,.0f}")

        submit_app = st.form_submit_button("🚀 Submit Application & Get AI Prediction",
                                            use_container_width=True)

    if submit_app:
        if not all([full_name, phone, email]):
            st.error("Please fill all required fields.")
        else:
            with st.spinner("Running AI prediction and SHAP analysis..."):
                # Prepare features
                enc_maps = {"gender": {"Male": 0, "Female": 1, "Other": 2},
                            "marital": {"Single": 0, "Married": 1, "Divorced": 2, "Widowed": 3},
                            "education": {"High School": 0, "Diploma": 1, "Graduate": 2,
                                          "Post Graduate": 3, "PhD": 4},
                            "occupation": {"Salaried": 0, "Business": 1, "Self-employed": 2,
                                           "Government Employee": 3, "Student": 4},
                            "loan_type": {lt: i for i, lt in enumerate(LOAN_TYPES)}}

                dti = (existing_emi / monthly_income) if monthly_income > 0 else 0
                lti = (loan_amount / (annual_income)) if annual_income > 0 else 0

                features_dict = {
                    "age": age, "dependents": dependents, "work_experience": work_exp,
                    "monthly_income": monthly_income, "annual_income": annual_income,
                    "existing_loans": existing_loans, "existing_emi": existing_emi,
                    "savings": savings, "assets": assets, "cibil_score": cibil,
                    "loan_amount": loan_amount, "interest_rate": rate, "loan_tenure": tenure,
                    "dti_ratio": dti, "loan_to_income_ratio": lti,
                    "gender_enc": enc_maps["gender"].get(gender, 0),
                    "marital_enc": enc_maps["marital"].get(marital, 0),
                    "education_enc": enc_maps["education"].get(education, 2),
                    "occupation_enc": enc_maps["occupation"].get(occupation, 0),
                    "loan_type_enc": enc_maps["loan_type"].get(loan_type, 0),
                }

                from shap_explainer import predict_and_explain
                pred_result = predict_and_explain(features_dict)

                # Eligibility
                elig_score = 0
                if cibil >= 750: elig_score += 30
                elif cibil >= 650: elig_score += 20
                elif cibil >= 550: elig_score += 5
                if monthly_income >= 50000: elig_score += 25
                elif monthly_income >= 25000: elig_score += 15
                if dti < 0.3: elig_score += 20
                elif dti < 0.5: elig_score += 10
                if 25 <= age <= 55: elig_score += 10
                elig_score += {"Government Employee": 15, "Salaried": 12, "Business": 10,
                               "Self-employed": 8, "Student": 5}.get(occupation, 8)
                eligible = elig_score >= 50

                app_data = {
                    "user_id": st.session_state.user["id"] if st.session_state.user else None,
                    "full_name": full_name, "age": age, "gender": gender,
                    "marital_status": marital, "dependents": dependents,
                    "phone": phone, "email": email, "education": education,
                    "occupation": occupation, "work_experience": work_exp,
                    "company_name": company, "monthly_income": monthly_income,
                    "annual_income": annual_income, "existing_loans": existing_loans,
                    "existing_emi": existing_emi, "savings": savings, "assets": assets,
                    "bank_name": bank, "cibil_score": cibil, "loan_type": loan_type,
                    "loan_amount": loan_amount, "interest_rate": rate, "loan_tenure": tenure,
                    "purpose": purpose, "emi": emi_result["emi"],
                    "total_interest": emi_result["total_interest"],
                    "total_payment": emi_result["total_payment"],
                    "eligibility": "Eligible" if eligible else "Not Eligible",
                    "approval_probability": pred_result.get("probability", 0),
                    "risk_level": pred_result.get("risk_level", "N/A"),
                    "prediction": "Default Risk" if pred_result.get("prediction") == 1 else "No Default",
                    "shap_values": pred_result.get("shap_values", {}),
                    **pred_result,
                }
                app_id = db.save_application(app_data)
                st.session_state.last_prediction = pred_result
                st.session_state.last_app_data = app_data

            st.success(f"✅ Application #{app_id} submitted successfully!")

            # Result display
            pred = pred_result.get("prediction", 0)
            prob = pred_result.get("probability", 0)
            risk = pred_result.get("risk_level", "N/A")

            if pred == 0:
                st.markdown('<div class="approved-banner">✅ LOW DEFAULT RISK — LIKELY TO APPROVE</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="rejected-banner">⚠️ HIGH DEFAULT RISK — REVIEW REQUIRED</div>',
                            unsafe_allow_html=True)

            c_r1, c_r2, c_r3, c_r4 = st.columns(4)
            c_r1.metric("Default Probability", f"{prob*100:.1f}%")
            c_r2.metric("Risk Level", risk)
            c_r3.metric("EMI", f"₹{emi_result['emi']:,.0f}/mo")
            c_r4.metric("Eligibility", "✅ Eligible" if eligible else "❌ Not Eligible")

            st.info("👈 Go to **🔍 XAI Dashboard** to see detailed SHAP explanations")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DEFAULT PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Default Prediction":
    st.markdown("## 🤖 Loan Default Prediction")
    st.markdown("Quick prediction without full application form")

    auto_train()
    if not models_trained():
        st.warning("Please train models first."); st.stop()

    col1, col2 = st.columns(2)
    with col1:
        p_income = st.number_input("Monthly Income (₹)", 5000.0, 1000000.0, 50000.0, 1000.0)
        p_cibil = st.slider("CIBIL Score", 300, 900, 700)
        p_loan = st.number_input("Loan Amount (₹)", 10000.0, 5000000.0, 500000.0, 10000.0)
        p_emi = st.number_input("Existing EMI (₹)", 0.0, 100000.0, 0.0, 500.0)
        p_savings = st.number_input("Savings (₹)", 0.0, 5000000.0, 100000.0)
    with col2:
        p_age = st.slider("Age", 18, 65, 30)
        p_dep = st.slider("Dependents", 0, 6, 1)
        p_loans = st.number_input("Existing Loans", 0, 10, 0)
        p_exp = st.number_input("Work Experience (yrs)", 0.0, 40.0, 3.0, 0.5)
        p_assets = st.number_input("Total Assets (₹)", 0.0, 10000000.0, 500000.0)
        p_occ = st.selectbox("Occupation", ["Salaried", "Business", "Self-employed",
                                             "Government Employee", "Student"])

    if st.button("🔮 Predict Default Risk", use_container_width=True):
        annual = p_income * 12
        dti = p_emi / p_income if p_income > 0 else 0
        lti = p_loan / annual if annual > 0 else 0
        occ_enc = {"Salaried": 0, "Business": 1, "Self-employed": 2,
                   "Government Employee": 3, "Student": 4}

        features_dict = {
            "age": p_age, "dependents": p_dep, "work_experience": p_exp,
            "monthly_income": p_income, "annual_income": annual,
            "existing_loans": p_loans, "existing_emi": p_emi,
            "savings": p_savings, "assets": p_assets, "cibil_score": p_cibil,
            "loan_amount": p_loan, "interest_rate": 10.5, "loan_tenure": 60,
            "dti_ratio": dti, "loan_to_income_ratio": lti,
            "gender_enc": 0, "marital_enc": 1, "education_enc": 2,
            "occupation_enc": occ_enc.get(p_occ, 0), "loan_type_enc": 0,
        }
        from shap_explainer import predict_and_explain
        result = predict_and_explain(features_dict)
        st.session_state.last_prediction = result

        prob = result.get("probability", 0)
        risk = result.get("risk_level", "")
        pred = result.get("prediction", 0)

        st.markdown("---")
        if pred == 0:
            st.markdown('<div class="approved-banner">✅ LOW DEFAULT RISK</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="rejected-banner">⚠️ HIGH DEFAULT RISK</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Default Probability", f"{prob*100:.1f}%")
        c2.metric("Risk Level", risk)
        c3.metric("Approval Chance", f"{(1-prob)*100:.1f}%")

        # Feature bar chart
        sv = result.get("shap_values", {})
        from shap_explainer import FEATURE_LABELS
        labels = [FEATURE_LABELS.get(k, k) for k in sv.keys()]
        values = list(sv.values())
        colors = ["#E17055" if v > 0 else "#00B894" for v in values]

        sorted_pairs = sorted(zip(labels, values, colors), key=lambda x: abs(x[1]), reverse=True)[:12]
        s_labels, s_vals, s_colors = zip(*sorted_pairs)

        fig = go.Figure(go.Bar(x=list(s_vals), y=list(s_labels), orientation="h",
                                marker_color=list(s_colors)))
        fig.update_layout(
            title="Top Feature Contributions (SHAP)",
            xaxis_title="SHAP Value (impact on risk)",
            height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans"), xaxis=dict(gridcolor="#E2E8F0"),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info("👈 Navigate to **🔍 XAI Dashboard** for full SHAP analysis")

        # Model comparison
        results_path = os.path.join(MODELS_DIR, "model_results.pkl")
        if os.path.exists(results_path):
            model_results = joblib.load(results_path)
            st.markdown("#### 📊 Model Performance Comparison")
            df_mr = pd.DataFrame(model_results).T.reset_index()
            df_mr.columns = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
            fig_mr = px.bar(df_mr.melt(id_vars="Model"), x="Model", y="value",
                            color="variable", barmode="group",
                            title="ML Model Performance Metrics",
                            color_discrete_sequence=px.colors.qualitative.Set2)
            fig_mr.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)", font=dict(family="DM Sans"),
                                  yaxis=dict(range=[0, 1.1]))
            st.plotly_chart(fig_mr, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: XAI DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 XAI Dashboard":
    st.markdown("## 🔍 Explainable AI Dashboard")
    st.markdown("SHAP-powered explanations for every prediction")

    if not models_trained():
        auto_train()
        if not models_trained():
            st.warning("Please train models first."); st.stop()

    result = st.session_state.last_prediction
    if not result:
        st.info("No prediction found. Please run a prediction first from **Loan Application** or **Default Prediction** pages.")
        from shap_explainer import get_feature_importance_data
        fi = get_feature_importance_data()
        if fi:
            st.markdown("### 📊 Global Feature Importance")
            fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True)[:15])
            fig_fi = px.bar(x=list(fi_sorted.values()), y=list(fi_sorted.keys()),
                            orientation="h", title="Feature Importance (Random Forest / XGBoost)",
                            color=list(fi_sorted.values()), color_continuous_scale="Blues")
            fig_fi.update_layout(height=480, paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)", font=dict(family="DM Sans"),
                                  xaxis=dict(gridcolor="#E2E8F0"))
            st.plotly_chart(fig_fi, use_container_width=True)
        st.stop()

    from shap_explainer import FEATURE_LABELS, get_feature_importance_data

    prob = result.get("probability", 0)
    risk = result.get("risk_level", "N/A")
    pred = result.get("prediction", 0)
    sv = result.get("shap_values", {})
    base_val = result.get("expected_value", 0.5)

    # Summary
    col1, col2, col3 = st.columns(3)
    col1.metric("Default Probability", f"{prob*100:.1f}%")
    col2.metric("Base Rate (Prior)", f"{base_val*100:.1f}%")
    col3.metric("Risk Level", risk)

    st.markdown("---")

    # Waterfall chart
    st.markdown("### 🌊 SHAP Waterfall Plot")
    st.markdown("Shows how each feature pushes the prediction from the base value.")
    labels = [FEATURE_LABELS.get(k, k) for k in sv.keys()]
    values = list(sv.values())
    sorted_pairs = sorted(zip(labels, values), key=lambda x: abs(x[1]), reverse=True)[:12]
    wf_labels, wf_values = zip(*sorted_pairs)

    running = base_val
    measures = []
    for v in wf_values:
        measures.append("increasing" if v > 0 else "decreasing")
    measures.append("total")
    wf_x = list(wf_labels) + ["Final Score"]
    wf_y = list(wf_values) + [running + sum(wf_values)]

    fig_wf = go.Figure(go.Waterfall(
        orientation="v",
        measure=measures,
        x=wf_x,
        y=wf_y,
        base=base_val,
        connector={"line": {"color": "#718096"}},
        increasing={"marker": {"color": "#E17055"}},
        decreasing={"marker": {"color": "#00B894"}},
        totals={"marker": {"color": "#0F3460"}},
        text=[f"{v:+.3f}" for v in wf_values] + [f"{prob:.3f}"],
        textposition="outside",
    ))
    fig_wf.update_layout(
        title="SHAP Waterfall — Feature Contribution to Default Risk",
        yaxis_title="Probability Impact", height=480,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans"), xaxis=dict(gridcolor="#E2E8F0"),
        yaxis=dict(gridcolor="#E2E8F0"),
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    # Force plot (horizontal bar)
    st.markdown("### ⚡ SHAP Force Plot")
    st.markdown("Feature impacts relative to base prediction value.")
    pos_feats = [(FEATURE_LABELS.get(k, k), v) for k, v in sv.items() if v > 0]
    neg_feats = [(FEATURE_LABELS.get(k, k), v) for k, v in sv.items() if v < 0]
    pos_feats.sort(key=lambda x: x[1], reverse=True)
    neg_feats.sort(key=lambda x: x[1])

    fig_force = go.Figure()
    for label, val in pos_feats[:8]:
        fig_force.add_trace(go.Bar(name=label, x=[val], y=["Risk Factors"],
                                    orientation="h", marker_color="#E17055",
                                    text=[f"{label}: {val:+.3f}"],
                                    hovertemplate="%{text}<extra></extra>"))
    for label, val in neg_feats[:8]:
        fig_force.add_trace(go.Bar(name=label, x=[val], y=["Protective Factors"],
                                    orientation="h", marker_color="#00B894",
                                    text=[f"{label}: {val:+.3f}"],
                                    hovertemplate="%{text}<extra></extra>"))
    fig_force.update_layout(
        barmode="stack", height=250, showlegend=False,
        title="Force Plot: Risk vs Protective Factors",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans"), xaxis=dict(gridcolor="#E2E8F0"),
        annotations=[dict(x=0, y=1.12, text=f"Base: {base_val:.3f}", showarrow=False,
                          font=dict(color="#0F3460", size=12))]
    )
    st.plotly_chart(fig_force, use_container_width=True)

    # SHAP Summary (all features)
    st.markdown("### 📊 SHAP Summary — All Features")
    all_labels = [FEATURE_LABELS.get(k, k) for k in sv.keys()]
    all_values = list(sv.values())
    fig_summary = go.Figure(go.Bar(
        x=all_values, y=all_labels, orientation="h",
        marker=dict(color=all_values, colorscale="RdYlGn_r",
                    colorbar=dict(title="Impact")),
    ))
    fig_summary.update_layout(
        title="All Feature SHAP Values",
        xaxis_title="SHAP Value", height=550,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans"), xaxis=dict(gridcolor="#E2E8F0"),
    )
    st.plotly_chart(fig_summary, use_container_width=True)

    # Explanation text
    st.markdown("### 💡 Plain English Explanation")
    top_pos = result.get("top_positive_factors", [])
    top_neg = result.get("top_negative_factors", [])

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**🔴 Risk-Increasing Factors**")
        if top_pos:
            for label, val in top_pos[:5]:
                bar_pct = min(int(abs(val) * 500), 100)
                st.markdown(f"""
                <div style="margin-bottom:10px">
                    <div style="display:flex;justify-content:space-between">
                        <span style="font-weight:500">{label}</span>
                        <span style="color:#E17055;font-weight:600">{val:+.4f}</span>
                    </div>
                    <div style="background:#FEE2E2;border-radius:4px;height:6px;margin-top:4px">
                        <div style="background:#E17055;width:{bar_pct}%;height:6px;border-radius:4px"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No significant risk-increasing factors.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_e2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**🟢 Risk-Decreasing Factors**")
        if top_neg:
            for label, val in top_neg[:5]:
                bar_pct = min(int(abs(val) * 500), 100)
                st.markdown(f"""
                <div style="margin-bottom:10px">
                    <div style="display:flex;justify-content:space-between">
                        <span style="font-weight:500">{label}</span>
                        <span style="color:#00B894;font-weight:600">{val:+.4f}</span>
                    </div>
                    <div style="background:#D1FAE5;border-radius:4px;height:6px;margin-top:4px">
                        <div style="background:#00B894;width:{bar_pct}%;height:6px;border-radius:4px"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No significant protective factors found.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Global feature importance
    st.markdown("---")
    st.markdown("### 🌐 Global Feature Importance")
    fi = get_feature_importance_data()
    if fi:
        fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True)[:15])
        fig_gfi = px.bar(x=list(fi_sorted.values()), y=list(fi_sorted.keys()),
                          orientation="h", color=list(fi_sorted.values()),
                          color_continuous_scale="Blues",
                          title="Overall Feature Importance (trained model)")
        fig_gfi.update_layout(height=480, paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", font=dict(family="DM Sans"),
                               xaxis=dict(gridcolor="#E2E8F0"))
        st.plotly_chart(fig_gfi, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics":
    st.markdown("## 📊 Analytics Dashboard")

    data = db.get_analytics_data()
    if not data:
        st.info("No application data yet. Submit a loan application to see analytics.")
        st.stop()

    df = pd.DataFrame(data)

    # Summary metrics
    total = len(df)
    approved = len(df[df["status"] == "Approved"])
    pending = len(df[df["status"] == "Pending"])
    rejected = len(df[df["status"] == "Rejected"])
    avg_loan = df["loan_amount"].mean() if "loan_amount" in df else 0
    high_risk = len(df[df["risk_level"] == "High Risk"]) if "risk_level" in df else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Applications", total)
    c2.metric("Approved", approved)
    c3.metric("Pending", pending)
    c4.metric("Avg Loan Amount", f"₹{avg_loan:,.0f}")
    c5.metric("High Risk Cases", high_risk)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        # Status distribution
        status_counts = df["status"].value_counts()
        fig_status = px.pie(values=status_counts.values, names=status_counts.index,
                             title="Application Status Distribution",
                             color_discrete_sequence=["#00B894", "#E8B86D", "#E17055", "#74B9FF"])
        fig_status.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(family="DM Sans"), height=350)
        st.plotly_chart(fig_status, use_container_width=True)

    with col2:
        # Loan type distribution
        if "loan_type" in df:
            lt_counts = df["loan_type"].value_counts()
            fig_lt = px.bar(x=lt_counts.index, y=lt_counts.values,
                             title="Applications by Loan Type",
                             color=lt_counts.values, color_continuous_scale="Blues",
                             text_auto=True)
            fig_lt.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font=dict(family="DM Sans"), height=350,
                                  xaxis=dict(gridcolor="#E2E8F0"), yaxis=dict(gridcolor="#E2E8F0"))
            st.plotly_chart(fig_lt, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        if "risk_level" in df and df["risk_level"].notna().any():
            risk_counts = df["risk_level"].value_counts()
            fig_risk = px.bar(x=risk_counts.index, y=risk_counts.values,
                               title="Risk Level Distribution",
                               color=risk_counts.index,
                               color_discrete_map={"Low Risk": "#00B894",
                                                   "Medium Risk": "#FDCB6E",
                                                   "High Risk": "#E17055"})
            fig_risk.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    font=dict(family="DM Sans"), height=350,
                                    xaxis=dict(gridcolor="#E2E8F0"), yaxis=dict(gridcolor="#E2E8F0"))
            st.plotly_chart(fig_risk, use_container_width=True)

    with col4:
        if "cibil_score" in df and df["cibil_score"].notna().any():
            fig_cibil = px.histogram(df, x="cibil_score", nbins=20,
                                      title="CIBIL Score Distribution",
                                      color_discrete_sequence=["#0F3460"])
            fig_cibil.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                     font=dict(family="DM Sans"), height=350,
                                     xaxis=dict(gridcolor="#E2E8F0"), yaxis=dict(gridcolor="#E2E8F0"))
            st.plotly_chart(fig_cibil, use_container_width=True)

    if "monthly_income" in df and "loan_amount" in df:
        fig_scatter = px.scatter(df, x="monthly_income", y="loan_amount",
                                  color="risk_level" if "risk_level" in df else None,
                                  title="Income vs Loan Amount by Risk Level",
                                  color_discrete_map={"Low Risk": "#00B894",
                                                      "Medium Risk": "#FDCB6E",
                                                      "High Risk": "#E17055"},
                                  labels={"monthly_income": "Monthly Income (₹)",
                                          "loan_amount": "Loan Amount (₹)"})
        fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   font=dict(family="DM Sans"), height=420,
                                   xaxis=dict(gridcolor="#E2E8F0"), yaxis=dict(gridcolor="#E2E8F0"))
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Bank distribution
    if "bank_name" in df:
        bank_counts = df["bank_name"].value_counts()
        fig_bank = px.bar(x=bank_counts.index, y=bank_counts.values,
                           title="Applications by Bank",
                           color=bank_counts.values, color_continuous_scale="Teal")
        fig_bank.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(family="DM Sans"), height=350,
                                xaxis=dict(gridcolor="#E2E8F0"), yaxis=dict(gridcolor="#E2E8F0"))
        st.plotly_chart(fig_bank, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MY APPLICATIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📁 My Applications":
    st.markdown("## 📁 My Applications")
    if not st.session_state.user:
        st.warning("Please login to view your applications.")
        st.stop()

    apps = db.get_user_applications(st.session_state.user["id"])
    if not apps:
        st.info("No applications found. Submit a loan application to get started.")
        st.stop()

    for app in apps:
        risk = app.get("risk_level", "N/A")
        risk_cls = {"Low Risk": "risk-badge-low", "Medium Risk": "risk-badge-medium",
                    "High Risk": "risk-badge-high"}.get(risk, "risk-badge-medium")
        status_color = {"Approved": "#00B894", "Pending": "#FDCB6E",
                        "Rejected": "#E17055"}.get(app.get("status", ""), "#718096")

        with st.expander(f"Application #{app['id']} — {app.get('loan_type','N/A')} | "
                          f"₹{app.get('loan_amount',0):,.0f} | {app.get('applied_at','')[:10]}"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Loan Type", app.get("loan_type", "N/A"))
            c2.metric("Amount", f"₹{app.get('loan_amount',0):,.0f}")
            c3.metric("EMI", f"₹{app.get('emi',0):,.0f}/mo")
            c4, c5, c6 = st.columns(3)
            c4.metric("Bank", app.get("bank_name", "N/A"))
            c5.metric("CIBIL Score", app.get("cibil_score", "N/A"))
            c6.metric("Risk", risk)
            st.markdown(f"**Status:** <span style='color:{status_color};font-weight:700'>"
                        f"{app.get('status','N/A')}</span>", unsafe_allow_html=True)
            st.markdown(f"**Prediction:** {app.get('prediction','N/A')} | "
                        f"**Eligibility:** {app.get('eligibility','N/A')}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: AI ASSISTANT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💬 AI Assistant":
    st.markdown("## 💬 AI Loan Assistant")
    st.markdown("Ask me anything about loans, EMI, eligibility, and financial planning.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Simple rule-based + context chatbot
    def get_bot_response(user_msg: str) -> str:
        msg = user_msg.lower()
        if any(w in msg for w in ["emi", "equated monthly"]):
            return ("**EMI Formula:** EMI = [P × R × (1+R)^N] / [(1+R)^N – 1]\n\n"
                    "Where P = Principal, R = Monthly rate (annual/12/100), N = Tenure in months.\n\n"
                    "Use the 🧮 EMI Calculator page for instant computation!")
        elif any(w in msg for w in ["cibil", "credit score"]):
            return ("**CIBIL Score Guide:**\n"
                    "- 750–900: **Excellent** — High approval chances\n"
                    "- 650–750: **Good** — Moderate approval\n"
                    "- 550–650: **Fair** — Low approval, high interest\n"
                    "- 300–550: **Poor** — Very difficult to get loans\n\n"
                    "Improve CIBIL by paying EMIs on time, reducing credit utilization.")
        elif any(w in msg for w in ["eligib", "qualify"]):
            return ("**Loan eligibility depends on:**\n"
                    "- Income (higher = better)\n- CIBIL score (750+ recommended)\n"
                    "- Debt-to-Income ratio (keep below 40%)\n- Employment stability\n"
                    "- Age (21–55 ideal)\n\nUse the ✅ Eligibility Checker for detailed analysis!")
        elif any(w in msg for w in ["interest rate", "rate", "interest"]):
            rates_info = "\n".join([f"- **{b}** Personal Loan: {BANK_RATES[b]['Personal Loan']}% p.a."
                                     for b in list(BANK_RATES.keys())[:4]])
            return f"**Current Bank Interest Rates (Personal Loan):**\n{rates_info}\n\nRates vary by loan type and bank. Check the 🧮 EMI Calculator for full details."
        elif any(w in msg for w in ["shap", "explain", "xai", "explainable"]):
            return ("**About Explainable AI (SHAP):**\n"
                    "SHAP (SHapley Additive exPlanations) explains ML predictions by assigning each feature an importance value.\n\n"
                    "- **Positive SHAP** = increases default risk\n"
                    "- **Negative SHAP** = decreases default risk\n\n"
                    "See the 🔍 XAI Dashboard for SHAP waterfall, force, and summary plots!")
        elif any(w in msg for w in ["default", "risk", "predict"]):
            return ("**Loan Default Risk Factors:**\n"
                    "- Low CIBIL score\n- High debt-to-income ratio\n"
                    "- Low income relative to loan amount\n- Many existing loans\n"
                    "- Job instability\n\nOur XGBoost + Random Forest model predicts with ~85% accuracy!")
        elif any(w in msg for w in ["loan type", "home loan", "personal loan", "education"]):
            return ("**Loan Types Available:**\n"
                    "1. **Personal Loan** — 10–11% p.a., quick disbursal\n"
                    "2. **Home Loan** — 8.25–8.9% p.a., up to 30 years\n"
                    "3. **Education Loan** — 8–9.5% p.a., government support\n"
                    "4. **Vehicle Loan** — 8.9–9.6% p.a., secured\n"
                    "5. **Gold Loan** — 7.25–8% p.a., instant approval\n"
                    "6. **Business Loan** — 10.8–12.5% p.a.\n"
                    "7. **Agricultural Loan** — 6.8–8.5% p.a.")
        elif any(w in msg for w in ["hi", "hello", "hey"]):
            return ("Hello! 👋 I'm the **LoanAI Assistant**.\n\n"
                    "I can help you with:\n"
                    "- Loan EMI calculations\n- CIBIL score guidance\n"
                    "- Eligibility requirements\n- Interest rates\n"
                    "- Understanding AI predictions\n\nWhat would you like to know?")
        else:
            return ("I'm here to help with loan-related queries!\n\n"
                    "Try asking about: **EMI, CIBIL score, eligibility, interest rates, SHAP, loan types**\n\n"
                    "Or navigate to the relevant module for hands-on tools.")

    if prompt := st.chat_input("Ask about loans, EMI, eligibility..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        response = get_bot_response(prompt)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: REPORTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📥 Reports":
    st.markdown("## 📥 Report Download")

    if not st.session_state.last_app_data and not st.session_state.last_prediction:
        st.info("Submit a loan application to generate a report.")
        st.stop()

    app_data = st.session_state.last_app_data or {}
    pred = st.session_state.last_prediction or {}
    merged = {**app_data, **pred}

    st.markdown("### Report Preview")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Applicant:** {merged.get('full_name', 'N/A')}")
        st.markdown(f"**Loan Type:** {merged.get('loan_type', 'N/A')}")
        st.markdown(f"**Loan Amount:** ₹{merged.get('loan_amount', 0):,.2f}")
        st.markdown(f"**EMI:** ₹{merged.get('emi', 0):,.2f}/month")
    with c2:
        st.markdown(f"**Risk Level:** {merged.get('risk_level', 'N/A')}")
        st.markdown(f"**Default Probability:** {merged.get('probability', 0)*100:.1f}%")
        st.markdown(f"**Eligibility:** {merged.get('eligibility', 'N/A')}")
        st.markdown(f"**CIBIL Score:** {merged.get('cibil_score', 'N/A')}")

    try:
        from report_generator import generate_pdf_report
        pdf_bytes = generate_pdf_report(merged)
        if pdf_bytes:
            st.download_button(
                label="📄 Download Full PDF Report",
                data=pdf_bytes,
                file_name=f"LoanAI_Report_{merged.get('full_name','Applicant').replace(' ','_')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.info("Install fpdf2 for PDF reports: pip install fpdf2")
    except Exception as e:
        st.error(f"PDF generation error: {e}")

    # CSV download
    if app_data:
        import io
        df_report = pd.DataFrame([merged])
        csv_buf = io.StringIO()
        df_report.to_csv(csv_buf, index=False)
        st.download_button(
            label="📊 Download CSV Report",
            data=csv_buf.getvalue(),
            file_name="LoanAI_Report.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ADMIN PANEL
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Admin Panel":
    if not st.session_state.user or st.session_state.user.get("role") != "admin":
        st.error("Admin access required.")
        st.stop()

    st.markdown("## ⚙️ Admin Panel")

    tab_apps, tab_users, tab_models, tab_train = st.tabs(
        ["📋 Applications", "👥 Users", "📈 Model Stats", "🚀 Train Models"])

    with tab_apps:
        apps = db.get_all_applications()
        if apps:
            df_apps = pd.DataFrame(apps)
            display_cols = [c for c in ["id", "full_name", "loan_type", "loan_amount",
                                         "bank_name", "cibil_score", "risk_level",
                                         "prediction", "status", "applied_at"] if c in df_apps.columns]
            st.dataframe(df_apps[display_cols], use_container_width=True, hide_index=True)

            st.markdown("#### Update Application Status")
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                app_id = st.number_input("Application ID", 1, len(apps), 1)
            with col_s2:
                new_status = st.selectbox("New Status", ["Pending", "Approved", "Rejected"])
            with col_s3:
                if st.button("Update Status"):
                    db.update_application_status(int(app_id), new_status)
                    st.success(f"Application #{app_id} updated to {new_status}")
                    st.rerun()

            import io
            buf = io.StringIO()
            df_apps.to_csv(buf, index=False)
            st.download_button("📊 Export All Applications CSV",
                                buf.getvalue(), "all_applications.csv", "text/csv")
        else:
            st.info("No applications yet.")

    with tab_users:
        users = db.get_all_users()
        if users:
            st.dataframe(pd.DataFrame(users), use_container_width=True, hide_index=True)
        else:
            st.info("No users.")

    with tab_models:
        results_path = os.path.join(MODELS_DIR, "model_results.pkl")
        if os.path.exists(results_path):
            model_results = joblib.load(results_path)
            best_name = joblib.load(os.path.join(MODELS_DIR, "best_model_name.pkl"))
            st.success(f"🏆 Best Model: **{best_name}**")
            df_results = pd.DataFrame(model_results).T
            st.dataframe(df_results.style.highlight_max(color="#D1FAE5", axis=0),
                          use_container_width=True)

            fig_models = px.bar(df_results.reset_index().melt(id_vars="index"),
                                x="index", y="value", color="variable",
                                barmode="group", title="Model Comparison",
                                color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_models.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                      font=dict(family="DM Sans"), height=400)
            st.plotly_chart(fig_models, use_container_width=True)
        else:
            st.info("Models not trained yet.")

    with tab_train:
        st.markdown("### 🚀 Train / Retrain AI Models")
        st.markdown("""
        This will generate a synthetic loan dataset and train:
        - Random Forest Classifier
        - XGBoost Classifier
        - Logistic Regression
        - Decision Tree Classifier
        """)
        if st.button("🚀 Start Training", use_container_width=True):
            with st.spinner("Training models... (approx 30–60 seconds)"):
                from train_model import train_all_models
                results, best = train_all_models()
            st.success(f"✅ Training complete! Best model: **{best}**")
            st.balloons()
            df_r = pd.DataFrame(results).T
            st.dataframe(df_r, use_container_width=True)


