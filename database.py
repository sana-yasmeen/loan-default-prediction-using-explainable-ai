"""
Database Module - SQLite
Handles all database operations for the Loan Default Prediction System
"""

import sqlite3
import os
import hashlib
import json

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database", "loan_system.db")


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            email TEXT,
            phone TEXT,
            role TEXT DEFAULT 'user',
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS loan_applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            full_name TEXT, age INTEGER, gender TEXT,
            marital_status TEXT, dependents INTEGER,
            phone TEXT, email TEXT, education TEXT,
            occupation TEXT, work_experience REAL, company_name TEXT,
            monthly_income REAL, annual_income REAL,
            existing_loans INTEGER, existing_emi REAL,
            savings REAL, assets REAL, bank_name TEXT, cibil_score INTEGER,
            loan_type TEXT, loan_amount REAL, interest_rate REAL,
            loan_tenure INTEGER, purpose TEXT,
            emi REAL, total_interest REAL, total_payment REAL,
            eligibility TEXT, approval_probability REAL,
            risk_level TEXT, prediction TEXT, shap_values TEXT,
            status TEXT DEFAULT 'Pending',
            applied_at TEXT DEFAULT (datetime('now'))
        )
    """)
    admin_hash = hashlib.sha256("admin123".encode()).hexdigest()
    c.execute("INSERT OR IGNORE INTO users (username,password_hash,full_name,email,role) VALUES (?,?,?,?,?)",
              ("admin", admin_hash, "System Administrator", "admin@loanai.com", "admin"))
    conn.commit()
    conn.close()


def hash_password(p): return hashlib.sha256(p.encode()).hexdigest()


def register_user(username, password, full_name, email, phone):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username,password_hash,full_name,email,phone) VALUES (?,?,?,?,?)",
                  (username, hash_password(password), full_name, email, phone))
        conn.commit()
        return True, "Registration successful!"
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    finally:
        conn.close()


def authenticate_user(username, password):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password_hash=?",
              (username, hash_password(password)))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


def save_application(data):
    conn = get_connection()
    c = conn.cursor()
    shap_str = json.dumps(data.get("shap_values", {}))
    c.execute("""INSERT INTO loan_applications (
        user_id,full_name,age,gender,marital_status,dependents,phone,email,education,
        occupation,work_experience,company_name,monthly_income,annual_income,
        existing_loans,existing_emi,savings,assets,bank_name,cibil_score,
        loan_type,loan_amount,interest_rate,loan_tenure,purpose,
        emi,total_interest,total_payment,eligibility,approval_probability,
        risk_level,prediction,shap_values
    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
    (data.get("user_id"),data.get("full_name"),data.get("age"),data.get("gender"),
     data.get("marital_status"),data.get("dependents"),data.get("phone"),data.get("email"),
     data.get("education"),data.get("occupation"),data.get("work_experience"),data.get("company_name"),
     data.get("monthly_income"),data.get("annual_income"),data.get("existing_loans"),
     data.get("existing_emi"),data.get("savings"),data.get("assets"),data.get("bank_name"),
     data.get("cibil_score"),data.get("loan_type"),data.get("loan_amount"),
     data.get("interest_rate"),data.get("loan_tenure"),data.get("purpose"),
     data.get("emi"),data.get("total_interest"),data.get("total_payment"),
     data.get("eligibility"),data.get("approval_probability"),
     data.get("risk_level"),data.get("prediction"),shap_str))
    app_id = c.lastrowid
    conn.commit()
    conn.close()
    return app_id


def get_all_applications():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM loan_applications ORDER BY applied_at DESC")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_user_applications(user_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM loan_applications WHERE user_id=? ORDER BY applied_at DESC",(user_id,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def update_application_status(app_id, status):
    conn = get_connection()
    c = conn.cursor()
    c.execute("UPDATE loan_applications SET status=? WHERE id=?",(status,app_id))
    conn.commit()
    conn.close()


def get_analytics_data():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM loan_applications")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_all_users():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id,username,full_name,email,phone,role,created_at FROM users")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows
