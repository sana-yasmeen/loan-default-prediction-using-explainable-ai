"""
EMI Calculator Module
Computes EMI, total interest, and repayment schedule.
"""

def calculate_emi(principal: float, annual_rate: float, tenure_months: int) -> dict:
    """
    EMI = [P × R × (1+R)^N] / [(1+R)^N – 1]
    """
    if annual_rate == 0:
        emi = principal / tenure_months
        return {
            "emi": round(emi, 2),
            "total_payment": round(principal, 2),
            "total_interest": 0.0,
            "principal": principal,
            "schedule": _build_schedule(principal, 0, tenure_months, emi)
        }

    R = annual_rate / (12 * 100)
    N = tenure_months
    factor = (1 + R) ** N
    emi = (principal * R * factor) / (factor - 1)
    total_payment = emi * N
    total_interest = total_payment - principal

    return {
        "emi": round(emi, 2),
        "total_payment": round(total_payment, 2),
        "total_interest": round(total_interest, 2),
        "principal": round(principal, 2),
        "schedule": _build_schedule(principal, R, N, emi)
    }


def _build_schedule(principal, R, N, emi):
    schedule = []
    balance = principal
    for month in range(1, N + 1):
        if R == 0:
            interest_comp = 0
            principal_comp = emi
        else:
            interest_comp = balance * R
            principal_comp = emi - interest_comp
        balance -= principal_comp
        if balance < 0:
            balance = 0
        schedule.append({
            "month": month,
            "emi": round(emi, 2),
            "principal_component": round(principal_comp, 2),
            "interest_component": round(interest_comp, 2),
            "balance": round(balance, 2)
        })
    return schedule


# Bank-wise interest rates (annual %)
BANK_RATES = {
    "SBI": {
        "Personal Loan": 10.50,
        "Home Loan": 8.40,
        "Education Loan": 8.15,
        "Vehicle Loan": 9.15,
        "Gold Loan": 7.50,
        "Business Loan": 11.20,
        "Agricultural Loan": 7.00,
    },
    "HDFC": {
        "Personal Loan": 10.75,
        "Home Loan": 8.70,
        "Education Loan": 9.00,
        "Vehicle Loan": 9.40,
        "Gold Loan": 7.75,
        "Business Loan": 11.90,
        "Agricultural Loan": 8.00,
    },
    "ICICI": {
        "Personal Loan": 10.85,
        "Home Loan": 8.75,
        "Education Loan": 9.20,
        "Vehicle Loan": 9.30,
        "Gold Loan": 7.90,
        "Business Loan": 12.00,
        "Agricultural Loan": 8.25,
    },
    "Axis Bank": {
        "Personal Loan": 11.00,
        "Home Loan": 8.90,
        "Education Loan": 9.50,
        "Vehicle Loan": 9.60,
        "Gold Loan": 8.00,
        "Business Loan": 12.50,
        "Agricultural Loan": 8.50,
    },
    "Canara Bank": {
        "Personal Loan": 10.40,
        "Home Loan": 8.30,
        "Education Loan": 8.10,
        "Vehicle Loan": 9.00,
        "Gold Loan": 7.35,
        "Business Loan": 11.00,
        "Agricultural Loan": 6.90,
    },
    "Union Bank": {
        "Personal Loan": 10.30,
        "Home Loan": 8.25,
        "Education Loan": 8.05,
        "Vehicle Loan": 8.90,
        "Gold Loan": 7.25,
        "Business Loan": 10.80,
        "Agricultural Loan": 6.80,
    },
}

LOAN_TYPES = list(next(iter(BANK_RATES.values())).keys())
BANKS = list(BANK_RATES.keys())


def get_rate(bank: str, loan_type: str) -> float:
    return BANK_RATES.get(bank, {}).get(loan_type, 10.0)
