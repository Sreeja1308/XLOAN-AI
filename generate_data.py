import numpy as np
import pandas as pd

def generate_loan_data(n=20000, random_state=42, gender_bias=True):
    np.random.seed(random_state)
    age = np.random.randint(18, 66, n)
    gender = np.random.choice(['Male', 'Female'], n, p=[0.5, 0.5])
    region = np.random.choice(['Urban', 'Rural'], n, p=[0.7, 0.3])
    income = np.random.randint(100_000, 2_000_000, n)
    credit_score = np.random.randint(300, 901, n)
    loan_amount = np.random.randint(50_000, 1_000_000, n)
    loan_term = np.random.choice([12, 24, 36, 48, 60, 72], n)
    employment_years = np.random.randint(0, 41, n)
    existing_loans = np.random.randint(0, 6, n)

    prob = 1 / (1 + np.exp(-(credit_score - 600) / 50))
    dti = loan_amount / income
    prob -= dti * 0.3
    prob -= existing_loans * 0.1
    if gender_bias:
        prob[gender == 'Female'] -= 0.1
    prob += np.random.normal(0, 0.05, n)
    prob = np.clip(prob, 0, 1)
    loan_approved = np.random.binomial(1, prob)

    df = pd.DataFrame({
        'age': age, 'gender': gender, 'region': region,
        'income': income, 'credit_score': credit_score,
        'loan_amount': loan_amount, 'loan_term': loan_term,
        'employment_years': employment_years, 'existing_loans': existing_loans,
        'loan_approved': loan_approved
    })
    return df

if __name__ == '__main__':
    df = generate_loan_data(gender_bias=True)
    df.to_csv('data/loan_data.csv', index=False)
    print(f"Generated {len(df)} records with gender bias.")