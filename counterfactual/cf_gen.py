import numpy as np
import pandas as pd

def find_counterfactual(instance, model, preprocessor, explainer, threshold=0.5):
    """
    Generate counterfactual suggestions based on SHAP values.
    """
    X = pd.DataFrame([instance])
    prob = model.predict_proba(X)[0][1]
    if prob >= threshold:
        return []   # already approved

    suggestions = []

    # Simple rule-based suggestions (can be enhanced with SHAP)
    if instance['credit_score'] < 700:
        suggestions.append("Increase your credit score by 30-50 points.")
    if instance['existing_loans'] > 2:
        suggestions.append("Reduce existing loans before reapplying.")
    if instance['loan_amount'] > instance['income'] * 0.5:
        suggestions.append("Consider a smaller loan amount.")
    if instance['employment_years'] < 2:
        suggestions.append("Gain more work experience (at least 2 years).")
    if not suggestions:
        suggestions.append("Add a co-applicant to strengthen your application.")
    return suggestions[:3]