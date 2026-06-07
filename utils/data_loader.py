import pandas as pd
import joblib

def load_dataset(path='data/loan_data.csv'):
    """Load the loan dataset."""
    return pd.read_csv(path)

def load_model(model_path='models/xgb_best.pkl'):
    """Load a trained model pipeline."""
    return joblib.load(model_path)

def load_preprocessor(preprocessor_path='models/preprocessor.pkl'):
    """Load the preprocessor."""
    return joblib.load(preprocessor_path)

def get_feature_names(preprocessor):
    """
    Extract feature names after preprocessing.
    Assumes preprocessor has named transformers 'num' and 'cat'.
    """
    numerical_cols = ['age', 'income', 'credit_score', 'loan_amount',
                      'loan_term', 'employment_years', 'existing_loans']
    # Categorical one-hot encoding
    ohe = preprocessor.named_transformers_['cat']
    # For gender: categories are ['Female', 'Male'] (example). We drop the first.
    gender_cats = ohe.categories_[0][1:]   # e.g., ['Male'] if Female is dropped
    region_cats = ohe.categories_[1][1:]   # e.g., ['Rural'] if Urban is dropped
    feature_names = numerical_cols
    for cat in gender_cats:
        feature_names.append(f'gender_{cat}')
    for cat in region_cats:
        feature_names.append(f'region_{cat}')
    return feature_names