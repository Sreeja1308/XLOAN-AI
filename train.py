import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Load data
df = pd.read_csv('data/loan_data.csv')
X = df.drop('loan_approved', axis=1)
y = df['loan_approved']

categorical_cols = ['gender', 'region']
numerical_cols = ['age', 'income', 'credit_score', 'loan_amount',
                  'loan_term', 'employment_years', 'existing_loans']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(drop='first'), categorical_cols)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- XGBoost with tuning ----
xgb_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
])
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__subsample': [0.8, 1.0]
}
xgb_grid = GridSearchCV(xgb_pipe, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
xgb_grid.fit(X_train, y_train)
xgb_best = xgb_grid.best_estimator_

# ---- Random Forest ----
rf_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
rf_pipe.fit(X_train, y_train)

# ---- Logistic Regression ----
lr_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])
lr_pipe.fit(X_train, y_train)

# Save models
joblib.dump(xgb_best, 'models/xgb_best.pkl')
joblib.dump(rf_pipe, 'models/rf_model.pkl')
joblib.dump(lr_pipe, 'models/lr_model.pkl')
joblib.dump(preprocessor, 'models/preprocessor.pkl')

# Save tuning results
with open('models/tuning_results.json', 'w') as f:
    json.dump(xgb_grid.best_params_, f)

# Evaluate
models = {
    'XGBoost (tuned)': xgb_best,
    'Random Forest': rf_pipe,
    'Logistic Regression': lr_pipe
}
results = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    results[name] = {'accuracy': acc, 'auc': auc}
    print(f"{name}: Accuracy={acc:.3f}, AUC={auc:.3f}")

with open('models/model_metrics.json', 'w') as f:
    json.dump(results, f)