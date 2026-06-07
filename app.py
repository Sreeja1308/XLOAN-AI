import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import json
import plotly.graph_objects as go
import plotly.express as px
from fairness.metrics import statistical_parity, disparate_impact, equal_opportunity_difference, predictive_parity
from counterfactual.cf_gen import find_counterfactual
from utils.data_loader import load_dataset, load_model, load_preprocessor, get_feature_names
from utils.visualization import (plot_roc_curve_plotly, plot_confusion_matrix_plotly,
                                 plot_shap_waterfall, plot_shap_bar_chart)

st.set_page_config(layout="wide")
st.title("🏦 X-LoanAI Enhanced")
st.subheader("Explainable & Fair AI for Transparent Loan Decisions")

# Load models and data
@st.cache_resource
def load_models():
    xgb = load_model('models/xgb_best.pkl')
    rf = load_model('models/rf_model.pkl')
    lr = load_model('models/lr_model.pkl')
    preprocessor = load_preprocessor('models/preprocessor.pkl')
    return xgb, rf, lr, preprocessor

@st.cache_data
def load_data():
    return load_dataset('data/loan_data.csv')

xgb, rf, lr, preprocessor = load_models()
df = load_data()

# SHAP explainer for XGBoost
explainer = shap.TreeExplainer(xgb.named_steps['classifier'])
feature_names = get_feature_names(preprocessor)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict Loan", "📊 Model Comparison", "⚖️ Fairness Dashboard", "🔍 Data Explorer"])

with tab1:
    st.header("Loan Application Evaluation")
    with st.form("loan_form"):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            region = st.selectbox("Region", ["Urban", "Rural"])
            income = st.number_input("Annual Income (₹)", 20000, 2000000, 500000, step=50000)
            credit_score = st.number_input("Credit Score", 300, 900, 700, step=10)
        with col2:
            age = st.number_input("Age", 18, 65, 30, step=1)
            loan_amount = st.number_input("Loan Amount (₹)", 50000, 1000000, 200000, step=50000)
            loan_term = st.number_input("Loan Term (months)", 6, 72, 36, step=6)
            employment_years = st.number_input("Employment Years", 0, 40, 3, step=1)
            existing_loans = st.number_input("Existing Loans", 0, 5, 1, step=1)
        submit = st.form_submit_button("Evaluate Loan")

    if submit:
        instance = pd.DataFrame([{
            'age': age, 'gender': gender, 'region': region,
            'income': income, 'credit_score': credit_score,
            'loan_amount': loan_amount, 'loan_term': loan_term,
            'employment_years': employment_years, 'existing_loans': existing_loans
        }])

        prob = xgb.predict_proba(instance)[0][1]
        decision = "APPROVED ✅" if prob >= 0.5 else "REJECTED ❌"
        if prob >= 0.5:
            st.success(decision)
        else:
            st.error(decision)
        st.metric("Approval Probability", f"{prob:.2%}")

        # SHAP Explanation
        st.subheader("🔍 Why this decision?")
        instance_trans = preprocessor.transform(instance)
        shap_values = explainer.shap_values(instance_trans)[0]
        shap_df = pd.DataFrame({'feature': feature_names, 'shap_value': shap_values}).sort_values('shap_value', ascending=False)

        # --- Natural Language Explanation ---
        st.subheader("📝 Explanation")
        if prob >= 0.5:
            reason_text = "approved"
            sentiment = "positive"
        else:
            reason_text = "rejected"
            sentiment = "negative"

        top_pos = shap_df[shap_df['shap_value'] > 0].nlargest(3, 'shap_value')
        top_neg = shap_df[shap_df['shap_value'] < 0].nsmallest(3, 'shap_value')

        explanation_sentences = []
        if sentiment == "positive":
            explanation_sentences.append(f"✅ Your loan application was **{reason_text}** with a probability of {prob:.1%}. The key reasons are:")
        else:
            explanation_sentences.append(f"❌ Your loan application was **{reason_text}** with a probability of {prob:.1%}. The main reasons are:")

        if not top_pos.empty:
            explanation_sentences.append("**Factors that helped:**")
            for _, row in top_pos.iterrows():
                feature = row['feature'].replace('_', ' ').replace('gender Male', 'being male').replace('gender Female', 'being female')
                explanation_sentences.append(f"• {feature}: increased your chance by {row['shap_value']:.2f}")
        else:
            explanation_sentences.append("No strong positive factors were identified.")

        if not top_neg.empty:
            explanation_sentences.append("**Factors that hurt:**")
            for _, row in top_neg.iterrows():
                feature = row['feature'].replace('_', ' ').replace('gender Male', 'being male').replace('gender Female', 'being female')
                explanation_sentences.append(f"• {feature}: decreased your chance by {abs(row['shap_value']):.2f}")
        else:
            explanation_sentences.append("No strong negative factors were identified.")

        for line in explanation_sentences:
            st.write(line)

        # SIMPLE BAR CHART (default)
        fig = plot_shap_bar_chart(shap_df, top_n=3)
        st.pyplot(fig)

        

        # Counterfactual suggestions
        if prob < 0.5:
            st.subheader("🔄 How to improve your chances?")
            suggestions = find_counterfactual(instance.iloc[0], xgb, preprocessor, explainer)
            for s in suggestions:
                st.write(f"• {s}")

with tab2:
    st.header("Model Comparison")
    with open('models/model_metrics.json') as f:
        metrics = json.load(f)
    st.subheader("Performance Metrics")
    df_metrics = pd.DataFrame(metrics).T
    st.dataframe(df_metrics.style.format("{:.3f}"))

    st.subheader("ROC Curves")
    X_test = df.drop('loan_approved', axis=1).sample(500, random_state=42)
    y_test = df.loc[X_test.index, 'loan_approved']
    fig = go.Figure()
    for name, model in zip(['XGBoost', 'Random Forest', 'Logistic Regression'], [xgb, rf, lr]):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_fig = plot_roc_curve_plotly(y_test, y_proba, name)
        for trace in roc_fig.data:
            fig.add_trace(trace)
    fig.update_layout(title="ROC Curves")
    st.plotly_chart(fig)

    st.subheader("Confusion Matrix (XGBoost)")
    y_pred = xgb.predict(X_test)
    fig_cm = plot_confusion_matrix_plotly(y_test, y_pred)
    st.plotly_chart(fig_cm)

with tab3:
    st.header("Fairness Analysis")
    X_test = df.drop('loan_approved', axis=1).sample(500, random_state=42)
    y_test = df.loc[X_test.index, 'loan_approved']
    y_pred = xgb.predict(X_test)

    gender_vals = X_test['gender'].values
    sp = statistical_parity(pd.DataFrame({'loan_approved': y_pred, 'gender': gender_vals}), 'loan_approved', 'gender')
    di = disparate_impact(pd.DataFrame({'loan_approved': y_pred, 'gender': gender_vals}), 'loan_approved', 'gender')
    eod = equal_opportunity_difference(y_test, y_pred, gender_vals)
    pp = predictive_parity(y_test, y_pred, gender_vals)

    st.subheader("Metrics by Gender")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Statistical Parity Difference", f"{sp:.3f}")
    col2.metric("Disparate Impact Ratio", f"{di:.3f}")
    col3.metric("Equal Opportunity Difference", f"{eod:.3f}")
    col4.metric("Predictive Parity Difference", f"{pp:.3f}")

    # Approval rates
    approval_rates = pd.DataFrame({'Gender': gender_vals, 'Approved': y_pred}).groupby('Gender')['Approved'].mean().reset_index()
    fig = px.bar(approval_rates, x='Gender', y='Approved', title='Approval Rate by Gender',
                 labels={'Approved': 'Approval Rate'}, text='Approved')
    fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    st.plotly_chart(fig)

    # Region fairness
    region_vals = X_test['region'].values
    sp_reg = statistical_parity(pd.DataFrame({'loan_approved': y_pred, 'region': region_vals}), 'loan_approved', 'region')
    di_reg = disparate_impact(pd.DataFrame({'loan_approved': y_pred, 'region': region_vals}), 'loan_approved', 'region')
    st.write(f"**Region** – Statistical Parity: {sp_reg:.3f}, Disparate Impact: {di_reg:.3f}")

with tab4:
    st.header("Data Explorer")
    st.dataframe(df.head(100))
    st.subheader("Distribution of Key Features")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x='credit_score', nbins=50, title='Credit Score Distribution')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(df, x='income', nbins=50, title='Income Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Approval Rate by Credit Score")
    # Convert bins to strings to avoid Interval serialization error
    df['credit_score_bin'] = pd.cut(df['credit_score'], bins=10).astype(str)
    approval_by_score = df.groupby('credit_score_bin', observed=False)['loan_approved'].mean().reset_index()
    # Sort bins by numeric value for proper order
    bins_sorted = sorted(approval_by_score['credit_score_bin'], 
                         key=lambda x: float(x.split(',')[0].strip('(')))
    approval_by_score['credit_score_bin'] = pd.Categorical(approval_by_score['credit_score_bin'], 
                                                           categories=bins_sorted, ordered=True)
    approval_by_score = approval_by_score.sort_values('credit_score_bin')
    fig = px.bar(approval_by_score, x='credit_score_bin', y='loan_approved', 
                 title='Approval Rate by Credit Score',
                 labels={'loan_approved': 'Approval Rate', 'credit_score_bin': 'Credit Score Range'})
    st.plotly_chart(fig)