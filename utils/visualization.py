import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_curve, auc, confusion_matrix

def plot_roc_curve_plotly(y_test, y_proba, model_name):
    """Return a Plotly figure for ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                             name=f'{model_name} (AUC={roc_auc:.3f})'))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                             name='Random', line=dict(dash='dash')))
    fig.update_layout(xaxis_title='False Positive Rate',
                      yaxis_title='True Positive Rate')
    return fig

def plot_confusion_matrix_plotly(y_true, y_pred, labels=['Reject', 'Approve']):
    """Return a Plotly figure for confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                    labels=dict(x="Predicted", y="Actual"),
                    x=labels, y=labels)
    return fig

def plot_shap_waterfall(explainer, instance_trans, feature_names, max_display=10):
    """Return a matplotlib figure of SHAP waterfall plot."""
    shap_values = explainer.shap_values(instance_trans)[0]
    fig, ax = plt.subplots()
    shap.waterfall_plot(shap.Explanation(values=shap_values,
                                         base_values=explainer.expected_value,
                                         data=instance_trans[0],
                                         feature_names=feature_names),
                        show=False, max_display=max_display)
    return fig

def plot_shap_bar_chart(shap_df, top_n=3):
    """Create a simple horizontal bar chart of top positive and negative contributors."""
    pos = shap_df[shap_df['shap_value'] > 0].nlargest(top_n, 'shap_value')
    neg = shap_df[shap_df['shap_value'] < 0].nsmallest(top_n, 'shap_value')
    combined = pd.concat([pos, neg])
    if combined.empty:
        combined = shap_df.nlargest(top_n*2, 'shap_value')
    fig, ax = plt.subplots(figsize=(8, max(4, len(combined)*0.5)))
    colors = ['green' if x > 0 else 'red' for x in combined['shap_value']]
    ax.barh(combined['feature'], combined['shap_value'], color=colors)
    ax.axvline(0, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Impact on Approval Probability')
    ax.set_title('How Each Feature Influenced the Decision')
    for i, (val, feat) in enumerate(zip(combined['shap_value'], combined['feature'])):
        ax.text(val, i, f' {val:.2f}', va='center')
    return fig