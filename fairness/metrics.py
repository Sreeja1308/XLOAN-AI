import numpy as np
import pandas as pd

def statistical_parity(data, target_col, protected_attr):
    """Difference in approval rates between groups."""
    rates = data.groupby(protected_attr)[target_col].mean()
    return rates.max() - rates.min()

def disparate_impact(data, target_col, protected_attr):
    """Ratio of approval rates (min/max)."""
    rates = data.groupby(protected_attr)[target_col].mean()
    return min(rates) / max(rates)

def equal_opportunity_difference(y_true, y_pred, protected_attr_values):
    """Difference in True Positive Rate between groups."""
    groups = np.unique(protected_attr_values)
    tprs = []
    for g in groups:
        mask = protected_attr_values == g
        tp = np.sum((y_true[mask] == 1) & (y_pred[mask] == 1))
        fn = np.sum((y_true[mask] == 1) & (y_pred[mask] == 0))
        tpr = tp / (tp + fn) if (tp+fn) > 0 else 0
        tprs.append(tpr)
    return max(tprs) - min(tprs)

def predictive_parity(y_true, y_pred, protected_attr_values):
    """Difference in Positive Predictive Value between groups."""
    groups = np.unique(protected_attr_values)
    ppvs = []
    for g in groups:
        mask = protected_attr_values == g
        tp = np.sum((y_true[mask] == 1) & (y_pred[mask] == 1))
        fp = np.sum((y_true[mask] == 0) & (y_pred[mask] == 1))
        ppv = tp / (tp + fp) if (tp+fp) > 0 else 0
        ppvs.append(ppv)
    return max(ppvs) - min(ppvs)