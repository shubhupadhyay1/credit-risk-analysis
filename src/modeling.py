# src/modeling.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, 
                             f1_score, roc_auc_score, mean_squared_error, roc_curve, auc)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Trains and evaluates a model, returning key metrics.
    """
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Predict probabilities for AUC-ROC (if applicable)
    y_test_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    metrics = {
        "Train Accuracy": accuracy_score(y_train, y_train_pred),
        "Test Accuracy": accuracy_score(y_test, y_test_pred),
        "Train F1 Score": f1_score(y_train, y_train_pred),
        "Test F1 Score": f1_score(y_test, y_test_pred),
    }
    
    if y_test_prob is not None:
        metrics["AUC-ROC"] = roc_auc_score(y_test, y_test_prob)
    
    print("\nConfusion Matrix (Test Data):")
    print(confusion_matrix(y_test, y_test_pred))
    print("\nClassification Report (Test Data):")
    print(classification_report(y_test, y_test_pred))
    
    return metrics

# Custom function to rank models based on multiple metrics
def rank_models(df, weights=None):
    """
    Ranks models based on weighted sum of metrics.
    
    Args:
        df (pd.DataFrame): DataFrame containing model metrics.
        weights (dict): Dictionary specifying weights for each metric.
        
    Returns:
        str: Name of the best model.
    """
    if weights is None:
        weights = {"AUC-ROC": 0.5, "Test F1 Score": 0.5}  # Default weights
    
    # Compute weighted score for each model
    weighted_score = df[list(weights.keys())].mul(pd.Series(weights)).sum(axis=1)
    
    # Return the name of the model with the highest score
    return weighted_score.sort_values(ascending=False).index[0]