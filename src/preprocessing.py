# src/preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from imblearn.over_sampling import SMOTE

def load_data(file_path):
    """
    Loads the dataset from a given file path.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.
    """
    df = pd.read_csv(file_path)
    return df


def impute_missing_values(df, cat_feats, num_feats):
    """
    Imputes missing values in the dataset.
    - Categorical features are imputed using the mode.
    - Numerical features are imputed using the median.
    
    Args:
        df (pd.DataFrame): Input dataframe with missing values.
        cat_feats (list): List of categorical feature columns.
        num_feats (list): List of numerical feature columns.
    
    Returns:
        pd.DataFrame: Dataframe with imputed values.
    """
    # Impute categorical features with mode
    for col in cat_feats:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)

    # Impute numerical features with median
    for col in num_feats:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
    
    return df


def encode_categorical_features(df, cat_feats):
    """
    Encodes categorical features using one-hot encoding.
    
    Args:
        df (pd.DataFrame): Input dataframe with categorical features.
        cat_feats (list): List of categorical feature columns.
        
    Returns:
        pd.DataFrame: Dataframe with encoded categorical features.
    """
    df_encoded = pd.get_dummies(df, columns=cat_feats, drop_first=True)
    return df_encoded


def scale_features(df, num_feats):
    """
    Scales numerical features using Min-Max scaling.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        num_feats (list): List of numerical feature columns to scale.
        
    Returns:
        pd.DataFrame: Dataframe with scaled numerical features.
    """
    scaler = MinMaxScaler()
    df[num_feats] = scaler.fit_transform(df[num_feats])
    return df


def balance_data(X, y):
    """
    Balances the dataset using SMOTE (Synthetic Minority Over-sampling Technique).
    
    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        
    Returns:
        pd.DataFrame, pd.Series: Balanced feature matrix and target variable.
    """
    sm = SMOTE(random_state=42)
    X_balanced, y_balanced = sm.fit_resample(X, y)
    return X_balanced, y_balanced
