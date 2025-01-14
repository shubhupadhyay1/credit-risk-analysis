# src/eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def basic_statistics(df):
    """
    Displays basic statistics and information about the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe.
    """
    print(f"Dataset shape: {df.shape}")
    print("\nBasic Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum().sort_values(ascending=False).head(10))
    print("\nDescriptive Statistics:")
    print(df.describe())


def plot_distributions(df, num_feats):
    """
    Plots the distribution of numerical features.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        num_feats (list): List of numerical feature columns.
    """
    for col in num_feats:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.show()


def plot_categorical_counts(df, cat_feats):
    """
    Plots the count of categorical features.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        cat_feats (list): List of categorical feature columns.
    """
    for col in cat_feats:
        plt.figure(figsize=(8, 4))
        sns.countplot(x=col, data=df)
        plt.title(f"Count plot of {col}")
        plt.xticks(rotation=45)
        plt.show()
