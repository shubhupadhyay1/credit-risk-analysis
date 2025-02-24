# src/feature_engineering.py

import pandas as pd
import numpy as np
#import pandas.core.algorithms as algos
import scipy.stats as stats
import re
import traceback

def mono_bin(Y, X, max_bin=20, force_bin=3):
    """
    Performs monotonic binning on a numerical feature using numpy.percentile for binning.
    
    Args:
        Y (pd.Series): Target variable.
        X (pd.Series): Numerical feature to be binned.
        max_bin (int): Maximum number of bins.
        force_bin (int): Number of bins to force if monotonic binning fails.
    
    Returns:
        pd.DataFrame: Binned data with IV and WOE calculations.
    """
    # Convert X to numeric and drop invalid data
    X = pd.to_numeric(X, errors='coerce')
    
    if X.isnull().all():
        raise ValueError("Feature X contains no valid numeric data after conversion.")

    df = pd.DataFrame({"X": X, "Y": Y})
    notmiss = df[df["X"].notnull()]
    justmiss = df[df["X"].isnull()]

    r = 0
    d2 = None

    while np.abs(r) < 1 and max_bin > 2:
        try:
            # Ensure X is numeric
            bins = np.percentile(notmiss["X"], np.linspace(0, 100, max_bin))
            d1 = pd.DataFrame({
                "X": notmiss["X"], 
                "Y": notmiss["Y"], 
                "Bucket": pd.cut(notmiss["X"], bins, duplicates="drop")
            })
            d2 = d1.groupby("Bucket", as_index=True)
            r, _ = stats.spearmanr(d2.mean()["X"], d2.mean()["Y"])
            max_bin -= 1
        except Exception as e:
            print(f"Error during binning: {e}")
            max_bin -= 1

    if d2 is None or len(d2) == 1 or max_bin <= 2:
        bins = np.percentile(notmiss["X"], np.linspace(0, 100, force_bin))
        d1 = pd.DataFrame({
            "X": notmiss["X"], 
            "Y": notmiss["Y"], 
            "Bucket": pd.cut(notmiss["X"], bins, include_lowest=True, duplicates="drop")
        })
        d2 = d1.groupby("Bucket", as_index=True)

    d3 = pd.DataFrame(d2.min().X, columns=["MIN_VALUE"])
    d3["MAX_VALUE"] = d2.max().X
    d3["COUNT"] = d2.count().Y
    d3["EVENT"] = d2.sum().Y
    d3["NONEVENT"] = d3["COUNT"] - d3["EVENT"]
    d3["EVENT_RATE"] = d3.EVENT / d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT / d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT / d3.EVENT.sum()
    d3["DIST_NON_EVENT"] = d3.NONEVENT / d3.NONEVENT.sum()
    d3["WOE"] = np.log(d3.DIST_EVENT / d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT - d3.DIST_NON_EVENT) * d3["WOE"]
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3["IV"] = d3["IV"].sum()

    return d3



def char_bin(Y, X):
    """
    Bins categorical features and computes IV and WOE.
    
    Args:
        Y (pd.Series): Target variable.
        X (pd.Series): Categorical feature to be binned.
    
    Returns:
        pd.DataFrame: Binned data with IV and WOE calculations.
    """
    df = pd.DataFrame({"X": X, "Y": Y})
    d2 = df.groupby("X").agg(
        COUNT=("Y", "count"),
        EVENT=("Y", "sum"),
        NONEVENT=("Y", lambda y: y.count() - y.sum())
    )

    d2["EVENT_RATE"] = d2.EVENT / d2.COUNT
    d2["NON_EVENT_RATE"] = d2.NONEVENT / d2.COUNT
    d2["DIST_EVENT"] = d2.EVENT / d2.EVENT.sum()
    d2["DIST_NON_EVENT"] = d2.NONEVENT / d2.NONEVENT.sum()
    d2["WOE"] = np.log(d2.DIST_EVENT / d2.DIST_NON_EVENT)
    d2["IV"] = (d2.DIST_EVENT - d2.DIST_NON_EVENT) * d2["WOE"]
    d2 = d2.replace([np.inf, -np.inf], 0)

    return d2.reset_index()



def compute_iv(df, target, threshold=0.02):
    """
    Computes IV for all features and drops those with IV below the threshold.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        target (str): Name of the target column.
        threshold (float): IV threshold for feature selection.
    
    Returns:
        pd.DataFrame: Dataframe with selected features.
        pd.DataFrame: IV values for all features.
    """
    iv_values = {}
    features = df.columns.drop(target)

    for feature in features:
        if df[feature].dtype == 'object':
            iv = char_bin(df[target], df[feature])["IV"].sum()
        else:
            iv = mono_bin(df[target], df[feature]).iloc[0]["IV"]

        iv_values[feature] = iv
    
    # Create a DataFrame for IV values
    iv_df = pd.DataFrame(list(iv_values.items()), columns=["Feature", "IV"]).sort_values(by="IV", ascending=False)
    
    # Select features with IV greater than or equal to the threshold
    selected_features = iv_df[iv_df["IV"] >= threshold]["Feature"].tolist()
    
    print(f"Selected {len(selected_features)} features with IV >= {threshold}")
    return df[selected_features + [target]], iv_df


def impute_missing_values(df, cat_feats, num_feats):
    """
    Imputes missing values in the dataframe.
    
    - Categorical features are imputed with the mode.
    - Numerical features are imputed with the median.
    - Only processes columns that exist in the dataframe.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        cat_feats (list): List of categorical feature columns.
        num_feats (list): List of numerical feature columns.
    
    Returns:
        pd.DataFrame: Dataframe with missing values imputed.
    """
    # Impute categorical features with mode
    for col in cat_feats:
        if col in df.columns and df[col].isnull().sum() > 0:  # Check if column exists and has missing values
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
    
    # Impute numerical features with median
    for col in num_feats:
        if col in df.columns and df[col].isnull().sum() > 0:  # Check if column exists and has missing values
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
    
    return df

