import pandas as pd
import numpy as np

def detect_problem_type(y):
    unique_ratio = y.nunique() / len(y)
    if unique_ratio > 0.5:
        return "regression"
    return "classification"

def split_features_target(df, target_col):
    X = df.drop(columns=[target_col])
    X = X.select_dtypes(include=np.number)
    y = df[target_col]
    return X, y
