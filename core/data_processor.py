import pandas as pd
import numpy as np
from sklearn.utils.multiclass import type_of_target


def detect_problem_type(y):
    """
    Uses sklearn built-in detection for robust classification/regression detection
    """

    target_type = type_of_target(y)

    if target_type in ["continuous", "continuous-multioutput"]:
        return "regression"

    if target_type in ["binary", "multiclass"]:
        return "classification"

    # Fallback logic
    if y.dtype.kind in "if":  # integer or float
        if y.nunique() > 20:
            return "regression"

    return "classification"


def split_features_target(df, target_col):
    X = df.drop(columns=[target_col])
    X = X.select_dtypes(include=np.number)

    y = df[target_col]

    return X, y


# import pandas as pd
# import numpy as np

# def detect_problem_type(y):
#     unique_ratio = y.nunique() / len(y)
#     if unique_ratio > 0.5:
#         return "regression"
#     return "classification"

# def split_features_target(df, target_col):
#     X = df.drop(columns=[target_col])
#     X = X.select_dtypes(include=np.number)
#     y = df[target_col]
#     return X, y
