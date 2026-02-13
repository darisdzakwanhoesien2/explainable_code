from scipy.stats import ks_2samp

def detect_drift(train_col, test_col):
    stat, p_value = ks_2samp(train_col, test_col)
    if p_value < 0.05:
        return "Data drift detected."
    return "No significant drift."
