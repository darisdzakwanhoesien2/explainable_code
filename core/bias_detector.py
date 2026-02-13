def detect_class_imbalance(y):
    value_counts = y.value_counts(normalize=True)
    if value_counts.max() > 0.8:
        return "Severe class imbalance detected."
    return "Class distribution appears balanced."
