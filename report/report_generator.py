def generate_report(problem_type, metrics, insight, shap_summary, stability):

    report = f"""
    ===== EXPLAINABLE ANALYTICS REPORT =====

    Problem Type: {problem_type}

    Metrics:
    {metrics}

    Model Insight:
    {insight}

    SHAP Summary:
    {shap_summary}

    Stability (std across runs):
    {stability}

    =========================================
    """

    return report
