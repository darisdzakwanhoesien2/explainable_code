import os
import json
import pandas as pd


BASE_DIR = "storage"
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
EXPERIMENT_DIR = os.path.join(BASE_DIR, "experiments")


def init_storage():
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)


def save_dataset(df, experiment_id):
    path = os.path.join(DATASET_DIR, f"{experiment_id}.csv")
    df.to_csv(path, index=False)
    return path


def save_experiment(experiment_id, metrics, config, shap_summary):

    exp_path = os.path.join(EXPERIMENT_DIR, experiment_id)
    os.makedirs(exp_path, exist_ok=True)

    with open(os.path.join(exp_path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    with open(os.path.join(exp_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    with open(os.path.join(exp_path, "shap_summary.json"), "w") as f:
        json.dump({"shap_summary": shap_summary}, f, indent=4)

    # Generate simple text report
    report_text = f"""
    EXPERIMENT ID: {experiment_id}

    CONFIG:
    {config}

    METRICS:
    {metrics}

    SHAP SUMMARY:
    {shap_summary}
    """

    with open(os.path.join(exp_path, "report.txt"), "w") as f:
        f.write(report_text)

    return exp_path
