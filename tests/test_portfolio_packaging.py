import ast
import csv
import json
from pathlib import Path

import numpy as np

from train.evaluate import save_metrics


ROOT = Path(__file__).resolve().parents[1]


def test_main_is_import_safe_cli_entrypoint():
    tree = ast.parse((ROOT / "main.py").read_text())

    function_names = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert "build_arg_parser" in function_names
    assert "run_pipeline" in function_names
    assert "main" in function_names

    last_node = tree.body[-1]
    assert isinstance(last_node, ast.If)
    assert isinstance(last_node.test, ast.Compare)
    assert ast.unparse(last_node.test) == "__name__ == '__main__'"


def test_save_metrics_writes_csv_and_json(tmp_path):
    metrics = [
        {
            "method": "DemoModel",
            "auprc": np.float64(0.91),
            "auc_roc": np.float64(0.98),
            "f1": 0.75,
            "precision": 0.8,
            "recall": 0.7,
            "threshold": 0.42,
            "false_alarms": np.int64(2),
            "missed_fraud": np.int64(3),
            "total_cost": np.int64(32),
        }
    ]

    csv_path, json_path = save_metrics(metrics, tmp_path)

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["method"] == "DemoModel"
    assert rows[0]["auprc"] == "0.91"

    with json_path.open() as f:
        data = json.load(f)
    assert data[0]["method"] == "DemoModel"
    assert data[0]["threshold"] == 0.42


def test_portfolio_docs_describe_problem_results_and_setup():
    readme = (ROOT / "README.md").read_text()
    requirements = (ROOT / "requirements.txt").read_text()

    for phrase in [
        "信用卡欺诈检测",
        "0.173%",
        "AUPRC",
        "Precision@k",
        "XGBoost+SMOTE",
        "python main.py --quick",
    ]:
        assert phrase in readme

    for package in [
        "pandas",
        "scikit-learn",
        "imbalanced-learn",
        "xgboost",
        "lightgbm",
        "torch",
    ]:
        assert package in requirements
