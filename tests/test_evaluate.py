import numpy as np

from train.evaluate import compute_metrics_from_validation_threshold


def test_compute_metrics_from_validation_threshold_does_not_tune_on_test_set():
    val_scores = np.array([0.0, 1.0])
    y_val = np.array([0, 1])
    test_scores = np.array([0.40, 0.45, 0.60, 0.70])
    y_test = np.array([1, 1, 0, 0])

    metrics = compute_metrics_from_validation_threshold(
        val_scores,
        y_val,
        test_scores,
        y_test,
        method_name="demo",
    )

    assert metrics["threshold"] == 0.5
    assert metrics["f1"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
