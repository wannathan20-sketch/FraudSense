import numpy as np
import pandas as pd

from data.dataset import split_and_preprocess


def test_split_and_preprocess_fits_scaler_on_train_only():
    n = 60
    df = pd.DataFrame(
        {
            "Time": np.arange(n, dtype=float),
            "V1": np.linspace(-5, 25, n),
            "Amount": np.linspace(10, 1000, n),
            "Class": [0, 1] * (n // 2),
        }
    )

    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_preprocess(df)

    assert len(X_train) + len(X_val) + len(X_test) == n
    assert len(y_train) + len(y_val) + len(y_test) == n
    assert np.allclose(X_train.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(X_train.std(axis=0), 1.0, atol=1e-6)

    X_all = np.vstack([X_train, X_val, X_test])
    assert not np.allclose(X_all.mean(axis=0), 0.0, atol=1e-6)

    raw_train = scaler.inverse_transform(X_train)
    assert np.allclose(raw_train.mean(axis=0), scaler.mean_)
