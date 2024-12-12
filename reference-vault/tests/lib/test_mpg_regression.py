import pytest
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from capstone.mpg_regression import MpgRegressionCapstone

@pytest.fixture
def mpg_instance():
    return MpgRegressionCapstone()


def test_preprocess(mpg_instance):
    X, y = mpg_instance.preprocess()
    assert X.shape[0] == len(y), "Features and labels should have the same number of samples"


def test_split_data(mpg_instance):
    X, y = mpg_instance.preprocess()
    X_train, X_test, y_train, y_test = mpg_instance.split_data(X, y)
    assert len(X_train) > 0 and len(X_test) > 0, "Training and testing sets should not be empty"
    assert len(y_train) > 0 and len(y_test) > 0, "Training and testing sets should not be empty"


def test_fit_and_predict(mpg_instance):
    X, y = mpg_instance.preprocess()
    X_train, X_test, y_train, y_test = mpg_instance.split_data(X, y)

    mpg_instance.fit(X_train, y_train)
    y_pred = mpg_instance.predict(X_test)

    mae = np.mean(np.abs(y_test - y_pred))
    mse = np.mean((y_test - y_pred) ** 2)
    r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

    assert mae > 0, "MAE should be greater than 0"
    assert mse > 0, "MSE should be greater than 0"
    assert -1 <= r2 <= 1, "R² score should be between -1 and 1"


def test_fit_and_predict_comparison(mpg_instance):
    X, y = mpg_instance.preprocess()
    X_train, X_test, y_train, y_test = mpg_instance.split_data(X, y)

    sklearn_model = LinearRegression()
    sklearn_model.fit(X_train, y_train)
    y_pred_sklearn = sklearn_model.predict(X_test)

    sklearn_mae = mean_absolute_error(y_test, y_pred_sklearn)
    sklearn_mse = mean_squared_error(y_test, y_pred_sklearn)
    sklearn_r2 = r2_score(y_test, y_pred_sklearn)

    mpg_instance.fit(X_train, y_train)
    y_pred_custom = mpg_instance.predict(X_test)

    custom_mae = np.mean(np.abs(y_test - y_pred_custom))
    custom_mse = np.mean((y_test - y_pred_custom) ** 2)
    custom_r2 = 1 - (np.sum((y_test - y_pred_custom) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

    assert np.isclose(sklearn_mae, custom_mae, atol=1e-5), f"MAE mismatch: {sklearn_mae} (sklearn) vs {custom_mae} (custom)"
    assert np.isclose(sklearn_mse, custom_mse, atol=1e-5), f"MSE mismatch: {sklearn_mse} (sklearn) vs {custom_mse} (custom)"
    assert np.isclose(sklearn_r2, custom_r2, atol=1e-5), f"R² mismatch: {sklearn_r2} (sklearn) vs {custom_r2} (custom)"

    print("Metrics comparison passed:")
    print(f"MAE: sklearn={sklearn_mae}, custom={custom_mae}")
    print(f"MSE: sklearn={sklearn_mse}, custom={custom_mse}")
    print(f"R²: sklearn={sklearn_r2}, custom={custom_r2}")

def test_predict_new_data(mpg_instance):
    mpg_instance.run()
    new_data = {
        "cylinders": 8,
        "displacement": 307.0,
        "horsepower": 130.0,
        "weight": 3504,
        "acceleration": 12.0,
        "model_year": 70,
        "origin": "usa",
        "name": "chevrolet chevelle malibu"
    }
    prediction = mpg_instance.predict_new(new_data)
    assert isinstance(prediction, float), "Prediction should return a single float value"
