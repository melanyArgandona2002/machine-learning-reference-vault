import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from capstone.tumor_classification import TumorClassificationCapstone  # Replace with the filename of your script

@pytest.fixture
def capstone_model():
    """
    Fixture to initialize the TumorClassificationCapstone class.
    """
    return TumorClassificationCapstone()

def test_train_model(capstone_model):
    """
    Test if the custom model can train without errors.
    """
    capstone_model.train()
    assert capstone_model.theta is not None, "Model parameters should not be None after training."

def test_predict_accuracy(capstone_model):
    """
    Compare the accuracy of the custom model with sklearn's LogisticRegression.
    """
    # Train the custom model
    capstone_model.train()

    # Train the sklearn model
    X_train, X_test = capstone_model.X_train, capstone_model.X_test
    y_train, y_test = capstone_model.y_train, capstone_model.y_test
    sklearn_model = LogisticRegression(max_iter=1000)
    sklearn_model.fit(X_train, y_train)

    # Predictions
    custom_predictions = capstone_model.predict(X_test)
    sklearn_predictions = sklearn_model.predict(X_test)

    # Accuracy comparison
    custom_accuracy = np.mean(custom_predictions == y_test) * 100
    sklearn_accuracy = accuracy_score(y_test, sklearn_predictions) * 100

    assert abs(custom_accuracy - sklearn_accuracy) < 5, (
        f"Accuracy difference is too large: Custom: {custom_accuracy:.2f}% vs "
        f"Sklearn: {sklearn_accuracy:.2f}%"
    )
