import numpy as np
from rich.console import Console
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class MpgRegressionCapstone:
    """
    A capstone project for MPG regression using the mpg dataset.
    https://www.kaggle.com/code/devanshbesain/exploration-and-analysis-auto-mpg
    """

    dataset: pd.DataFrame
    console: Console
    beta: np.ndarray

    def __init__(self) -> None:
        self.console = Console()
        self.dataset = sns.load_dataset("mpg").dropna()

    def preprocess(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the data: separate features and labels, and encode categorical variables.
        """

        self.console.print("[bold yellow]Preprocessing data...[/bold yellow]")

        self.dataset = pd.get_dummies(self.dataset, columns=["origin"], drop_first=True)

        X = self.dataset.drop(columns=["mpg", "name"])
        y = self.dataset["mpg"]  # Etiqueta

        X = X.astype(float)

        X = np.hstack([np.ones((X.shape[0], 1)), X.values])
        return X, y.values

    def split_data(self, X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8):
        """
        Divide the data into training and testing sets.
        """

        self.console.print("[bold yellow]Splitting data into training and testing sets...[/bold yellow]")
        split_idx = int(train_ratio * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        return X_train, X_test, y_train, y_test

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Train the regression model using the Normal Equation.
        """

        self.console.print("[bold yellow]Training the model...[/bold yellow]")
        X_transpose = X.T
        self.beta = np.linalg.inv(X_transpose @ X) @ X_transpose @ y
        self.console.print("[bold green]Model coefficients (β):[/bold green]", self.beta)
        return self.beta

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the learned coefficients.
        """

        return X @ self.beta

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Evaluate the model using MAE, MSE, and R² metrics.
        """

        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred) ** 2)
        r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

        self.console.print(f"[bold cyan]Mean Absolute Error (MAE):[/bold cyan] {mae:.2f}")
        self.console.print(f"[bold cyan]Mean Squared Error (MSE):[/bold cyan] {mse:.2f}")
        self.console.print(f"[bold cyan]R² Score:[/bold cyan] {r2:.2f}")

    def predict_new(self, new_data: dict) -> float:
        """
        Predict the mpg for a new data point.
        """

        self.console.print("[bold yellow]Predicting mpg for new data...[/bold yellow]")

        new_df = pd.DataFrame([new_data])

        new_df = pd.get_dummies(new_df, columns=["origin"], drop_first=True)

        for col in self.dataset.columns:
            if col not in new_df:
                new_df[col] = 0

        new_df = new_df.drop(columns=["mpg", "name"], errors="ignore")
        X_new = np.hstack([np.ones((new_df.shape[0], 1)), new_df.values])

        y_pred = self.predict(X_new)
        self.console.print(f"[bold green]Predicted mpg:[/bold green] {y_pred[0]:.2f}")
        return y_pred[0]

    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Plot a scatter plot of true vs. predicted values.
        """
        self.console.print("[bold yellow]Plotting predictions...[/bold yellow]")

        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.7, color="blue", label="Predictions")
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color="red", linestyle="--", label="Ideal Fit")
        plt.xlabel("True MPG")
        plt.ylabel("Predicted MPG")
        plt.title("True vs. Predicted MPG")
        plt.legend()
        plt.grid(True)
        # plt.show()
        plt.savefig("predictions_plot.png")
        self.console.print("[bold green]Plot saved as 'predictions_plot.png'[/bold green]")

    def run(self) -> None:
        """
        Run the entire pipeline: preprocess, train, and evaluate.
        """

        self.console.print("Vehicle Dataset Overview", style="bold")
        self.console.print(self.dataset.head())

        self.console.print("\n")

        self.console.print("DataFrame Structure and Summary of the Vehicle Dataset", style="bold")
        self.dataset.info()

        self.console.print("\n")

        self.console.print("[bold yellow]Starting MPG regression pipeline...[/bold yellow]")

        X, y = self.preprocess()
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        self.console.print("\n")

        self.fit(X_train, y_train)
        y_pred = self.predict(X_test)
        self.evaluate(y_test, y_pred)

        self.console.print("\n")

        # Plot predictions
        self.plot_predictions(y_test, y_pred)

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

        self.predict_new(new_data)