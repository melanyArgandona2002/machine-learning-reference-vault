import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix

class TumorClassificationCapstone:
    """
    A capstone project for tumor classification using the breast cancer dataset.
    https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
    """

    def __init__(self):
        """
        Initializes the project by loading the dataset and preparing the data.
        """
        
        self.dataset = load_breast_cancer(as_frame=True)
        self.X = self.dataset.data
        self.y = self.dataset.target
        
        self.X_scaled = (self.X - self.X.mean()) / self.X.std()
        
        train_size = int(0.8 * len(self.X_scaled))
        self.X_train, self.X_test = self.X_scaled[:train_size], self.X_scaled[train_size:]
        self.y_train, self.y_test = self.y[:train_size], self.y[train_size:]
        
        self.theta = np.zeros(self.X_train.shape[1] + 1)
        self.learning_rate = 0.1
        self.iterations = 1000

    def sigmoid(self, z):
        """
        Sigmoid activation function.
        """
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, X, y, theta):
        """
        Calculates the cost function (cross-entropy loss).
        """
        m = len(y)
        predictions = self.sigmoid(np.dot(X, theta))
        cost = -1/m * np.sum(y * np.log(predictions + 1e-15) + (1 - y) * np.log(1 - predictions + 1e-15))
        return cost

    def gradient_descent(self, X, y, theta, learning_rate, iterations):
        """
        Executes the gradient descent algorithm to adjust the parameters.
        """
        m = len(y)
        cost_history = []
        
        for _ in range(iterations):
            predictions = self.sigmoid(np.dot(X, theta))
            theta -= (learning_rate/m) * np.dot(X.T, predictions - y)
            cost_history.append(self.compute_cost(X, y, theta))
        
        return theta, cost_history

    def train(self):
        """
        Trains the model using gradient descent.
        """
        X_train_ones = np.c_[np.ones((self.X_train.shape[0], 1)), self.X_train]  # Add bias term
        self.theta, self.cost_history = self.gradient_descent(
            X_train_ones, self.y_train, self.theta, self.learning_rate, self.iterations
        )

    def predict(self, X):
        """
        Predicts whether a tumor is malignant (1) or benign (0).
        """
        X_ones = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
        predictions = self.sigmoid(np.dot(X_ones, self.theta))
        return [1 if p >= 0.5 else 0 for p in predictions]

    def evaluate(self):
        """
        Evaluates the model using test data and calculates the accuracy and confusion matrix.
        """
        predictions = self.predict(self.X_test)
        
        accuracy = np.mean(predictions == self.y_test) * 100
        print(f'Model Accuracy: {accuracy:.2f}%')
        
        cm = confusion_matrix(self.y_test, predictions)
        print("Confusion Matrix:")
        print(cm)

    def plot_cost_history(self):
        """
        Plots the cost function history over iterations.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.cost_history)), self.cost_history, label='Cost')
        plt.title('Cost Function History')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.legend()
        plt.grid()
        plt.savefig("cost_history.png")
        plt.close()
        # plt.show()

    def plot_data_distribution(self):
        """
        Plots the distribution of classes in the dataset.
        """
        plt.figure(figsize=(8, 5))
        sns.countplot(x=self.y, palette='viridis')
        plt.title('Distribution of Tumor Classes')
        plt.xlabel('Class (0: Benign, 1: Malignant)')
        plt.ylabel('Count')
        plt.show()

    def run(self):
        """
        Trains and evaluates the model.
        """
        self.plot_data_distribution()
        self.train()
        self.plot_cost_history()
        self.evaluate()
