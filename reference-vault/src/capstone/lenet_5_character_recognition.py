import os
import numpy as np
from rich.console import Console
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

class CharacterRecognitionCapstone:
    """
    A capstone project for LeNet-5 using the MNIST dataset.
    https://www.kaggle.com/datasets/hojjatk/mnist-dataset
    """

    dataset: pd.DataFrame
    console: Console
    beta: np.ndarray
    train_loader: DataLoader
    test_loader: DataLoader

    def __init__(self) -> None:
        self.console = Console()
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.batch_size = 64
        self.num_epochs = 5
        self.learning_rate = 0.001

    def convert(self, imgf, labelf, outf, n):
        imgf = os.path.join(self.base_path, imgf)
        labelf = os.path.join(self.base_path, labelf)
        outf = os.path.join(self.base_path, outf)

        with open(imgf, "rb") as f, open(labelf, "rb") as l, open(outf, "w") as o:
            f.read(16)
            l.read(8)
            images = []

            for i in range(n):
                image = [ord(l.read(1))]
                for j in range(28 * 28):
                    image.append(ord(f.read(1)))
                images.append(image)

            for image in images:
                o.write(",".join(str(pix) for pix in image) + "\n")

    def load_data(self):
        """
        Load the training and test data from CSV files and prepare them for PyTorch.
        """
        train_file = os.path.join(self.base_path, 'data/mnist_train.csv')
        test_file = os.path.join(self.base_path, 'data/mnist_test.csv')

        if not os.path.exists(train_file):
            self.console.print(f"[bold red]Error:[/bold red] {train_file} not found.")
            return
        if not os.path.exists(test_file):
            self.console.print(f"[bold red]Error:[/bold red] {test_file} not found.")
            return

        train_data = pd.read_csv(train_file, header=None)
        test_data = pd.read_csv(test_file, header=None)

        X_train = train_data.iloc[:, 1:].values / 255.0
        y_train = train_data.iloc[:, 0].values
        X_test = test_data.iloc[:, 1:].values / 255.0
        y_test = test_data.iloc[:, 0].values

        X_train = torch.tensor(X_train, dtype=torch.float32).view(-1, 1, 28, 28)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_test = torch.tensor(X_test, dtype=torch.float32).view(-1, 1, 28, 28)
        y_test = torch.tensor(y_test, dtype=torch.long)

        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    def define_model(self):
        class LeNet5(nn.Module):
            def __init__(self):
                super(LeNet5, self).__init__()
                self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
                self.fc1 = nn.Linear(16 * 4 * 4, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)

            def forward(self, x):
                x = self.pool(torch.relu(self.conv1(x)))
                x = self.pool(torch.relu(self.conv2(x)))
                x = x.view(-1, 16 * 4 * 4)
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        return LeNet5()

    def train_model(self, model, criterion, optimizer):
        for epoch in range(self.num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in self.train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(self.train_loader)
            epoch_acc = 100 * correct / total
            self.console.print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    def evaluate_model(self, model):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_acc = 100 * correct / total
        self.console.print(f"Test Accuracy: {test_acc:.2f}%")

    def preprocess_image(self, image_path):
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        image = Image.open(image_path)
        image = transform(image)
        image = image.unsqueeze(0)
        return image

    def predict(self, model, image_paths):
        model.eval()
        predicted_digits = []

        for image_path in image_paths:
            image = self.preprocess_image(image_path)
            with torch.no_grad():
                output = model(image)
                _, predicted = torch.max(output, 1)
                predicted_digits.append(predicted.item())

        return predicted_digits

    def run(self) -> None:
        self.convert("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", "data/mnist_train.csv", 60000)
        self.convert("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", "data/mnist_test.csv", 10000)

        self.load_data()

        model = self.define_model()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        self.train_model(model, criterion, optimizer)
        self.evaluate_model(model)

        image_paths = [
            'src/capstone/examples/example_eight.png',
            'src/capstone/examples/nine_example.png',
            'src/capstone/examples/two_example.png',
            'src/capstone/examples/four_example.png'
        ]
        predicted_digits = self.predict(model, image_paths)
        self.console.print(f"Predicted Digits: {predicted_digits}")
