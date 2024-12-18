import pytest
import torch
import os
from unittest.mock import patch, MagicMock
from PIL import Image
from torchvision import transforms
from capstone.lenet_5_character_recognition import CharacterRecognitionCapstone

@pytest.fixture
def capstone_project():
    return CharacterRecognitionCapstone()

def test_define_model(capstone_project):
    model = capstone_project.define_model()
    
    # Check that the model is of type LeNet5
    assert isinstance(model, torch.nn.Module)
    assert hasattr(model, 'conv1')
    assert hasattr(model, 'conv2')
    assert hasattr(model, 'fc1')
    assert hasattr(model, 'fc2')
    assert hasattr(model, 'fc3')

def test_preprocess_image(capstone_project):
    # Mocking PIL Image and torchvision transforms
    mock_image = MagicMock(spec=Image)
    mock_transform = MagicMock(spec=transforms.Compose)

    with patch("PIL.Image.open", return_value=mock_image):
        # Assuming the transform object is used inside preprocess_image method
        with patch.object(transforms, 'Compose', return_value=mock_transform):
            preprocessed_image = capstone_project.preprocess_image("path/to/image.png")
            assert preprocessed_image is not None

def test_predict(capstone_project):
    # Mocking the model prediction
    model = capstone_project.define_model()
    image_paths = ['path1.png', 'path2.png']

    with patch.object(model, 'eval', return_value=None):
        with patch.object(capstone_project, 'preprocess_image', return_value=torch.zeros(1, 1, 28, 28)):
            predicted_digits = capstone_project.predict(model, image_paths)
            assert isinstance(predicted_digits, list)
            assert len(predicted_digits) == len(image_paths)
            assert all(isinstance(digit, int) for digit in predicted_digits)
