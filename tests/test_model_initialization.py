import pytest
from your_package.module1 import ModelInitializer


def test_model_initialization():
    model = ModelInitializer()
    assert model is not None  # Replace with more specific checks as needed
