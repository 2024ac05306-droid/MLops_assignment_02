import json
import os
import sys
from pathlib import Path

import pytest
from PIL import Image

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import settings
from src.model_predict import predict


@pytest.fixture
def dummy_image(tmp_path):
    """Creates a temporary sample RGB image for unit testing inference."""
    img_dir = tmp_path / "test_samples"
    img_dir.mkdir()
    img_path = img_dir / "sample_cat.jpg"

    # Create a synthetic 224x224 RGB image
    img = Image.new("RGB", (224, 224), color="blue")
    img.save(img_path)
    return str(img_path)


def test_predict_single_image_structure(dummy_image):
    """Verifies that predict() returns expected schema for a valid image."""
    if not os.path.exists(settings.model_path):
        pytest.skip(f"Model file missing at '{settings.model_path}'. Run training first.")

    results = predict(input_path=dummy_image)

    assert isinstance(results, list)
    assert len(results) == 1

    prediction = results[0]
    assert "file_path" in prediction
    assert "predicted_label" in prediction
    assert "confidence" in prediction
    assert "probabilities" in prediction

    assert prediction["predicted_label"] in ["Cat", "Dog"]
    assert 0.0 <= prediction["confidence"] <= 1.0
    assert set(prediction["probabilities"].keys()) == {"Cat", "Dog"}


def test_predictions_json_artifact_generated(dummy_image):
    """Verifies that predictions.json artifact is generated in output directory."""
    if not os.path.exists(settings.model_path):
        pytest.skip(f"Model file missing at '{settings.model_path}'. Run training first.")

    predict(input_path=dummy_image)

    predictions_file = os.path.join(settings.output_dir, "predictions.json")
    assert os.path.exists(predictions_file)

    with open(predictions_file, "r") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) >= 1


def test_predict_invalid_path_raises_error():
    """Ensures FileNotFoundError is raised when input path does not exist."""
    invalid_path = os.path.join("data", "non_existent_folder_999")
    with pytest.raises(FileNotFoundError):
        predict(input_path=invalid_path)