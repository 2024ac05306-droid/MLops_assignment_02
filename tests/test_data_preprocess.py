import os
import sys
from pathlib import Path
from PIL import Image
import pytest

# Ensure project root is in sys.path for test discovery
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_preprocess import preprocess_single_image


def test_preprocess_single_image(tmp_path):
    """
    Tests that preprocess_single_image correctly converts non-standard images
    (e.g., RGBA, 100x150) into 224x224 3-channel RGB JPEG format.
    """
    # 1. Setup paths using pytest's temporary path fixture
    dummy_input_path = tmp_path / "test_input.png"
    dummy_output_path = tmp_path / "test_output.jpg"

    # 2. Create a non-standard 100x150 RGBA image
    img = Image.new("RGBA", (100, 150), color=(255, 0, 0, 128))
    img.save(dummy_input_path)

    # 3. Run the preprocessing function
    success = preprocess_single_image(
        src_path=str(dummy_input_path),
        dest_path=str(dummy_output_path),
        target_size=224,
    )

    # 4. Assertions
    assert success is True
    assert os.path.exists(dummy_output_path)

    # Verify processed image properties
    with Image.open(dummy_output_path) as processed_img:
        assert processed_img.size == (224, 224)  # Resized to 224x224
        assert processed_img.mode == "RGB"  # Converted to 3-channel RGB