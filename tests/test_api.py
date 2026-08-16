import json
import os
import sys
import time
import requests
from pathlib import Path

# Resolve project root (one directory up from tests/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# API Configuration
BASE_URL = "http://127.0.0.1:8000"
HEALTH_URL = f"{BASE_URL}/health"
PREDICT_URL = f"{BASE_URL}/predict"

# Set target image path relative to project root
DEFAULT_IMAGE_PATH = str(PROJECT_ROOT / "data" / "processed" / "test" / "Cat" / "1.jpg")