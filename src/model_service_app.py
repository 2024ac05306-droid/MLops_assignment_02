import io
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import settings
from src.model_train import SimpleCNN
from utils import get_device, get_transforms, logger

# 1. Initialize FastAPI Application
app = FastAPI(
    title="Cat vs Dog Image Classification API",
    description="REST API for serving PyTorch SimpleCNN model predictions.",
    version="1.0.0",
)

# 2. Global State Setup
device = get_device()
transforms_pipeline = get_transforms(image_size=settings.image_size, augment=False)
model = None
CLASS_NAMES = ["Cat", "Dog"]  # PyTorch ImageFolder default alphabetical ordering


# 3. Startup Event: Load Model into Memory
@app.on_event("startup")
def load_model():
    """Loads PyTorch model state dictionary from settings.model_path on API startup."""
    global model
    model_path = settings.model_path
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(
            f"Model weights not found at {model_path}. Run 'python -m src.model_train' first!"
        )

    try:
        model = SimpleCNN(num_classes=len(CLASS_NAMES)).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        logger.info(f"Model loaded successfully from {model_path} onto {device}")
    except Exception as e:
        logger.error(f"Failed to load model state: {str(e)}")
        raise e


# 4. Pydantic Response Schemas
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


class PredictionResponse(BaseModel):
    filename: str
    predicted_label: str
    confidence: float
    probabilities: dict[str, float]


# 5. Endpoint 1: Health Check
@app.get("/health", response_model=HealthResponse, tags=["Health Check"])
def health_check():
    """Returns API health status, model initialization status, and compute device."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        device=str(device),
    )


# 6. Endpoint 2: Image Prediction
@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(file: UploadFile = File(...)):
    """Accepts an uploaded image file (JPEG/PNG) and returns predicted label and class probabilities."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Please upload an image file.",
        )

    if model is None:
        raise HTTPException(status_code=500, detail="Model is not initialized.")

    try:
        # Read image file bytes & convert to RGB PIL Image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Apply preprocessing transforms (224x224 resize, normalization)
        input_tensor = transforms_pipeline(image).unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1).squeeze(0)

        # Map predictions to class labels
        confidence, pred_idx = torch.max(probs, dim=0)
        predicted_label = CLASS_NAMES[pred_idx.item()]
        probabilities_dict = {
            CLASS_NAMES[i]: round(probs[i].item(), 4) for i in range(len(CLASS_NAMES))
        }

        logger.info(
            f"Prediction: '{predicted_label}' ({confidence.item():.4f}) for file '{file.filename}'"
        )

        return PredictionResponse(
            filename=file.filename,
            predicted_label=predicted_label,
            confidence=round(confidence.item(), 4),
            probabilities=probabilities_dict,
        )

    except Exception as e:
        logger.error(f"Error during inference for '{file.filename}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")