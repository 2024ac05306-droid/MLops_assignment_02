import io
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image
from pydantic import BaseModel

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import settings
from src.model_train import SimpleCNN
from utils import get_device, get_transforms, logger

# Global State Setup
device = get_device()
transforms_pipeline = get_transforms(image_size=settings.image_size, augment=False)
model = None
CLASS_NAMES = ["Cat", "Dog"]  # PyTorch ImageFolder default alphabetical ordering


# 1. Define Lifespan Manager (Replaces @app.on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads PyTorch model state dictionary from settings.model_path on API startup."""
    global model
    model_path = settings.model_path
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(
            f"Model weights not found at {model_path}. Run 'python -m src.model_train' first!"
        )

    try:
        logger.info(f"Loading PyTorch model weights from: {model_path}")
        model = SimpleCNN(num_classes=len(CLASS_NAMES)).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        logger.info(f"Model successfully loaded onto compute device: {device}")
    except Exception as e:
        logger.error(f"Failed to load model state: {str(e)}")
        raise e

    yield  # The app serves requests while paused here

    # Code after yield executes on application shutdown (optional cleanup)
    model = None


# 2. Initialize FastAPI Application with Lifespan
app = FastAPI(
    title="Cat vs Dog Image Classification API",
    description="REST API for serving PyTorch SimpleCNN model predictions.",
    version="1.0.0",
    lifespan=lifespan,
)


# Middleware for Request Timing & Progress Tracking
@app.middleware("http")
async def log_request_progress(request: Request, call_next):
    start_time = time.time()
    logger.info(f"Incoming request: {request.method} {request.url.path}")
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    logger.info(
        f"Completed {request.method} {request.url.path} - Status: {response.status_code} - Latency: {process_time:.2f}ms"
    )
    return response


# 3. Pydantic Response Schemas
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


class PredictionResponse(BaseModel):
    filename: str
    predicted_label: str
    confidence: float
    probabilities: dict[str, float]


# 4. Endpoint 1: Health Check
@app.get("/health", response_model=HealthResponse, tags=["Health Check"])
def health_check():
    """Returns API health status, model initialization status, and compute device."""
    logger.info("Executing health check...")
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        device=str(device),
    )


# 5. Endpoint 2: Image Prediction
@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(file: UploadFile = File(...)):
    """Accepts an uploaded image file (JPEG/PNG) and returns predicted label and class probabilities."""
    if not file.content_type.startswith("image/"):
        logger.warning(f"Rejected invalid file type: '{file.content_type}'")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Please upload an image file.",
        )

    if model is None:
        logger.error("Inference attempted before model was initialized.")
        raise HTTPException(status_code=500, detail="Model is not initialized.")

    try:
        # Step 1: Read image bytes & convert to RGB
        logger.info(f"[1/3] Processing uploaded file: '{file.filename}'")
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Step 2: Apply preprocessing transforms
        logger.info(f"[2/3] Applying image preprocessing pipeline (size={settings.image_size})...")
        input_tensor = transforms_pipeline(image).unsqueeze(0).to(device)

        # Step 3: Run model forward pass
        logger.info("[3/3] Running model inference...")
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1).squeeze(0)

        # Process output probabilities
        confidence, pred_idx = torch.max(probs, dim=0)
        predicted_label = CLASS_NAMES[pred_idx.item()]
        probabilities_dict = {
            CLASS_NAMES[i]: round(probs[i].item(), 4) for i in range(len(CLASS_NAMES))
        }

        logger.info(
            f"Prediction completed: Label='{predicted_label}' (Confidence={confidence.item():.4f})"
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