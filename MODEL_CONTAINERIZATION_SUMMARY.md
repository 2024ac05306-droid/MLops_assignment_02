# Model Containerization - Summary

## Overview
This is a **Cat vs Dog Image Classification API** built with PyTorch and FastAPI. The model is served via a containerized REST API.

---

## ✅ API Endpoints

### 1. Health Check Endpoint
- **Endpoint:** `GET /health`
- **Location:** Line 93 in `src/model_service_app.py`
- **Response:** Returns API health status, model initialization status, and compute device
- **Response Schema:**
  ```json
  {
    "status": "healthy",
    "model_loaded": true,
    "device": "cpu"
  }
  ```

### 2. Prediction Endpoint
- **Endpoint:** `POST /predict`
- **Location:** Line 105 in `src/model_service_app.py`
- **Input:** Accepts **image file uploads** (JPEG/PNG format)
- **Not JSON-based:** Accepts `UploadFile` parameter, not JSON payload
- **Response Schema:**
  ```json
  {
    "filename": "sample_image.jpg",
    "predicted_label": "Cat",
    "confidence": 0.9523,
    "probabilities": {
      "Cat": 0.9523,
      "Dog": 0.0477
    }
  }
  ```

---

## 🐳 Container Configuration

### Dockerfile Setup
✅ **Uses Python 3.11-slim** — Efficient lightweight base image

✅ **System Dependencies** — Installs build-essential for PyTorch compilation

✅ **Working Directory** — Sets `/app` as work directory

✅ **Dependencies Installation** — Installs all requirements from `requirements.txt`

✅ **Non-root User** — Runs as `appuser` (UID 1000) for security

✅ **Port Exposure** — Exposes port 8000

✅ **Health Check** — Configured with 30s interval, 10s timeout, 5s startup grace period

✅ **Uvicorn Server** — Runs `src.model_service_app:app` on `0.0.0.0:8000`

---

## 📦 Dependencies

**Core ML Framework:**
- torch==2.2.1
- torchvision==0.17.1
- pillow==10.2.0 (image processing)
- numpy==1.26.4
- scikit-learn==1.7.0

**API & Web Server:**
- fastapi==0.116.1
- uvicorn[standard]==0.35.0
- pydantic==2.11.7

**MLOps & Experiment Tracking:**
- mlflow==2.11.1
- dvc==3.48.0
- dvc-s3==3.2.0

**Testing & Quality:**
- pytest==8.4.1
- httpx==0.27.0

**Monitoring:**
- prometheus-client==0.20.0
- python-json-logger==2.0.7

---

## 🎯 Model Details

**Model Type:** SimpleCNN (PyTorch)

**Classes:** 2 (Cat, Dog)

**Input:** Image file (JPEG/PNG)

**Output:**
- Predicted label (Cat or Dog)
- Confidence score (0-1)
- Per-class probabilities

**Device Support:** CPU/GPU (auto-detected)

---

## 🚀 Running the Container

```bash
# Build the Docker image
docker build -t mlops-cat-dog-api .

# Run the container
docker run -p 8000:8000 mlops-cat-dog-api

# Test health check
curl http://localhost:8000/health

# Test prediction (with image file)
curl -X POST "http://localhost:8000/predict" \
  -F "file=@path/to/image.jpg"
```

---

## 📝 Testing

**Note:** No JSON sample file exists. To test the `/predict` endpoint, use an actual image file (JPG/PNG).

Example test:
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@sample_cat.jpg"
```

---

## ⚠️ Important Notes

1. **Not Heart Disease Prediction** — This is an image classification API, not medical prediction
2. **Image Input Required** — `/predict` accepts file uploads, not JSON payloads
3. **Model Must Be Trained** — Run `python -m src.model_train` before starting the API
4. **DVC Integration** — Model artifacts are managed via DVC for versioning
5. **Production Ready** — Includes health checks, logging, error handling, and non-root execution
