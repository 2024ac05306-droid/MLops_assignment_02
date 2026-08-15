# MLops_assignment_02
## MLops assignment 

Design and implement an end-to-end MLOps pipeline for model building, artifact/image
creation, packaging, containerization, and CI/CD-based deployment using open-source tools

**Use case** : Binary image classification (Cats vs Dogs) for a pet adoption platform.

**Dataset** : Cats and Dogs classification dataset (CATS and Dogs binary classification dataset from Kaggle)

Pre-process to 224x224 RGB images for standard CNNs

------------------------------------------------------------------------------------------------------------------

## Open-Source Tools Stack

| Module | Requirement | Recommended Tool | Roles & Justification |
|---|---|---|---|
| M1 | Code & Data Versioning | Git & DVC (with local or S3 remote) | Git tracks source code; DVC versions raw and 224×224 pre-processed dataset splits without bloating Git. |
| M1 | Model Building | PyTorch / Torchvision  | Trains baseline CNN/MobileNetV2 on 224×224 RGB images and serializes the model (`.pt` / `.h5`). |
| M1 | Experiment Tracking | MLflow | Logs hyperparameters, train/validation loss curves, confusion matrices, and model artifacts locally or to a remote server. |
| M2 | REST API Service | FastAPI + Uvicorn | Provides fast, auto-documented endpoints (`/health` and `/predict`) returning class probabilities. |
| M2 | Packaging & Container | Pip (`requirements.txt`) & Docker | Environment version-pinning and containerization for reproducible deployment. |
| M3 | Automated Testing | pytest | Runs unit tests for image pre-processing (resize/normalize) and the model inference pipeline. |
| M3 | CI Pipeline & Registry | GitHub Actions & Docker Hub (or GHCR) | Automatically builds, tests, builds the Docker image, and pushes it to the registry on push/merge. |
| M4 | Deployment Target | Docker Compose or Kubernetes (kind / minikube) | Declarative orchestration using `docker-compose.yml` or Kubernetes `deployment.yaml` + `service.yaml`. |
| M4 | CD Flow & Smoke Tests | GitHub Actions + cURL / Python script | Automates service pull/update on the main branch and runs post-deployment smoke tests against `/health` and `/predict`. |
| M5 | Logging & Metrics | FastAPI Middleware + Prometheus Client | Captures request latency, hit count, prediction payloads, and logging for post-deployment tracking. |


