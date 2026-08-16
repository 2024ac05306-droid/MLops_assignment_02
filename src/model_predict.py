import json
import os
import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm  # <--- Step 1: Add tqdm import

# Ensure project root directory is added to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import settings
from src.model_train import SimpleCNN
from utils import get_device, get_transforms, logger


def get_dvc_dataset_hash(dvc_file_path: str = "data/Preprocessed.dvc") -> str:
    """Reads the md5 hash from the DVC tracking file to record dataset lineage."""
    if os.path.exists(dvc_file_path):
        with open(dvc_file_path, "r") as f:
            for line in f:
                if "md5:" in line:
                    return line.split(":")[1].strip()
    return "untracked_local_dataset"


def predict(input_path: str = None):
    """Executes offline model inference on a target image or directory, logging results to MLflow and DVC."""
    device = get_device()
    logger.info(f"Using compute device for prediction: {device}")

    # 1. Setup MLflow Tracking
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    # 2. Setup Input Path & Preprocessing
    target_path = input_path or os.path.join(settings.processed_data_path, "test")
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Input path for prediction not found at '{target_path}'.")

    transforms_pipeline = get_transforms(image_size=settings.image_size, augment=False)
    class_names = ["Cat", "Dog"]

    # 3. Load Trained Model Weights
    model_path = settings.model_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Run 'python -m src.model_train' first!"
        )

    model = SimpleCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info(f"Loaded model weights from {model_path}")

    # 4. Gather Image Paths
    if os.path.isfile(target_path):
        image_paths = [target_path]
    else:
        image_paths = [
            os.path.join(dp, f)
            for dp, _, filenames in os.walk(target_path)
            for f in filenames
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

    logger.info(f"Running predictions on {len(image_paths)} images...")

    # 5. Batch Inference Loop with tqdm Progress Tracking
    results = []
    progress_bar = tqdm(image_paths, desc="Running Predictions", unit="img") # <--- Step 2: Wrap loop

    with torch.no_grad():
        for img_path in progress_bar:
            try:
                img = Image.open(img_path).convert("RGB")
                tensor = transforms_pipeline(img).unsqueeze(0).to(device)
                outputs = model(tensor)
                probs = F.softmax(outputs, dim=1).squeeze(0)

                confidence, pred_idx = torch.max(probs, dim=0)
                pred_label = class_names[pred_idx.item()]

                results.append({
                    "file_path": img_path,
                    "predicted_label": pred_label,
                    "confidence": round(confidence.item(), 4),
                    "probabilities": {
                        class_names[i]: round(probs[i].item(), 4) for i in range(len(class_names))
                    },
                })
            except Exception as e:
                logger.error(f"Failed to process image '{img_path}': {str(e)}")

    # 6. Save Predictions Output
    os.makedirs(settings.output_dir, exist_ok=True)
    predictions_file = os.path.join(settings.output_dir, "predictions.json")
    with open(predictions_file, "w") as f:
        json.dump(results, f, indent=4)

    # 7. Log Run & Artifacts to MLflow
    logger.info("Logging inference run to MLflow...")
    with mlflow.start_run(run_name="batch_inference_stage"):
        mlflow.log_params({
            "stage": "batch_inference",
            "model_path": model_path,
            "total_images_processed": len(results),
            "dvc_dataset_hash": get_dvc_dataset_hash(),  # DVC Integration
        })

        mlflow.log_artifact(predictions_file, artifact_path="inference_outputs")
        logger.info(f"Predictions saved to {predictions_file} and logged to MLflow!")

    return results


if __name__ == "__main__":
    predict()