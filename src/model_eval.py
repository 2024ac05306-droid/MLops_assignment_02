import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torchvision import datasets

# Ensure project root directory is added to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import settings
from src.model_train import SimpleCNN
from utils import get_device, get_transforms, logger, set_seed


def get_dvc_dataset_hash(dvc_file_path: str = "data/Preprocessed.dvc") -> str:
    """Reads the md5 hash from the DVC tracking file for dataset lineage."""
    if os.path.exists(dvc_file_path):
        with open(dvc_file_path, "r") as f:
            for line in f:
                if "md5:" in line:
                    return line.split(":")[1].strip()
    return "untracked_local_dataset"


def evaluate():
    set_seed(settings.random_seed)
    device = get_device()
    logger.info(f"Using compute device for evaluation: {device}")

    # Set MLflow tracking URI & experiment
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    test_dir = os.path.join(settings.processed_data_path, "test")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(
            f"Test directory not found at {test_dir}. Run src/data_prep.py first!"
        )

    # Load Data
    test_transforms = get_transforms(image_size=settings.image_size, augment=False)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)
    test_loader = DataLoader(
        test_dataset, batch_size=settings.batch_size, shuffle=False, num_workers=0
    )
    class_names = test_dataset.classes

    # Load Model Weights
    model_path = settings.model_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Run src/model_train.py first!"
        )

    model = SimpleCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Evaluation Loop
    all_preds, all_targets = [], []
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    # Calculate Standard Metrics
    test_loss = running_loss / len(test_dataset)
    accuracy = accuracy_score(all_targets, all_preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro"
    )

    logger.info(
        f"Test Loss: {test_loss:.4f} | Test Acc: {accuracy:.2f}% | "
        f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}"
    )

    # Log to MLflow
    with mlflow.start_run(run_name="model_evaluation_stage"):
        # Log Hyperparameters + DVC Hash
        mlflow.log_params({
            "stage": "evaluation",
            "eval_batch_size": settings.batch_size,
            "model_path": model_path,
            "test_sample_count": len(test_dataset),
            "dvc_dataset_hash": get_dvc_dataset_hash(),  # DVC Integration
        })

        # Log Metrics
        mlflow.log_metrics({
            "test_loss": test_loss,
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1_score": f1,
        })

        # Log Confusion Matrix Plot
        os.makedirs(settings.output_dir, exist_ok=True)
        cm = confusion_matrix(all_targets, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap="Blues")
        ax.set_title("Test Set Confusion Matrix")

        cm_path = os.path.join(settings.output_dir, "test_confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()

        mlflow.log_artifact(cm_path, artifact_path="evaluation_artifacts")

        # Save metrics.json for DVC tracking
        metrics_dict = {
            "test_loss": test_loss,
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1_score": f1,
        }
        json_path = os.path.join(settings.output_dir, "metrics.json")
        with open(json_path, "w") as f:
            json.dump(metrics_dict, f, indent=4)

        mlflow.log_artifact(json_path, artifact_path="evaluation_artifacts")
        logger.info("MLflow & DVC evaluation metrics saved successfully!")


if __name__ == "__main__":
    evaluate()