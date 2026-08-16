import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets

# Add root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import settings
from utils import logger, set_seed, get_device, get_transforms, save_model

set_seed(settings.random_seed)
device = get_device()
print(f"[STATUS] Using device: {device}", flush=True)

def get_dvc_dataset_hash(dvc_file_path: str = "data/Preprocessed.dvc") -> str:
    if os.path.exists(dvc_file_path):
        with open(dvc_file_path, "r") as f:
            for line in f:
                if "md5:" in line:
                    return line.split(":")[1].strip()
    return "untracked_local_dataset"

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_dataloaders():
    print("[STATUS] Loading datasets and applying transformations...", flush=True)
    train_transforms = get_transforms(image_size=settings.image_size, augment=True)
    val_transforms = get_transforms(image_size=settings.image_size, augment=False)

    train_dir = os.path.join(settings.processed_data_path, "train")
    val_dir = os.path.join(settings.processed_data_path, "val")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(
            f"Dataset splits not found in {settings.processed_data_path}. Run src/data_prep.py first!"
        )

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=settings.batch_size, shuffle=False, num_workers=2)

    print(f"[STATUS] Datasets loaded successfully. Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}", flush=True)
    return train_loader, val_loader, train_dataset.classes

def plot_and_save_metrics(train_losses, val_losses, train_accs, val_accs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    epochs_range = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs_range, train_losses, label="Train Loss", color="blue")
    ax1.plot(epochs_range, val_losses, label="Val Loss", color="orange")
    ax1.set_title("Loss Curves")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs_range, train_accs, label="Train Acc", color="blue")
    ax2.plot(epochs_range, val_accs, label="Val Acc", color="orange")
    ax2.set_title("Accuracy Curves")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)

    plot_path = os.path.join(output_dir, "training_curves.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def plot_and_save_confusion_matrix(y_true, y_pred, class_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title("Validation Confusion Matrix")

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    return cm_path

def train():
    print(f"\n==================================================", flush=True)
    print(f"[START] Initializing training pipeline on device: {device}", flush=True)
    print(f"==================================================\n", flush=True)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    train_loader, val_loader, class_names = get_dataloaders()
    model = SimpleCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=settings.learning_rate)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    logger.info("Starting MLflow tracking run...")
    with mlflow.start_run() as run:
        mlflow.log_params({
            "model_architecture": settings.model_type,
            "epochs": settings.epochs,
            "batch_size": settings.batch_size,
            "learning_rate": settings.learning_rate,
            "image_size": settings.image_size,
            "optimizer": "Adam",
            "device": str(device),
            "random_seed": settings.random_seed,
            "dvc_dataset_hash": get_dvc_dataset_hash(),
        })

        best_val_acc = 0.0
        total_batches = len(train_loader)

        for epoch in range(1, settings.epochs + 1):
            print(f"\n--------------------------------------------------", flush=True)
            print(f"[EPOCH START] Epoch {epoch}/{settings.epochs}", flush=True)
            print(f"--------------------------------------------------", flush=True)
            
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            
            for step, (images, labels) in enumerate(train_loader, 1):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                # Batch progress update every 20% of batches or every 20 batches
                if step % max(1, total_batches // 5) == 0 or step == total_batches:
                    current_loss = running_loss / total
                    current_acc = (correct / total) * 100
                    print(
                        f"  -> Batch [{step}/{total_batches}] | "
                        f"Current Train Loss: {current_loss:.4f} | Current Train Acc: {current_acc:.2f}%",
                        flush=True
                    )

            epoch_train_loss = running_loss / total
            epoch_train_acc = (correct / total) * 100

            print(f"[STATUS] Running Validation for Epoch {epoch}...", flush=True)
            model.eval()
            val_running_loss, val_correct, val_total = 0.0, 0, 0
            all_preds, all_targets = [], []

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_running_loss += loss.item() * images.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(labels.cpu().numpy())

            epoch_val_loss = val_running_loss / val_total
            epoch_val_acc = (val_correct / val_total) * 100

            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)
            train_accs.append(epoch_train_acc)
            val_accs.append(epoch_val_acc)

            mlflow.log_metrics({
                "train_loss": epoch_train_loss,
                "train_accuracy": epoch_train_acc,
                "val_loss": epoch_val_loss,
                "val_accuracy": epoch_val_acc,
            }, step=epoch)

            summary_msg = (
                f"[EPOCH END {epoch}/{settings.epochs}] "
                f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
                f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%"
            )
            print(summary_msg, flush=True)
            logger.info(summary_msg)

            if epoch_val_acc > best_val_acc:
                print(f"  *** New Best Validation Accuracy: {epoch_val_acc:.2f}% (Previous: {best_val_acc:.2f}%) ***", flush=True)
                best_val_acc = epoch_val_acc

        print(f"\n[STATUS] Saving model state dict to {settings.model_path}...", flush=True)
        save_model(model, settings.model_path)
        print("[STATUS] Model saved successfully!", flush=True)

        print("[STATUS] Generating evaluation plots...", flush=True)
        curves_path = plot_and_save_metrics(
            train_losses, val_losses, train_accs, val_accs, settings.output_dir
        )
        cm_path = plot_and_save_confusion_matrix(
            all_targets, all_preds, class_names, settings.output_dir
        )

        mlflow.log_artifact(curves_path, artifact_path="plots")
        mlflow.log_artifact(cm_path, artifact_path="plots")
        mlflow.pytorch.log_model(model, artifact_path="model")

        print(f"\n==================================================", flush=True)
        print(f"[FINISHED] MLflow Run completed successfully! Run ID: {run.info.run_id}", flush=True)
        print(f"==================================================\n", flush=True)
        logger.info(f"MLflow Run completed successfully! Run ID: {run.info.run_id}")

if __name__ == "__main__":
    train()