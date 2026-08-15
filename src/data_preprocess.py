import glob
import json
import os
import sys
from pathlib import Path
import mlflow
from PIL import Image
from sklearn.model_selection import train_test_split

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import settings
from utils import ensure_directories, logger, set_seed


def get_dvc_raw_hash(dvc_file_path: str = "data/raw.dvc") -> str:
    """Reads the md5 hash from raw data DVC tracking file for MLflow lineage."""
    if os.path.exists(dvc_file_path):
        with open(dvc_file_path, "r") as f:
            for line in f:
                if "md5:" in line:
                    return line.split(":")[1].strip()
    return "untracked_raw_dataset"


def preprocess_single_image(
    src_path: str, dest_path: str, target_size: int = 224
) -> bool:
    """Loads an image, converts it to 3-channel RGB, resizes to target_size, and saves as JPEG."""
    try:
        with Image.open(src_path) as img:
            img_rgb = img.convert("RGB")
            img_resized = img_rgb.resize(
                (target_size, target_size), Image.Resampling.BILINEAR
            )

            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            img_resized.save(dest_path, format="JPEG")
            return True
    except Exception as e:
        logger.warning(f"Skipping unreadable or corrupted image {src_path}: {e}")
        return False


def prepare_data():
    """Reads raw images, applies reproducible 80/10/10 split, and saves 224x224 RGB images."""
    set_seed(settings.random_seed)
    ensure_directories([settings.processed_data_path, settings.output_dir])

    # Set up MLflow tracking
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    raw_dir = settings.data_path
    processed_dir = settings.processed_data_path
    img_size = settings.image_size
    classes = ["cats", "dogs"]
    split_counts = {"train": 0, "val": 0, "test": 0}

    logger.info("Starting data preparation with MLflow tracking...")

    with mlflow.start_run(run_name="data_preparation_stage"):
        # 1. Log Data Parameters & DVC Hash
        mlflow.log_params({
            "stage": "data_prep",
            "image_size": img_size,
            "train_size": settings.train_size,
            "val_size": settings.val_size,
            "test_size": settings.test_size,
            "random_seed": settings.random_seed,
            "raw_dir": raw_dir,
            "processed_dir": processed_dir,
            "dvc_raw_hash": get_dvc_raw_hash(),
        })

        for cls in classes:
            cls_path = os.path.join(raw_dir, cls)
            if not os.path.exists(cls_path):
                cls_path = os.path.join(raw_dir, cls.capitalize()[:-1])  # 'Cat' or 'Dog'

            if not os.path.exists(cls_path):
                logger.warning(
                    f"Could not find raw folder for class '{cls}' at {cls_path}"
                )
                continue

            # Gather all valid image file paths
            image_files = []
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                image_files.extend(glob.glob(os.path.join(cls_path, ext)))
                image_files.extend(glob.glob(os.path.join(cls_path, ext.upper())))

            if not image_files:
                logger.warning(f"No images found for class '{cls}'.")
                continue

            # 80% train, 20% temp (val + test) split
            train_files, temp_files = train_test_split(
                image_files,
                test_size=(settings.val_size + settings.test_size),
                random_state=settings.random_seed,
            )

            # 50/50 split on temp set (10% val, 10% test total)
            val_files, test_files = train_test_split(
                temp_files, test_size=0.5, random_state=settings.random_seed
            )

            splits = {"train": train_files, "val": val_files, "test": test_files}

            for split_name, files in splits.items():
                logger.info(
                    f"Processing '{split_name}' split for '{cls}' ({len(files)} files)..."
                )
                successful_count = 0
                for idx, src_file in enumerate(files):
                    dest_file = os.path.join(
                        processed_dir, split_name, cls, f"{cls}_{idx}.jpg"
                    )
                    if preprocess_single_image(src_file, dest_file, target_size=img_size):
                        successful_count += 1

                # Correctly increment total processed counts
                split_counts[split_name] += successful_count
                logger.info(
                    f"Successfully processed {successful_count}/{len(files)} images for {split_name}/{cls}"
                )

        # 2. Log Dataset Metrics to MLflow
        mlflow.log_metrics({
            "num_train_images": split_counts["train"],
            "num_val_images": split_counts["val"],
            "num_test_images": split_counts["test"],
            "total_processed_images": sum(split_counts.values()),
        })

        # 3. Save & Log Summary Artifact
        summary_path = os.path.join(settings.output_dir, "dataset_summary.json")
        with open(summary_path, "w") as f:
            json.dump(split_counts, f, indent=4)

        mlflow.log_artifact(summary_path, artifact_path="data_summary")
        logger.info("Data pre-processing completed and logged to MLflow successfully!")


if __name__ == "__main__":
    prepare_data()