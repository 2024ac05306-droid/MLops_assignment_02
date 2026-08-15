import logging
import os
import random
import numpy as np
import torch
from torchvision import transforms

# Initialize Logger
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("MLOps Pipeline")

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_transforms(image_size: int = 224, augment: bool = False):
    if augment:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def save_model(model: torch.nn.Module, target_path: str):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    torch.save(model.state_dict(), target_path)
    logger.info(f"Model saved successfully to {target_path}")

def ensure_directories(dirs: list):
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)