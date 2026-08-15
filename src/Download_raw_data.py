import os
import sys
from pathlib import Path
import urllib.request
import zipfile


# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings
from utils import logger

DATASET_URL = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"

def download_raw_dataset():
    target_dir = os.path.dirname(settings.data_path)  # Resolves to ./data/raw
    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, "dataset.zip")
    
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    logger.info("Downloading Cats vs Dogs dataset...")
    urllib.request.urlretrieve(DATASET_URL, zip_path)
    
    logger.info(f"Extracting images to {target_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
        
    if os.path.exists(zip_path):
        os.remove(zip_path)
        
    logger.info(f"Dataset successfully saved and extracted to {target_dir}!")

if __name__ == "__main__":
    download_raw_dataset()