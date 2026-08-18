# Model Development & Experiment Tracking - Summary

## Objective
Build a baseline model, track experiments, and version all artifacts for a Cat vs Dog binary image classification task.

---

## 1. Data & Code Versioning

### ✅ Git for Source Code Versioning
- **Repository**: GitHub repository with public access
- **Structure**:
  ```
  .
  ├── src/
  │   ├── model_train.py          # Training pipeline
  │   ├── model_eval.py            # Evaluation pipeline
  │   ├── model_predict.py         # Prediction pipeline
  │   ├── Download_raw_data.py     # Data download script
  │   └── data_preprocess.py       # Data preprocessing
  ├── config.py                    # Centralized configuration
  ├── utils.py                     # Utility functions
  ├── requirements.txt             # Python dependencies
  ├── Dockerfile                   # Container configuration
  ├── dvc.yaml                     # DVC pipeline definition
  ├── .gitignore                   # Git ignore rules
  └── tests/                       # Test suite
  ```

### ✅ DVC (Data Versioning Control)
- **Pipeline File**: `dvc.yaml` (lines 1-54)
- **Stages Defined**:
  1. **download** — Downloads raw pet images from Kaggle
  2. **prepare** — Preprocesses images to 224×224 RGB format
  3. **train** — Trains SimpleCNN model and logs metrics
  4. **evaluate** — Evaluates model on test set
  5. **predict** — Generates predictions on test data

- **Tracked Outputs**:
  - `data/raw/PetImages` — Raw dataset
  - `data/Preprocessed` — 224×224 preprocessed splits (train/val/test)
  - `models/baseline_cnn.pt` — Trained model weights

- **DVC Integration**:
  - `.dvcignore` — Specifies files to exclude from DVC tracking
  - `dvc.lock` — Records exact versions of all pipeline artifacts
  - DVC cache configured locally (`.dvc/cache`)

---

## 2. Model Building

### ✅ Baseline Model: SimpleCNN
**Location**: `src/model_train.py` (lines 32-60)

**Architecture**:
```
Input Image (3 × 224 × 224)
    ↓
Conv2d(3 → 16) + BatchNorm2d + ReLU + MaxPool2d
    ↓
Conv2d(16 → 32) + BatchNorm2d + ReLU + MaxPool2d
    ↓
Conv2d(32 → 64) + BatchNorm2d + ReLU + MaxPool2d
    ↓
Flatten → Linear(64*28*28 → 128) + ReLU + Dropout(0.3)
    ↓
Linear(128 → 2)  [Binary Classification: Cat/Dog]
    ↓
Output Logits
```

**Key Components**:
- **Layers**: 3 convolutional blocks + 2 fully connected layers
- **Activation**: ReLU
- **Normalization**: Batch Normalization after each convolution
- **Regularization**: Dropout (0.3) in classifier
- **Loss Function**: CrossEntropyLoss (lines 136)
- **Optimizer**: Adam with learning rate 0.001 (line 137)

### ✅ Model Serialization
- **Format**: PyTorch `.pt` (state_dict)
- **Save Location**: `models/baseline_cnn.pt` (line 240)
- **Save Method**: `torch.save(model.state_dict(), target_path)` (utils.py line 41)
- **Load Method**: `torch.load(model_path, map_location=device)` (model_service_app.py line 43)

---

## 3. Experiment Tracking with MLflow

### ✅ MLflow Configuration
**Location**: `config.py` (lines 26-29)
```python
mlflow_tracking_uri: str = "file:./mlruns"
mlflow_experiment_name: str = "cats_vs_dogs_classification"
```

### ✅ Tracked Artifacts & Metrics
**MLflow Run Setup** (lines 143-154):

**Hyperparameters Logged**:
- `model_architecture`: "SimpleCNN"
- `epochs`: 5
- `batch_size`: 64
- `learning_rate`: 0.001
- `image_size`: 224
- `optimizer`: "Adam"
- `device`: "cuda" or "cpu"
- `random_seed`: 42
- `dvc_dataset_hash`: DVC version identifier

**Metrics Logged Per Epoch** (lines 220-225):
- `train_loss`: Training loss
- `train_accuracy`: Training accuracy (%)
- `val_loss`: Validation loss
- `val_accuracy`: Validation accuracy (%)

**Artifacts Logged** (lines 251-253):
1. **Training Curves Plot** (`training_curves.png`):
   - Loss curve (train vs validation)
   - Accuracy curve (train vs validation)
   - Location: `outputs/training_curves.png` (lines 84-109)

2. **Confusion Matrix** (`confusion_matrix.png`):
   - Validation set confusion matrix
   - Classes: Cat, Dog
   - Location: `outputs/confusion_matrix.png` (lines 111-124)

3. **PyTorch Model**:
   - Full model artifact in MLflow format
   - Stored in: `mlruns/<experiment_id>/<run_id>/artifacts/model`
   - Line 253: `mlflow.pytorch.log_model(model, artifact_path="model")`

### ✅ Training Pipeline Execution
**Location**: `src/model_train.py` (lines 126-259)

**Workflow**:
1. Set MLflow tracking URI and experiment name (lines 131-132)
2. Load train/val dataloaders (line 134)
3. Initialize SimpleCNN model and optimizer (lines 135-137)
4. Start MLflow run context (line 143)
5. Log hyperparameters (lines 144-154)
6. For each epoch (lines 159-226):
   - Train step with batch progress logging (lines 164-191)
   - Validation step with confusion matrix collection (lines 193-210)
   - Log metrics per epoch (lines 220-225)
7. Save model state_dict (lines 239-241)
8. Generate and log evaluation plots (lines 243-252)
9. Log PyTorch model artifact (line 253)
10. Complete run with Run ID tracking (line 256)

### ✅ Data Augmentation & Preprocessing
**Location**: `utils.py` (lines 26-37)

**Training Augmentation**:
- RandomHorizontalFlip (p=0.5)
- RandomRotation (±15 degrees)
- ToTensor + Normalization (ImageNet stats)

**Validation/Test (No Augmentation)**:
- Only ToTensor + Normalization

---

## 4. Dataset Configuration

**From config.py** (lines 12-18):
- **Raw Data Path**: `./data/raw/PetImages`
- **Processed Data Path**: `./data/Preprocessed`
- **Image Size**: 224×224 RGB
- **Train/Val/Test Split**: 80/10/10
- **Batch Size**: 64

**Data Loading** (model_train.py lines 62-82):
- Uses `torchvision.datasets.ImageFolder` for hierarchical directory structure
- Splits automatically: `data/Preprocessed/train`, `data/Preprocessed/val`, `data/Preprocessed/test`
- DataLoader with 2 workers for parallel loading

---

## 5. Experiment Tracking Features

| Feature | Implementation | Status |
|---------|-----------------|--------|
| **Run Tracking** | MLflow context manager | ✅ Automatic run creation & logging |
| **Hyperparameter Logging** | `mlflow.log_params()` | ✅ 8 hyperparameters tracked |
| **Metrics Logging** | `mlflow.log_metrics()` per epoch | ✅ 4 metrics × 5 epochs = 20 data points |
| **Artifact Storage** | Local file system (`./mlruns`) | ✅ Plots & model weights saved |
| **Confusion Matrix** | Scikit-learn + Matplotlib | ✅ Generated & logged per run |
| **Loss Curves** | Matplotlib plots | ✅ Generated & logged per run |
| **Model Versioning** | MLflow PyTorch flavor | ✅ Full model logged in MLflow format |
| **Run ID Tracking** | `run.info.run_id` | ✅ Printed & logged per training session |

---

## 6. Reproducibility & Version Control

**Reproducibility Measures**:
- ✅ **Seed Management**: Fixed random seed (42) across NumPy, PyTorch, Python random (utils.py lines 16-21)
- ✅ **Device Tracking**: Device (CPU/GPU) logged as parameter
- ✅ **DVC Hash**: Dataset hash logged to link runs to specific data versions
- ✅ **Configuration Centralization**: All hyperparameters in `config.py`
- ✅ **Git Tracking**: Source code versioned via Git

---

## 7. Workflow Execution Commands

```bash
# 1. Download raw data (DVC stage)
dvc repro download

# 2. Prepare/preprocess data (DVC stage)
dvc repro prepare

# 3. Train model (DVC stage) + MLflow tracking
dvc repro train
# Alternative: python -m src.model_train

# 4. Evaluate model (DVC stage)
dvc repro evaluate

# 5. Generate predictions (DVC stage)
dvc repro predict

# View MLflow UI
mlflow ui --backend-store-uri file:./mlruns
```

---

## 8. Output Artifacts Generated

**Outputs Directory**: `./outputs/`
- `training_curves.png` — Loss & accuracy plots
- `confusion_matrix.png` — Validation confusion matrix
- `metrics.json` — Test set metrics (from evaluate stage)
- `predictions.json` — Test set predictions

**Model Directory**: `./models/`
- `baseline_cnn.pt` — Trained model weights (state_dict)

**MLflow Directory**: `./mlruns/`
- `experiments/` — Experiment metadata
- `<exp_id>/<run_id>/` — Run artifacts, params, metrics

---

## 9. Key Features Summary

✅ **Git-based Code Versioning**
- All source code tracked in GitHub
- `.gitignore` excludes data, models, outputs

✅ **DVC-based Data Versioning**
- Pipeline defined in `dvc.yaml` (5 stages)
- Raw and preprocessed data tracked
- `dvc.lock` ensures reproducibility

✅ **SimpleCNN Baseline Model**
- 3 convolutional blocks + 2 FC layers
- Binary classification (Cat/Dog)
- Serialized as PyTorch `.pt` format

✅ **MLflow Experiment Tracking**
- 8 hyperparameters logged per run
- 4 metrics tracked per epoch
- Training/validation curves saved
- Confusion matrix generated
- Full model artifact logged
- Local file-based tracking (`./mlruns`)

✅ **Reproducibility**
- Fixed random seed
- Configuration centralization
- Data version linking via DVC hash
- Device tracking

---

## 10. Training Statistics Example

**Configuration**:
- Epochs: 5
- Batch Size: 64
- Learning Rate: 0.001
- Image Size: 224×224
- Device: CPU/GPU (auto-detected)

**Expected Outputs**:
- Model weights: `models/baseline_cnn.pt` (~5-10 MB)
- Training curves: `outputs/training_curves.png`
- Confusion matrix: `outputs/confusion_matrix.png`
- MLflow run: Logged with Run ID in `./mlruns/`

---

## Notes

1. **DVC Pipeline**: Fully orchestrated via `dvc.yaml` for reproducible end-to-end training
2. **MLflow UI**: Access via `mlflow ui --backend-store-uri file:./mlruns` on `http://localhost:5000`
3. **Model Portability**: Trained model can be loaded in `model_service_app.py` for inference
4. **Extensibility**: Easy to add more experiments, hyperparameter tuning, or model comparisons via MLflow
