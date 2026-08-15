from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # General
    environment: str = "development"
    debug: bool = True
    random_seed: int = 42
    log_level: str = "INFO"

    # Data & Image Config
    data_path: str = "./data/raw/PetImages"
    processed_data_path: str = "./data/Preprocessed"
    image_size: int = 224
    train_size: float = 0.8
    val_size: float = 0.1
    test_size: float = 0.1

    # Training Hyperparameters
    model_type: str = "SimpleCNN"
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 0.001

    # MLflow Tracking
    #mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment_name: str = "cats_vs_dogs_classification"

    # Paths
    model_output_path: str = "./models"
    model_path: str = "models/baseline_cnn.pt"
    log_dir: str = "./logs"
    output_dir: str = "./outputs"

    # API Server
    host: str = "0.0.0.0"
    port: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra environment variables if present

settings = Settings()