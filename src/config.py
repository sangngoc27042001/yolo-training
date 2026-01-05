from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Training configuration settings using Pydantic."""

    # Dataset configuration
    DATASET_NAME: str = "phone-1"
    DATASET_PATH: Path = Path("datasets/phone-1")
    DATA_YAML: Path = Path("datasets/phone-1/data.yaml")

    # Model configuration
    MODEL_NAME: str = "yolo11n.pt"  # Options: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x

    # Training hyperparameters
    EPOCHS: int = 1
    BATCH_SIZE: int = 16
    IMAGE_SIZE: int = 640
    DEVICE: str = "cpu"  # Options: cpu, 0, 1, etc. for GPU

    # Training parameters
    PATIENCE: int = 50  # Early stopping patience
    SAVE_PERIOD: int = 10  # Save checkpoint every N epochs
    WORKERS: int = 8  # Number of worker threads

    # Output configuration
    PROJECT_NAME: str = "runs/detect"
    EXPERIMENT_NAME: str = "phone_detection"

    # Performance settings
    VERBOSE: bool = True
    SAVE: bool = True
    PLOTS: bool = True

    class Config:
        env_prefix = "YOLO_"
        case_sensitive = False


# Global settings instance
settings = Settings()
