"""
YOLOv11 Training Script for Phone Detection
"""
import sys
from pathlib import Path

from ultralytics import YOLO

from src.config import settings


def main():
    """Main training function for YOLOv11 model."""

    print("=" * 60)
    print("YOLOv11 Training Configuration")
    print("=" * 60)
    print(f"Dataset: {settings.DATASET_NAME}")
    print(f"Data YAML: {settings.DATA_YAML}")
    print(f"Model: {settings.MODEL_NAME}")
    print(f"Epochs: {settings.EPOCHS}")
    print(f"Batch Size: {settings.BATCH_SIZE}")
    print(f"Image Size: {settings.IMAGE_SIZE}")
    print(f"Device: {settings.DEVICE}")
    print("=" * 60)

    # Verify data.yaml exists
    if not settings.DATA_YAML.exists():
        print(f"Error: Data YAML file not found at {settings.DATA_YAML}")
        sys.exit(1)

    # Load the model
    print(f"\nLoading model: {settings.MODEL_NAME}")
    model = YOLO(settings.MODEL_NAME)

    # Train the model
    print("\nStarting training...")
    results = model.train(
        data=str(settings.DATA_YAML),
        epochs=settings.EPOCHS,
        batch=settings.BATCH_SIZE,
        imgsz=settings.IMAGE_SIZE,
        device=settings.DEVICE,
        patience=settings.PATIENCE,
        save_period=settings.SAVE_PERIOD,
        workers=settings.WORKERS,
        project=settings.PROJECT_NAME,
        name=settings.EXPERIMENT_NAME,
        verbose=settings.VERBOSE,
        save=settings.SAVE,
        plots=settings.PLOTS,
    )

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    print(f"Results saved to: {Path(settings.PROJECT_NAME) / settings.EXPERIMENT_NAME}")

    # Validate the model
    print("\nRunning validation...")
    metrics = model.val()

    print("\nValidation Metrics:")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    return results


if __name__ == "__main__":
    main()
