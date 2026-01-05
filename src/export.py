"""
YOLO Model Export Script
Exports trained YOLO model to TensorFlow Lite formats with various image sizes
"""

from ultralytics import YOLO
import os
import shutil
from pathlib import Path
from src.config import settings
import re


def find_latest_run() -> int:
    """
    Find the latest run number by scanning the project directory.

    Returns:
        The highest run number found, or 1 if none exist
    """
    project_path = Path(settings.PROJECT_NAME)
    if not project_path.exists():
        raise FileNotFoundError(f"Project directory not found: {project_path}")

    # Pattern to match experiment names like "phone_detection", "phone_detection2", etc.
    pattern = re.compile(rf"^{re.escape(settings.EXPERIMENT_NAME)}(\d*)$")

    run_numbers = []
    for item in project_path.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                # Empty string means run 1 (no number suffix)
                run_num = int(match.group(1)) if match.group(1) else 1
                run_numbers.append(run_num)

    if not run_numbers:
        raise FileNotFoundError(f"No runs found matching pattern: {settings.EXPERIMENT_NAME}")

    return max(run_numbers)


def export_model(run_number: int = None):
    """
    Export YOLO model to TensorFlow Lite format with different image sizes.
    Creates best_saved_model folders for each size and copies them to exported_models directory.

    Args:
        run_number: The run number to export. If None, finds the latest run automatically.
    """
    # Find latest run if not specified
    if run_number is None:
        run_number = find_latest_run()
        print(f"Auto-detected latest run: {run_number}")

    # Construct paths from config
    # Handle run_number=1 as no suffix, run_number>1 as suffix
    experiment_name_full = settings.EXPERIMENT_NAME if run_number == 1 else f"{settings.EXPERIMENT_NAME}{run_number}"
    weights_dir = Path(settings.PROJECT_NAME) / experiment_name_full / "weights"
    weights_path = weights_dir / "best.pt"
    export_dir = Path(settings.PROJECT_NAME) / experiment_name_full / "exported_models"

    # Validate weights file exists
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found at: {weights_path}")

    # Create export directory if it doesn't exist
    os.makedirs(export_dir, exist_ok=True)

    # Image sizes to export
    image_sizes = [640, 320, 160]

    print("="*60)
    print("YOLO Model Export - Multiple Image Sizes")
    print("="*60)
    print(f"Model: {weights_path}")
    print(f"Export directory: {export_dir}")
    print(f"Image sizes: {image_sizes}")
    print("="*60)

    # Export each image size
    for i, imgsz in enumerate(image_sizes, 1):
        print(f"\n{'='*60}")
        print(f"Export {i}/{len(image_sizes)}: Image size {imgsz}x{imgsz}")
        print("="*60)

        # Clean up previous best_saved_model if it exists
        saved_model_path = weights_dir / "best_saved_model"
        if saved_model_path.exists():
            print(f"Cleaning up previous export folder: {saved_model_path}")
            shutil.rmtree(saved_model_path)

        # Load and export model
        print(f"Loading model from {weights_path}...")
        model = YOLO(str(weights_path))

        print(f"Exporting TFLite model with imgsz={imgsz}...")
        model.export(
            format="tflite",
            imgsz=imgsz,
            dynamic=False,
            simplify=True,
            batch=1,
            nms=True,
        )

        # Copy the best_saved_model folder to export directory
        if saved_model_path.exists():
            dest_folder = export_dir / f"best_saved_model_{imgsz}"
            if dest_folder.exists():
                print(f"Removing existing folder: {dest_folder}")
                shutil.rmtree(dest_folder)

            print(f"Copying {saved_model_path} -> {dest_folder}")
            shutil.copytree(saved_model_path, dest_folder)
            print(f"✓ Exported model saved to: {dest_folder}")

            # List contents of the exported folder
            print(f"\nContents of {dest_folder.name}:")
            for item in dest_folder.rglob("*"):
                if item.is_file():
                    size_mb = item.stat().st_size / (1024 * 1024)
                    rel_path = item.relative_to(dest_folder)
                    print(f"  - {rel_path} ({size_mb:.2f} MB)")
        else:
            print(f"⚠ Warning: best_saved_model folder not found after export")

    # Clean up the last best_saved_model in weights directory
    if saved_model_path.exists():
        print(f"\nCleaning up temporary folder: {saved_model_path}")
        shutil.rmtree(saved_model_path)

    print("\n" + "="*60)
    print("Export Complete!")
    print("="*60)
    print(f"\nAll exported models are saved in: {export_dir}")
    print(f"\nExported folders:")
    for imgsz in image_sizes:
        folder = export_dir / f"best_saved_model_{imgsz}"
        if folder.exists():
            print(f"  ✓ best_saved_model_{imgsz}/ (contains FP16 and FP32 variants)")
        else:
            print(f"  ✗ best_saved_model_{imgsz}/ (not found)")
    print("\nEach folder contains TFLite models with NMS included")


if __name__ == "__main__":
    # Automatically finds and exports the latest run
    # Or you can specify: export_model(run_number=2)
    export_model()
