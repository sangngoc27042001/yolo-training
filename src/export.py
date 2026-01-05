"""
YOLO Model Export Script
Exports trained YOLO model to ONNX and TensorFlow Lite formats with various configurations
"""

from ultralytics import YOLO
import os
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
    Export YOLO model to multiple formats.

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
    weights_path = Path(settings.PROJECT_NAME) / experiment_name_full / "weights" / "best.pt"
    export_dir = Path(settings.PROJECT_NAME) / experiment_name_full / "exported_weights"

    # Validate weights file exists
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found at: {weights_path}")

    # Create export directory if it doesn't exist
    os.makedirs(export_dir, exist_ok=True)

    # Load the trained model
    print(f"Loading model from {weights_path}...")
    model = YOLO(str(weights_path))

    # Export to ONNX
    print("\n" + "="*60)
    print("Exporting to ONNX format...")
    print("="*60)
    onnx_path = model.export(
        format='onnx',
        nms=True,       # Include Non-Maximum Suppression to filter detections
        simplify=True,  # Simplify the model graph
    )
    # Move ONNX file to export directory
    if onnx_path:
        onnx_file = Path(onnx_path)
        dest_onnx = export_dir / onnx_file.name
        onnx_file.rename(dest_onnx)
        print(f"✓ ONNX model saved to: {dest_onnx}")

    # TensorFlow Lite export configurations
    tflite_configs = [
        {'imgsz': 640, 'half': True, 'name': 'tflite_640_fp16'},
        {'imgsz': 320, 'half': True, 'name': 'tflite_320_fp16'},
        {'imgsz': 160, 'half': True, 'name': 'tflite_160_fp16'},
        {'imgsz': 640, 'half': False, 'name': 'tflite_640_fp32'},
        {'imgsz': 320, 'half': False, 'name': 'tflite_320_fp32'},
        {'imgsz': 160, 'half': False, 'name': 'tflite_160_fp32'},
    ]

    # Export each TensorFlow Lite configuration
    for i, config in enumerate(tflite_configs, 1):
        print("\n" + "="*60)
        print(f"Exporting TFLite model {i}/{len(tflite_configs)}: {config['name']}")
        print(f"Image size: {config['imgsz']}, Precision: {'FP16' if config['half'] else 'FP32'}")
        print("="*60)

        # Reload model for each export to avoid issues
        model = YOLO(str(weights_path))

        tflite_path = model.export(
            format='tflite',
            imgsz=config['imgsz'],
            half=config['half'],
            nms=True,       # Include Non-Maximum Suppression to filter detections
        )

        # Move TFLite file to export directory with custom name
        if tflite_path:
            tflite_file = Path(tflite_path)
            dest_tflite = export_dir / f"{config['name']}.tflite"
            tflite_file.rename(dest_tflite)
            print(f"✓ TFLite model saved to: {dest_tflite}")

    print("\n" + "="*60)
    print("Export Complete!")
    print("="*60)
    print(f"\nAll exported models are saved in: {export_dir}")
    print("\nExported files:")
    print("  - 1 ONNX model (with NMS and simplified graph)")
    print("  - 6 TensorFlow Lite models:")
    print("    • 640px (FP16 & FP32)")
    print("    • 320px (FP16 & FP32)")
    print("    • 160px (FP16 & FP32)")
    print("\nAll models include NMS (Non-Maximum Suppression)")


if __name__ == "__main__":
    # Automatically finds and exports the latest run
    # Or you can specify: export_model(run_number=2)
    export_model()
