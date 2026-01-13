"""
Combined Pose and Object Detection Inference
Processes images with both pose estimation and object detection models.
Outputs annotated images and JSON files with detection results.
"""

import json
from pathlib import Path
from PIL import Image

from src.pose_object_detection_running import object_detection_inference
from src.pose_object_detection_running import pose_inference


def combine_annotations(image: Image.Image,
                       object_detections: list,
                       pose_detections: list) -> Image.Image:
    """
    Combine both object detection and pose estimation annotations on a single image.

    Args:
        image: Original PIL Image
        object_detections: List of object detections
        pose_detections: List of pose detections

    Returns:
        PIL Image with combined annotations
    """
    # Start with the original image
    combined_image = image.copy()

    # First annotate with object detections
    combined_image = object_detection_inference.annotate_image(combined_image, object_detections)

    # Then annotate with pose detections (on top of object annotations)
    combined_image = pose_inference.annotate_image(combined_image, pose_detections)

    return combined_image


def process_image(image_path: Path,
                 output_dir: Path,
                 object_model_path: str = "src/pose_object_detection_running/saved_models/yolo11n_saved_model/yolo11n_float16.tflite",
                 pose_model_path: str = "src/pose_object_detection_running/saved_models/yolo11n-pose_saved_model/yolo11n-pose_float16.tflite") -> dict:
    """
    Process a single image with both pose and object detection.

    Args:
        image_path: Path to input image
        output_dir: Directory to save outputs
        object_model_path: Path to object detection model
        pose_model_path: Path to pose estimation model

    Returns:
        dict: Combined detection results
    """
    print(f"\nProcessing: {image_path.name}")
    print("-" * 60)

    # Load image
    image = Image.open(image_path)
    print(f"Image size: {image.size}")

    # Run object detection
    print("Running object detection...")
    object_detections = object_detection_inference.run_inference(
        image,
        model_path=object_model_path
    )
    print(f"Found {len(object_detections)} object(s)")

    # Run pose estimation
    print("Running pose estimation...")
    pose_detections = pose_inference.run_inference(
        image,
        model_path=pose_model_path
    )
    print(f"Found {len(pose_detections)} person(s) with pose")

    # Combine annotations on image
    print("Creating combined annotation...")
    annotated_image = combine_annotations(image, object_detections, pose_detections)

    # Prepare output filenames
    stem = image_path.stem
    output_image_path = output_dir / f"{stem}_annotated.jpg"
    output_json_path = output_dir / f"{stem}_detections.json"

    # Save annotated image
    if annotated_image.mode == 'RGBA':
        annotated_image = annotated_image.convert('RGB')
    annotated_image.save(output_image_path, quality=95)
    print(f"Saved annotated image: {output_image_path}")

    # Prepare combined results
    results = {
        "image_name": image_path.name,
        "image_size": {
            "width": image.size[0],
            "height": image.size[1]
        },
        "object_detections": object_detections,
        "pose_detections": pose_detections,
        "summary": {
            "num_objects": len(object_detections),
            "num_persons": len(pose_detections)
        }
    }

    # Save JSON results
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved detections JSON: {output_json_path}")

    return results


def main():
    """
    Main function to process all images in the test folder.
    """
    # Define paths
    input_dir = Path("src/pose_object_detection_running/yolo_test_folder")
    output_dir = Path("src/pose_object_detection_running/output")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp']
    image_files = [f for f in input_dir.iterdir()
                   if f.is_file() and f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    print("=" * 60)
    print(f"Combined Pose and Object Detection Inference")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(image_files)} image(s) to process")
    print("=" * 60)

    # Process each image
    all_results = []
    for idx, image_path in enumerate(sorted(image_files), 1):
        print(f"\n[{idx}/{len(image_files)}]")
        try:
            results = process_image(image_path, output_dir)
            all_results.append(results)
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
            import traceback
            traceback.print_exc()

    # Save summary JSON
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "total_images": len(image_files),
            "processed_images": len(all_results),
            "results": all_results
        }, f, indent=2)

    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"Processed {len(all_results)}/{len(image_files)} images")
    print(f"Summary saved to: {summary_path}")
    print(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
