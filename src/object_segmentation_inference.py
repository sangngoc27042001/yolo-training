"""
YOLO Object Segmentation Inference (Bounding Boxes Only)
Performs object detection on images using a TFLite YOLO segmentation model.
Note: This uses a segmentation model but only returns bounding boxes, ignoring masks.
Input: PIL Image
Output: Annotated PIL Image with bounding boxes and labels
"""

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
from pathlib import Path
import os

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_FILE_PATH = "src/yolo26n-seg_saved_model/yolo26n-seg_float16.tflite"  # Path to TFLite segmentation model
OUTPUT_PATH = "yolo26n-seg_output"  # Output directory for inference results


# COCO class names for YOLO
CLASS_NAMES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
    57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
    62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

# Filtered classes to detect
TARGET_CLASSES = [0, 62, 63, 67]  # person, tv, laptop, cell phone

# Colors for different classes (RGB format for PIL)
CLASS_COLORS = {
    0: (255, 0, 0),      # person - Red
    62: (0, 255, 0),     # tv - Green
    63: (0, 0, 255),     # laptop - Blue
    67: (255, 255, 0),   # cell phone - Yellow
}

CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for detections
IOU_THRESHOLD = 0.45  # IoU threshold for Non-Maximum Suppression
TOP_K = 30  # Keep only top 30 detections


def preprocess_image(image: Image.Image, target_size: tuple = (640, 640)) -> tuple:
    """
    Preprocess PIL image for YOLO model input.

    Args:
        image: PIL Image object
        target_size: Target size for model input (width, height)

    Returns:
        tuple: (preprocessed_array, scale, pad_x, pad_y)
    """
    # Get original dimensions
    orig_width, orig_height = image.size

    # Calculate scaling to maintain aspect ratio
    scale = min(target_size[0] / orig_width, target_size[1] / orig_height)
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)

    # Resize image
    resized_image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)

    # Create padded image (centered)
    padded_image = Image.new('RGB', target_size, (114, 114, 114))
    pad_x = (target_size[0] - new_width) // 2
    pad_y = (target_size[1] - new_height) // 2
    padded_image.paste(resized_image, (pad_x, pad_y))

    # Convert to numpy array and normalize to [0, 1]
    image_array = np.array(padded_image).astype(np.float32) / 255.0

    # Add batch dimension: (1, H, W, 3)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array, scale, pad_x, pad_y


def calculate_iou(box1: tuple, box2: tuple) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: Bounding box (x1, y1, x2, y2)
        box2: Bounding box (x1, y1, x2, y2)
    
    Returns:
        float: IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area
    
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area


def apply_nms(detections: list, iou_threshold: float = 0.45) -> list:
    """
    Apply Non-Maximum Suppression to remove overlapping detections.
    
    Args:
        detections: List of detections with 'bbox', 'confidence', 'class_id'
        iou_threshold: IoU threshold for suppression
    
    Returns:
        list: Filtered detections after NMS
    """
    if len(detections) == 0:
        return []
    
    # Sort by confidence (highest first)
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    # Group detections by class
    class_detections = {}
    for det in detections:
        class_id = det['class_id']
        if class_id not in class_detections:
            class_detections[class_id] = []
        class_detections[class_id].append(det)
    
    # Apply NMS per class
    final_detections = []
    for class_id, class_dets in class_detections.items():
        keep = []
        while len(class_dets) > 0:
            # Take the detection with highest confidence
            current = class_dets.pop(0)
            keep.append(current)
            
            # Remove detections with high IoU overlap
            class_dets = [
                det for det in class_dets
                if calculate_iou(current['bbox'], det['bbox']) < iou_threshold
            ]
        
        final_detections.extend(keep)
    
    # Sort final detections by confidence
    final_detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    return final_detections


def postprocess_output(output: np.ndarray, orig_size: tuple, input_size: tuple,
                       conf_threshold: float = 0.3, target_classes: list = None,
                       top_k: int = 30, iou_threshold: float = 0.45) -> list:
    """
    Postprocess YOLO segmentation model output to extract object detections (bounding boxes only).
    
    Note: This function ignores segmentation masks and only extracts bounding boxes.

    Args:
        output: Model output array with shape (1, N, 6+mask_coeffs) or similar
                Format per detection: [x1_norm, y1_norm, x2_norm, y2_norm, conf, class_id, ...]
                All coordinates are normalized to [0, 1] relative to input size
        orig_size: Original image size (width, height)
        input_size: Model input size (width, height)
        conf_threshold: Confidence threshold for detections
        target_classes: List of class IDs to keep (None = keep all)
        top_k: Keep only top K detections by confidence
        iou_threshold: IoU threshold for Non-Maximum Suppression

    Returns:
        list: List of detections, each containing bbox, class, and confidence
    """
    detections = []
    orig_width, orig_height = orig_size
    input_width, input_height = input_size

    # Calculate the scale factor and padding used during preprocessing
    scale = min(input_width / orig_width, input_height / orig_height)
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)
    pad_x = (input_width - new_width) // 2
    pad_y = (input_height - new_height) // 2

    # Process each detection
    for detection in output[0]:
        # Extract bbox, confidence, and class
        # Format: [x1_norm, y1_norm, x2_norm, y2_norm, confidence, class_id, ...mask_coefficients]
        # We only care about the first 6 values (bounding box info)
        x1_norm, y1_norm, x2_norm, y2_norm = detection[0:4]
        confidence = detection[4]
        class_id = int(detection[5])

        # Filter by confidence
        if confidence < conf_threshold:
            continue

        # Filter by target classes
        if target_classes is not None and class_id not in target_classes:
            continue

        # Convert normalized coordinates to pixel coordinates in input space
        x1_input = x1_norm * input_width
        y1_input = y1_norm * input_height
        x2_input = x2_norm * input_width
        y2_input = y2_norm * input_height

        # Remove padding
        x1_scaled = x1_input - pad_x
        y1_scaled = y1_input - pad_y
        x2_scaled = x2_input - pad_x
        y2_scaled = y2_input - pad_y

        # Scale back to original image size
        x1 = x1_scaled / scale
        y1 = y1_scaled / scale
        x2 = x2_scaled / scale
        y2 = y2_scaled / scale

        # Convert to integers
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        # Clip to image boundaries
        x1 = max(0, min(x1, orig_width))
        y1 = max(0, min(y1, orig_height))
        x2 = max(0, min(x2, orig_width))
        y2 = max(0, min(y2, orig_height))

        detections.append({
            'bbox': (x1, y1, x2, y2),
            'confidence': float(confidence),
            'class_id': class_id,
            'class_name': CLASS_NAMES.get(class_id, f'class_{class_id}')
        })

    # Apply Non-Maximum Suppression
    detections = apply_nms(detections, iou_threshold)
    
    # Keep top K detections after NMS
    detections = detections[:top_k]

    return detections


def annotate_image(image: Image.Image, detections: list) -> Image.Image:
    """
    Draw detection annotations on the image.

    Args:
        image: PIL Image to annotate
        detections: List of detections from postprocess_output

    Returns:
        PIL Image with annotations
    """
    # Create a copy to avoid modifying original
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)

    # Try to use a better font, fall back to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    for detection in detections:
        bbox = detection['bbox']
        class_id = detection['class_id']
        class_name = detection['class_name']
        confidence = detection['confidence']

        # Get color for this class
        color = CLASS_COLORS.get(class_id, (128, 128, 128))

        # Draw bounding box
        draw.rectangle(bbox, outline=color, width=3)

        # Prepare label text
        label = f"{class_name}: {confidence:.2f}"

        # Get text size for background
        bbox_text = draw.textbbox((bbox[0], bbox[1] - 20), label, font=font)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]

        # Draw background rectangle for text
        draw.rectangle(
            [(bbox[0], bbox[1] - text_height - 4), (bbox[0] + text_width + 4, bbox[1])],
            fill=color
        )

        # Draw label text
        draw.text((bbox[0] + 2, bbox[1] - text_height - 2), label, fill=(255, 255, 255), font=font)

    # Add summary text
    summary = f"Detected: {len(detections)} objects (Top {min(len(detections), TOP_K)})"
    draw.text((10, 10), summary, fill=(255, 255, 255), font=font)
    draw.text((10, 9), summary, fill=(0, 0, 0), font=font)  # Shadow for readability

    return annotated_image


def inference(image: Image.Image, model_path: str = None,
              target_classes: list = TARGET_CLASSES, top_k: int = TOP_K) -> tuple:
    """
    Perform object detection inference on a PIL image using a segmentation model.
    Note: Only bounding boxes are returned; segmentation masks are ignored.

    Args:
        image: Input PIL Image
        model_path: Path to the TFLite model (defaults to MODEL_FILE_PATH from config)
        target_classes: List of class IDs to detect (None = detect all)
        top_k: Keep only top K detections by confidence

    Returns:
        tuple: (annotated_image, detections, is_second_laptop)
    """
    # Use configured model path if not provided
    if model_path is None:
        model_path = MODEL_FILE_PATH
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get input shape
    input_shape = input_details[0]['shape']
    target_size = (input_shape[2], input_shape[1])  # (width, height)

    print(f"Model input shape: {input_shape}")
    print(f"Model input size: {target_size}")

    # Preprocess image
    preprocessed, scale, pad_x, pad_y = preprocess_image(image, target_size)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], preprocessed)
    interpreter.invoke()

    # Get output (first output contains bounding boxes)
    output = interpreter.get_tensor(output_details[0]['index'])
    print(f"Model output shape: {output.shape}")

    # Postprocess output (extract bounding boxes only, ignore masks)
    detections = postprocess_output(
        output, image.size, target_size,
        conf_threshold=CONFIDENCE_THRESHOLD,
        target_classes=target_classes,
        top_k=top_k,
        iou_threshold=IOU_THRESHOLD
    )

    # Annotate image
    annotated_image = annotate_image(image, detections)

    # Detect second laptop
    laptop_detections = [det for det in detections if det['class_id'] == 63]
    laptop_detections_count = len(laptop_detections)
    print(f"Number of laptops detected: {laptop_detections_count}")
    is_second_laptop = laptop_detections_count >= 2

    return annotated_image, detections, is_second_laptop


def process_input(input_path: str, model_path: str = None):
    """
    Process input which can be either a URL to an image or a path to a folder with images.

    Args:
        input_path: Either a URL (starting with http:// or https://) or a folder path
        model_path: Path to the TFLite model (defaults to MODEL_FILE_PATH from config)
    """
    # Use configured model path if not provided
    if model_path is None:
        model_path = MODEL_FILE_PATH
    
    # Create output directory if it doesn't exist
    output_dir = Path(OUTPUT_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Model path: {model_path}")
    print(f"Output directory: {OUTPUT_PATH}")
    print(f"Target classes: {[CLASS_NAMES[c] for c in TARGET_CLASSES]}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"IoU threshold (NMS): {IOU_THRESHOLD}")
    print(f"Top K detections: {TOP_K}\n")

    # Check if input is a URL
    if input_path.startswith('http://') or input_path.startswith('https://'):
        print("=" * 60)
        print("Processing image from URL")
        print("=" * 60)
        print(f"Downloading image from {input_path}...")

        try:
            response = requests.get(input_path)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            print(f"Image size: {image.size}")

            # Run inference
            print("\nRunning object detection inference (segmentation model, bounding boxes only)...")
            annotated_image, detections, is_second_laptop = inference(image, model_path)

            # Print detections
            print(f"\nFound {len(detections)} detections:")
            for i, det in enumerate(detections, 1):
                print(f"  {i}. {det['class_name']}: {det['confidence']:.3f} at {det['bbox']}")

            # Save result to output directory
            output_path = output_dir / "object_segmentation_result.jpg"
            if annotated_image.mode == 'RGBA':
                annotated_image = annotated_image.convert('RGB')
            annotated_image.save(output_path)
            print(f"\nSaved annotated image to {output_path}")

        except Exception as e:
            print(f"Error processing URL: {e}")

    else:
        # Input is a folder path
        folder_path = Path(input_path)

        if not folder_path.exists():
            print(f"Error: Path '{input_path}' does not exist!")
            return

        if not folder_path.is_dir():
            print(f"Error: Path '{input_path}' is not a directory!")
            return

        print("=" * 60)
        print(f"Processing images from folder: {input_path}")
        print("=" * 60)

        # Get all image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp']
        image_files = [f for f in folder_path.iterdir()
                      if f.is_file() and f.suffix.lower() in image_extensions]

        if not image_files:
            print(f"No image files found in {folder_path}")
            return

        print(f"Found {len(image_files)} images\n")

        for idx, image_path in enumerate(sorted(image_files), 1):
            print(f"\n[{idx}/{len(image_files)}] Processing: {image_path.name}")
            print("-" * 60)

            try:
                # Load image
                image = Image.open(image_path)
                print(f"Image size: {image.size}")

                # Run inference
                annotated_image, detections, is_second_laptop = inference(image, model_path)

                # Print detections
                print(f"\nFound {len(detections)} detections:")
                for i, det in enumerate(detections, 1):
                    print(f"  {i}. {det['class_name']}: {det['confidence']:.3f} at {det['bbox']}")

                # Save result to output directory
                output_filename = output_dir / f"detection_{image_path.stem}.jpg"
                if annotated_image.mode == 'RGBA':
                    annotated_image = annotated_image.convert('RGB')
                annotated_image.save(output_filename)
                print(f"Saved to {output_filename}")

            except Exception as e:
                print(f"Error processing {image_path.name}: {e}")

        print("\n" + "=" * 60)
        print("Processing complete!")
        print("=" * 60)


if __name__ == "__main__":
    import sys

    # Default input (URL or folder path)
    default_input = "src/yolo_test_folder"

    # Check if input path provided via command line
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        # Optional: model path as second argument
        model_path = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        # Use default URL, can be changed to "yolo_test_folder" for folder processing
        input_path = default_input
        model_path = None
        print(f"No input provided, using default: {input_path}")
        print(f"Using model: {MODEL_FILE_PATH}")
        print(f"Output directory: {OUTPUT_PATH}")
        print("Usage: python object_segmentation_inference.py <url_or_folder_path> [model_path]\n")

    # Process the input
    process_input(input_path, model_path)
