"""
YOLO Pose Estimation Inference
Performs pose estimation on images using a TFLite YOLO model.
Input: PIL Image
Output: Annotated PIL Image with pose keypoints
"""

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO


# COCO Keypoint configuration (17 keypoints)
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Skeleton connections for drawing limbs
SKELETON = [
    (0, 1), (0, 2),  # nose to eyes
    (1, 3), (2, 4),  # eyes to ears
    (0, 5), (0, 6),  # nose to shoulders
    (5, 6),  # shoulders
    (5, 7), (7, 9),  # left arm
    (6, 8), (8, 10),  # right arm
    (5, 11), (6, 12),  # shoulders to hips
    (11, 12),  # hips
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]

# Colors for visualization
KEYPOINT_COLOR = (255, 0, 0)  # Red for keypoints
SKELETON_COLOR = (0, 255, 0)  # Green for skeleton
BBOX_COLOR = (0, 0, 255)  # Blue for bounding box
CONFIDENCE_THRESHOLD = 0.78  # Minimum confidence for keypoints


def preprocess_image(image: Image.Image, target_size: tuple = (320, 320)) -> tuple:
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

    # Add batch dimension: (1, 320, 320, 3)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array, scale, pad_x, pad_y


def postprocess_output(output: np.ndarray, orig_size: tuple, input_size: tuple,
                       conf_threshold: float = 0.3) -> list:
    """
    Postprocess YOLO model output to extract pose information.

    Args:
        output: Model output array with shape (1, 300, 57)
                Format per detection: [x_norm, y_norm, w_norm, h_norm, conf, class_conf, 17*3 keypoints]
                All coordinates are normalized to [0, 1] relative to input size (320x320)
                Keypoint format: [x_norm, y_norm, confidence] for each of 17 keypoints
        orig_size: Original image size (width, height)
        input_size: Model input size (width, height), typically (320, 320)
        conf_threshold: Confidence threshold for detections

    Returns:
        list: List of detections, each containing bbox and keypoints
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

    # Output shape: (1, 300, 57)
    # 57 = 4 (bbox) + 1 (obj_conf) + 1 (class_conf) + 51 (17 keypoints * 3)
    for detection in output[0]:
        # Extract bbox and confidence
        # Bbox format: [x1_norm, y1_norm, x2_norm, y2_norm] (corner coordinates, normalized)
        x1_norm, y1_norm, x2_norm, y2_norm = detection[0:4]
        obj_conf = detection[4]

        # For single-class models, just use obj_conf
        # (class_conf at index 5 is always 0)
        confidence = obj_conf

        if confidence < conf_threshold:
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

        # Extract keypoints (17 keypoints * 3 values)
        keypoints = []
        for i in range(17):
            kp_idx = 6 + i * 3
            kp_x_norm = detection[kp_idx]
            kp_y_norm = detection[kp_idx + 1]
            kp_conf = detection[kp_idx + 2]

            # Convert normalized keypoint coordinates to pixel coordinates
            kp_x_input = kp_x_norm * input_width
            kp_y_input = kp_y_norm * input_height

            # Remove padding
            kp_x_scaled = kp_x_input - pad_x
            kp_y_scaled = kp_y_input - pad_y

            # Scale back to original image size
            kp_x = kp_x_scaled / scale
            kp_y = kp_y_scaled / scale

            keypoints.append({
                'name': KEYPOINT_NAMES[i],
                'x': float(kp_x),
                'y': float(kp_y),
                'confidence': float(kp_conf)
            })

        detections.append({
            'bbox': (x1, y1, x2, y2),
            'confidence': float(confidence),
            'keypoints': keypoints
        })

    return detections


def annotate_image(image: Image.Image, detections: list,
                   draw_bbox: bool = True, draw_keypoints: bool = True,
                   draw_skeleton: bool = True) -> Image.Image:
    """
    Draw pose annotations on the image.

    Args:
        image: PIL Image to annotate
        detections: List of detections from postprocess_output
        draw_bbox: Whether to draw bounding boxes
        draw_keypoints: Whether to draw keypoints
        draw_skeleton: Whether to draw skeleton connections

    Returns:
        PIL Image with annotations
    """
    # Create a copy to avoid modifying original
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)

    # Try to use a better font, fall back to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        font = ImageFont.load_default()

    for detection in detections:
        bbox = detection['bbox']
        keypoints = detection['keypoints']
        confidence = detection['confidence']

        # Draw bounding box
        if draw_bbox:
            draw.rectangle(bbox, outline=BBOX_COLOR, width=2)
            # Draw confidence
            conf_text = f"Person: {confidence:.2f}"
            draw.text((bbox[0], bbox[1] - 20), conf_text, fill=BBOX_COLOR, font=font)

        # Draw skeleton connections
        if draw_skeleton:
            for start_idx, end_idx in SKELETON:
                start_kp = keypoints[start_idx]
                end_kp = keypoints[end_idx]

                # Only draw if both keypoints are confident
                if (start_kp['confidence'] > CONFIDENCE_THRESHOLD and
                    end_kp['confidence'] > CONFIDENCE_THRESHOLD):
                    start_pos = (int(start_kp['x']), int(start_kp['y']))
                    end_pos = (int(end_kp['x']), int(end_kp['y']))
                    draw.line([start_pos, end_pos], fill=SKELETON_COLOR, width=2)

        # Draw keypoints
        if draw_keypoints:
            for kp in keypoints:
                if kp['confidence'] > CONFIDENCE_THRESHOLD:
                    x, y = int(kp['x']), int(kp['y'])
                    radius = 4
                    draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)],
                               fill=KEYPOINT_COLOR, outline=(255, 255, 255), width=1)

    return annotated_image


def run_inference(image: Image.Image, model_path: str = "saved_models/yolo11n-pose_saved_model/yolo11n-pose_float16.tflite") -> list:
    """
    Perform pose estimation inference on a PIL image and return detections only.

    Args:
        image: Input PIL Image
        model_path: Path to the TFLite model

    Returns:
        list: List of detections with keypoints
    """
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get input shape
    input_shape = input_details[0]['shape']
    target_size = (input_shape[2], input_shape[1])  # (width, height)

    # Preprocess image
    preprocessed, scale, pad_x, pad_y = preprocess_image(image, target_size)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], preprocessed)
    interpreter.invoke()

    # Get output
    output = interpreter.get_tensor(output_details[0]['index'])

    # Postprocess output
    detections = postprocess_output(output, image.size, target_size)

    return detections


def get_detections(image: Image.Image, model_path: str = "yolo11n-pose_saved_model/yolo11n-pose_float16.tflite") -> list:
    """
    Alias for run_inference to get pose detections.
    """
    return run_inference(image, model_path)


def annotate_image_with_detections(image: Image.Image, detections: list,
                                   draw_bbox: bool = True, draw_keypoints: bool = True,
                                   draw_skeleton: bool = True) -> Image.Image:
    """
    Annotate image with pose detections.
    """
    return annotate_image(image, detections, draw_bbox, draw_keypoints, draw_skeleton)


def inference(image: Image.Image, model_path: str = "yolo11n-pose_saved_model/yolo11n-pose_float16.tflite") -> tuple:
    """
    Perform pose estimation inference on a PIL image.

    Args:
        image: Input PIL Image
        model_path: Path to the TFLite model

    Returns:
        tuple: (annotated_image, detections)
    """
    detections = run_inference(image, model_path)
    annotated_image = annotate_image(image, detections)

    return annotated_image, detections


if __name__ == "__main__":
    # Download image from URL
    # image_url = "https://i.ibb.co/JWZj7kLx/image-10.png"
    image_url = "https://i.ibb.co/d4L8tnv8/image-13.png"
    print(f"Downloading image from {image_url}...")
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    print(f"Image size: {image.size}")

    # Run inference
    print("Running pose estimation inference...")
    annotated_image = inference(image)

    # Save result
    output_path = "pose_estimation_result.jpg"
    # Convert RGBA to RGB if necessary (JPEG doesn't support transparency)
    if annotated_image.mode == 'RGBA':
        rgb_image = Image.new('RGB', annotated_image.size, (255, 255, 255))
        rgb_image.paste(annotated_image, mask=annotated_image.split()[3])
        rgb_image.save(output_path)
    else:
        annotated_image.save(output_path)
    print(f"Saved annotated image to {output_path}")
