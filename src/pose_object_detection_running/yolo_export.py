from ultralytics import YOLO


def export_pose_model():
    """Export YOLO pose estimation model to TFLite format."""
    print("Exporting pose model...")
    model = YOLO("yolo11n-pose.pt")

    model.export(
        format="tflite",
        int8=True,
        nms=True,
        simplify=True,
        # imgsz=320,
    )
    print("Pose model exported successfully!")


def export_object_detection_model():
    """Export YOLO object detection model to TFLite format."""
    print("Exporting object detection model...")
    model = YOLO("yolo11n.pt")

    model.export(
        format="tflite",
        int8=True,
        nms=True,
        simplify=True,
        # imgsz=320,
    )
    print("Object detection model exported successfully!")


def main():
    export_pose_model()
    export_object_detection_model()


if __name__ == "__main__":
    main()
