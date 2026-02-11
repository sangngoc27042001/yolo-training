.PHONY: train export-weights pod-inference camera-guide

train:
	uv run python -m src.train

export-weights:
	uv run python -m src.export

pod-inference:
	uv run python -m src.pose_object_detection_running.run_inference

camera-guide:
	uv run python -m src.camera_position_guide_yolo
