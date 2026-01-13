.PHONY: train export-weights pod-inference

train:
	uv run python -m src.train

export-weights:
	uv run python -m src.export

pod-inference:
	uv run python -m src.pose_object_detection_running.run_inference
