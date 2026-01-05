.PHONY: train export

train:
	uv run python -m src.train

export-weights:
	uv run python -m src.export
