install:
	uv sync

lint:
	uv run ruff check mem0_tcvectordb

test:
	uv run pytest


.PHONY: install lint test
