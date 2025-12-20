setup:
	curl -LsSf https://astral.sh/uv/install.sh | sh
	uv pip install swig
	uv sync

format:
	uv run ruff format

check:
	uv run ruff check
