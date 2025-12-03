.PHONY: format check all

format:
	clear
	uv run ruff format

check:
	clear
	uv run ruff check

all: format check
