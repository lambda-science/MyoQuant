black:
	black myoquant
	black tests

black-diff:
	black myoquant --diff
	black tests --diff

build:
	poetry build

install:
	poetry install
	pre-commit install

mkserve:
	mkdocs serve

mypy:
	mypy myoquant/ --ignore-missing-imports

pre-commit: black mypy ruff test

publish:
	poetry publish

publish-release: pre-commit build publish

ruff:
	ruff app tests

test:
	pytest tests

_update:
	poetry update

update-all: _update publish-release

update-all-diff:
	poetry update --dry-run | grep -i updat
