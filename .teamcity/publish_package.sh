#!/bin/bash

python3 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip wheel build twine
pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cpu

python -m build

twine upload --repository-url ${NEXUS_REPO} --skip-existing -u ${NEXUS_USERNAME} -p ${NEXUS_PASSWORD} --verbose dist/*
