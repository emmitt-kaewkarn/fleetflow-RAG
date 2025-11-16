# FleetFlow RAG - Makefile helpers
# Usage:
#   make run          # create venv, install deps, and start Streamlit on port 8501
#   make deps         # (re)install Python dependencies into venv
#   make venv         # create the Python virtual environment
#   make doctor       # print versions (python, pydantic, typing-extensions)
#   make clean        # remove virtual environment

PYTHON ?= python3
VENV    ?= venv
VBIN     = $(VENV)/bin
PIP      = $(VBIN)/pip
PY       = $(VBIN)/python
STREAMLIT= $(VBIN)/streamlit

.PHONY: run deps venv doctor clean

run: deps
	$(STREAMLIT) run app/ui/streamlit_app.py --server.headless true --server.port 8501

deps: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

venv:
	@if [ ! -d "$(VENV)" ]; then \
		$(PYTHON) -m venv $(VENV); \
	fi

doctor: venv
	$(PY) -c "import sys; print('python', sys.version)"
	$(PY) -c "import pydantic; print('pydantic', pydantic.__version__)"
	$(PY) -c "import typing_extensions as te; print('typing-extensions', getattr(te, '__version__', 'unknown'))"

clean:
	rm -rf $(VENV)