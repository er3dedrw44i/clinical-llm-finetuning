.PHONY: help install test audit train-0.5b train-7b eval-0.5b eval-7b benchmark app

help:
	@echo "Clinical AI Engineering Pipeline - Commands:"
	@echo "  make install      - Install pinned dependencies"
	@echo "  make test         - Run automated unit test suite"
	@echo "  make audit        - Run dataset diversity & token coverage audits"
	@echo "  make train-0.5b   - Run Experiment A baseline training on Apple Silicon MPS"
	@echo "  make train-7b     - Run Experiment B 4-bit NF4 QLoRA training on NVIDIA GPU"
	@echo "  make eval-0.5b    - Evaluate 0.5B model on 1,000 held-out cases"
	@echo "  make eval-7b      - Evaluate 7B QLoRA model on 1,000 held-out cases"
	@echo "  make benchmark    - Run inference latency profiling on Apple Silicon"
	@echo "  make app          - Launch Streamlit Clinical Decision Support UI"

install:
	pip install -r requirements.txt

test:
	python3 -m unittest discover tests

audit:
	python3 prepare_dataset.py
	python3 audit_dataset.py
	python3 audit_truncation.py

train-0.5b:
	python3 train.py

train-7b:
	python3 qlora_train.py

eval-0.5b:
	python3 evaluate_model.py

eval-7b:
	python3 evaluate_7b.py

benchmark:
	python3 benchmark.py

app:
	streamlit run app.py
