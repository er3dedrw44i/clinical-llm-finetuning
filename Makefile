.PHONY: help install test audit train eval benchmark export app demo

help:
	@echo "Clinical AI 7B QLoRA Pipeline - Commands:"
	@echo "  make install      - Install pinned dependencies"
	@echo "  make test         - Run automated unit test suite"
	@echo "  make audit        - Run dataset diversity & token coverage audits"
	@echo "  make train        - Run 7B 4-bit NF4 QLoRA SFT training loop"
	@echo "  make eval         - Run 7B evaluation on 1,000 held-out cases"
	@echo "  make benchmark    - Benchmark 7B inference latency & throughput"
	@echo "  make export       - Fuse LoRA weights via merge_and_unload()"
	@echo "  make app          - Launch Streamlit Clinical Decision Support UI"
	@echo "  make demo         - Run interactive CLI terminal demo"

install:
	pip install -r requirements.txt

test:
	python3 -m unittest discover tests

audit:
	python3 prepare_dataset.py
	python3 audit_dataset.py
	python3 audit_truncation.py

train:
	python3 qlora_train.py

eval:
	python3 evaluate_7b.py

benchmark:
	python3 benchmark.py

export:
	python3 export_model.py

app:
	streamlit run app.py

demo:
	python3 interactive_demo.py
