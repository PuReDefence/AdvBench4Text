# AdvBench4Text

This repository contains the official codebase for our paper "Tougher Text, Smarter Models: Raising the Bar for Adversarial Defence Benchmarks", presented at COLING 2025.

# Getting Started

Clone the repository:

```bash
git clone https://github.com/PuReDefence/AdvBench4Text.git
cd AdvBench4Text
```

Before running the script, set the necessary environment variables:

```bash
export OUTPUT_DIR="/path/to/output_dir"
export HF_HOME="/path/to/huggingface_home"
export PYTHON_PATH="/path/to/python_executable"
```

Run the script:

```bash
bash scripts/text_classification/run_finetune.sh
```