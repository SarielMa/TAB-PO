# TAB-PO: Preference Optimization with a Token-Level Adaptive Barrier for Token-Critical Structured Generation

Official implementation of the paper:

**TAB-PO: Preference Optimization with a Token-Level Adaptive Barrier for Token-Critical Structured Generation**  
https://arxiv.org/abs/2603.00025

TAB-PO is a preference optimization framework designed for **token-critical structured generation tasks**, where certain tokens (such as labels, codes, or spans) carry critical semantic meaning. Errors in these tokens can significantly degrade downstream structured extraction performance. TAB-PO introduces a **token-level adaptive barrier mechanism** that increases learning pressure on semantically important tokens during preference optimization.

This repository contains the training pipeline, inference scripts, and evaluation utilities used in the paper.

---

# Overview

Large language models often struggle with structured generation tasks where the correctness of **specific tokens** is crucial. Standard preference optimization approaches such as DPO treat all tokens uniformly, which can lead to suboptimal training behavior when only certain tokens determine correctness.

TAB-PO addresses this limitation by introducing:

- **Token-level weighting**
- **Adaptive barrier constraints**
- **Difference-token emphasis**
- **Efficient log-probability computation**

These mechanisms focus the optimization process on **token-critical regions**, improving reliability in structured prediction tasks.

---

# Repository Structure

```
TAB-PO
├── train_dpo.py
├── prepare_dpo_data.py
├── infer_vllm_and_confusion.py
├── merge_lora.py
├── pv_utils.py
├── apply_server_DPO.sh
├── dpo_from_confusion_to_eval_all.sh
├── environment.yml
└── README.md
```

Main components:

- **train_dpo.py**  
  Implements TAB-PO training based on preference optimization.

- **prepare_dpo_data.py**  
  Converts structured annotation outputs into preference training pairs.

- **infer_vllm_and_confusion.py**  
  Runs model inference and produces confusion statistics.

- **merge_lora.py**  
  Merges LoRA adapters into the base model checkpoint.

- **pv_utils.py**  
  Utility functions used for structured extraction evaluation.

- **apply_server_DPO.sh**  
  Example training script.

- **dpo_from_confusion_to_eval_all.sh**  
  Pipeline script for running inference and evaluation.

---

# Installation

Create the environment using the provided configuration:

```
conda env create -f environment.yml
conda activate finben_vllm3
```

Main dependencies include:

- PyTorch  
- HuggingFace Transformers  
- PEFT  
- Datasets  
- vLLM  

---

# Inference on Previous SFT Model and Confusion Statistics

To run inference using a previously trained SFT model and generate confusion statistics:

```
python infer_vllm_and_confusion.py
```

This script performs:

- Model inference (via vLLM)
- Structured output parsing
- Confusion and error analysis

---

# Preparing Training Data

Training data must be converted into **preference pairs** consisting of:

- chosen output  
- rejected output  

Use:

```
python prepare_dpo_data.py
```

This script converts structured extraction outputs into the format required for preference optimization.

---

# Training with TAB-PO

Training is performed using:

```
python train_dpo.py
```

Key training features include:

- Token-weighted preference optimization
- Adaptive token-level barrier
- Prompt masking
- LoRA fine-tuning for large models

Example training command:

```
bash apply_server_DPO.sh
```

---

# LoRA Merge

After training, merge the LoRA adapters into the base model:

```
python merge_lora.py
```

This produces a standalone checkpoint suitable for inference or evaluation.

---

# Evaluation

Evaluation is performed using **FinBen**, a benchmark framework for evaluating LLMs.

Official repository:

https://github.com/The-FinAI/finlm_eval

For this project we provide a **modified setup**:

https://github.com/SarielMa/finben_modified

Please follow the installation instructions in that repository to configure FinBen.

The evaluation utilities provided in this repository (such as `pv_utils.py`) are compatible with the FinBen evaluation pipeline.

---

# Whole Pipeline

Configure the following paths first:

- Training data path  
- FinBen path  
- Previous SFT model path  
- TAB-PO model output path  

Then run the full pipeline with one command:

```
sh dpo_from_confusion_to_eval_all.sh
```

This script executes the entire workflow, including:

1. Inference on the previous SFT model  
2. Confusion analysis  
3. Preference data construction  
4. TAB-PO training  
5. Evaluation using FinBen  

---

# Pretrained Models

The pretrained TAB-PO models used in the paper are available on Hugging Face:

```
lm2445/TABPO_llama3.3_70B_3epoch
lm2445/TABPO_llama3.2_3B_3epoch
lm2445/TABPO_llama3.1_8B_3epoch
lm2445/TABPO_qwen2.5_1.5B_3epoch
```

These models were trained using the TAB-PO preference optimization framework.

---

# Citation

If you use this work, please cite:

```
@misc{fodeh2026tabpopreferenceoptimizationtokenlevel,
      title={TAB-PO: Preference Optimization with a Token-Level Adaptive Barrier for Token-Critical Structured Generation}, 
      author={Samah Fodeh and Linhai Ma and Ganesh Puthiaraju and Srivani Talakokkul and Afshan Khan and Ashley Hagaman and Sarah R. Lowe and Aimee Kendall Roundtree},
      year={2026},
      eprint={2603.00025},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.00025}
}
```

---

# License

This repository is released for research purposes. Please refer to the repository license for details.