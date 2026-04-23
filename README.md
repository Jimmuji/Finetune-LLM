# Finetune LLMs for Disease Diagnosis

Fine-tuning Qwen3 for structured disease label extraction from free-text radiology CT/MRI findings.

## Overview

| Task | Method | Model | F1 |
|------|--------|-------|----|
| Baseline | Vocabulary-constrained few-shot prompt | Qwen3-4B | 0.4340 |
| Fine-tuned | QLoRA (r=16) via Unsloth | Qwen3-4B | 0.6734 |

**Fine-tuned model:** [huggingface.co/Jimmu1215/Qwen3-4B-disease-diagnosis](https://huggingface.co/Jimmu1215/Qwen3-4B-disease-diagnosis)

---

## Dataset

- **Total:** 1,236 labeled samples
- **Split:** 1,000 train / 236 test (seed=42)
- **Input:** Free-text CT/MRI radiology findings (`input_finding`)
- **Output:** Comma-separated disease labels (`output_disease`)
- **Vocabulary:** 84 unique disease labels enumerated from the dataset
- **Avg labels/sample:** 2.28

---

## Repository Structure

```
Finetune-LLM/
├── notebooks/
│   ├── task1_baseline.py      # Task 1: zero-shot baseline (Colab-ready)
│   └── task2_finetune.py      # Task 2: QLoRA fine-tuning (Colab-ready)
├── src/
│   └── metrics.py             # Entity-level F1, Jaccard, Exact Match
├── results/
│   ├── Technical_Report.docx
│   ├── task1_Qwen3-4B_results.xlsx
│   └── task2_finetuned_results.xlsx
├── requirements.txt
└── README.md
```

---

## Environment Setup

```bash
# Clone repo
git clone https://github.com/Jimmuji/Finetune-LLM.git
cd Finetune-LLM

# Install dependencies
pip install -r requirements.txt
```

**Google Colab:** Runtime → Change runtime type → T4 GPU

---

## How to Run

### Task 1 — Zero-Shot Baseline

1. Upload `notebooks/task1_baseline.py` and `train-test-data.xlsx` to Colab
2. Run all sections top to bottom
3. Results saved to `results/task1_Qwen3-4B_results.xlsx`

Key prompt design choices:
- **Vocabulary list:** All 84 disease labels included in system prompt — prevents label name mismatch
- **Few-shot examples:** 3 examples (1, 2, and 3-label cases) anchor the output format
- **Thinking disabled:** `enable_thinking=False` gives faster, more consistent output
- **Post-processing:** Model output snapped to canonical vocabulary labels

### Task 2 — QLoRA Fine-Tuning

1. Upload `notebooks/task2_finetune.py` and `train-test-data.xlsx` to Colab
2. Set `HF_REPO_ID` to your HuggingFace repo name at the top of the file
3. Run all sections; training takes ~31 min on T4 for Qwen3-4B
4. Results saved to `results/task2_finetuned_results.xlsx`

Fine-tuning configuration:
| Parameter | Value |
|-----------|-------|
| Framework | Unsloth |
| Base model | Qwen3-4B (4-bit NF4) |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| Target modules | q/k/v/o/gate/up/down proj |
| Optimizer | paged_adamw_8bit |
| Batch size | 2 × 8 accumulation = 16 effective |
| Learning rate | 2e-4 (cosine decay) |
| Epochs | 3 |
| Max seq length | 1024 tokens |
| Training time | ~31 min on NVIDIA T4 |

---

## Evaluation Metric

**Entity-level F1** (macro-averaged, primary metric):

```
Precision = |pred ∩ gold| / |pred|
Recall    = |pred ∩ gold| / |gold|
F1        = 2 × P × R / (P + R)
```

Each comma-separated disease name is treated as one entity. Labels are normalized (lowercase, stripped) before comparison. Also reported: Jaccard similarity and Exact Match rate.

Run evaluation standalone:

```python
from src.metrics import evaluate, print_metrics

metrics = evaluate(predictions=["Renal cyst, Kidney stone"], gold_labels=["Renal cyst"])
print_metrics(metrics)
```

---

## Results

| Model | Setting | Precision | Recall | F1 | Jaccard | Exact Match |
|-------|---------|-----------|--------|----|---------|-------------|
| Qwen3-4B | Zero-shot | 0.4202 | 0.4931 | 0.4340 | 0.3356 | 0.0932 |
| Qwen3-4B | Fine-tuned (QLoRA) | — | — | 0.6734 | — | — |

---

## Reproducibility Checklist

- [x] Fixed random seed (42) for train/test split
- [x] Pinned dependency versions in `requirements.txt`
- [x] All hyperparameters logged in training config
- [x] Model adapter uploaded to HuggingFace
- [x] Inference results saved as Excel (all test samples)
- [x] Same train/test split used for both Task 1 and Task 2

---

## Contact

maojia.wang@mail.utoronto.ca
