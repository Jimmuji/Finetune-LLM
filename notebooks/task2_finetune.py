# ============================================================
# Task 2: Efficient Fine-Tuning with QLoRA (Unsloth)
# Compatible with Google Colab (T4 GPU, 15 GB VRAM)
#
# Uses Unsloth for 2× faster training and ~30% less VRAM vs
# raw transformers+peft — critical for Colab free tier.
#
# Model recommendations:
#   Colab T4 (15 GB): Qwen3-4B  ← reliable
#   Colab L4 / A100:  Qwen3-8B  ← recommended for best results
# ============================================================

# %% [markdown]
# ## 1. Install Dependencies (Colab)

# %%
# Run once per Colab session
# %%capture
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps "trl>=0.9.0" peft accelerate bitsandbytes
# !pip install pandas openpyxl scikit-learn

# %% [markdown]
# ## 2. Imports & Configuration

# %%
import os, re, sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from metrics import evaluate, print_metrics

# ── Hyperparameters ─────────────────────────────────────────
MODEL_ID     = "unsloth/Qwen3-4B-bnb-4bit"   # change to Qwen3-8B if using A100
HF_REPO_ID   = "YOUR_HF_USERNAME/Qwen3-4B-disease-diagnosis"  # set before running
DATA_PATH    = "train-test-data.xlsx"
OUTPUT_DIR   = "qwen3_lora_ckpt"

MAX_SEQ_LEN  = 1024    # findings are ~150 words; 512 tokens is enough
LORA_RANK    = 16
LORA_ALPHA   = 16      # alpha = rank keeps effective scale at 1
LORA_DROPOUT = 0.05

TRAIN_EPOCHS = 3
BATCH_SIZE   = 2
GRAD_ACCUM   = 8       # effective batch = 16
LR           = 2e-4
WARMUP_RATIO = 0.05
TEST_SIZE    = 236
RANDOM_SEED  = 42      # same seed as Task 1 for identical split

print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available(),
      "—", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# %% [markdown]
# ## 3. Load & Split Data

# %%
df = pd.read_excel(DATA_PATH)
df["input_finding"]  = df["input_finding"].astype(str).str.strip()
df["output_disease"] = df["output_disease"].fillna("").astype(str).str.strip()

train_df, test_df = train_test_split(
    df, test_size=TEST_SIZE, random_state=RANDOM_SEED, shuffle=True
)
train_df = train_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

print(f"Train: {len(train_df)}  |  Test: {len(test_df)}")

# %% [markdown]
# ## 4. Format Training Data for SFT
#
# Each sample is formatted as a chat conversation:
#   system → task instructions
#   user   → radiology finding
#   assistant → disease labels  (the target tokens the model must learn)

# %%
SYSTEM_PROMPT = (
    "You are a radiologist specializing in CT/MRI diagnosis. "
    "Extract disease labels from the following radiology finding. "
    "Output ONLY comma-separated disease names, nothing else. "
    "If no disease is present, output: No acute abnormality"
)

def format_sample(row: pd.Series) -> dict:
    """Convert a data row to the conversation format expected by SFTTrainer."""
    finding = row["input_finding"]
    labels  = row["output_disease"] if row["output_disease"] else "No acute abnormality"
    return {
        "conversations": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": f"Finding: {finding}"},
            {"role": "assistant", "content": labels},
        ]
    }

train_data = [format_sample(row) for _, row in train_df.iterrows()]
test_data  = [format_sample(row) for _, row in test_df.iterrows()]

print("Sample formatted train example:")
print(train_data[0])

# %% [markdown]
# ## 5. Load Base Model with Unsloth

# %%
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,          # auto-detect (bfloat16 on Ampere+)
    load_in_4bit=True,   # QLoRA base: 4-bit NF4
)

# %% [markdown]
# ## 6. Apply LoRA Adapters

# %%
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
    use_gradient_checkpointing="unsloth",   # saves ~30% VRAM
    random_state=RANDOM_SEED,
)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
      f"({100 * trainable_params / total_params:.2f}%)")

# %% [markdown]
# ## 7. Training

# %%
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

train_dataset = Dataset.from_list(train_data)
eval_dataset  = Dataset.from_list(test_data[:50])   # quick eval during training

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=TRAIN_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=WARMUP_RATIO,
    max_seq_length=MAX_SEQ_LEN,
    optim="paged_adamw_8bit",
    bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
    fp16=not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8),
    gradient_checkpointing=True,
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
    dataset_text_field=None,         # we use conversations format
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

print("Starting training ...")
train_result = trainer.train()
print("\nTraining complete.")
print(f"  Total steps:     {train_result.global_step}")
print(f"  Train loss:      {train_result.training_loss:.4f}")
print(f"  Runtime (min):   {train_result.metrics['train_runtime']/60:.1f}")

# %% [markdown]
# ## 8. Save Adapter & Push to HuggingFace

# %%
# Save adapter locally
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Adapter saved to ./{OUTPUT_DIR}/")

# Push to HuggingFace Hub
# Set HF_REPO_ID at the top of this file before running
from huggingface_hub import login
# login()   # uncomment and run interactively, or set HF_TOKEN env var
model.push_to_hub(HF_REPO_ID, token=os.environ.get("HF_TOKEN"))
tokenizer.push_to_hub(HF_REPO_ID, token=os.environ.get("HF_TOKEN"))
print(f"Model pushed to: https://huggingface.co/{HF_REPO_ID}")

# %% [markdown]
# ## 9. Inference with Fine-Tuned Model

# %%
# Switch model to fast inference mode (unsloth optimization)
FastLanguageModel.for_inference(model)

@torch.no_grad()
def run_inference_ft(finding: str) -> str:
    """Run inference with the fine-tuned model."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Finding: {finding}"},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = out[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    return response


def run_batch_ft(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        pred = run_inference_ft(row["input_finding"])
        rows.append({
            "case_id":       row["case_id"],
            "input_finding": row["input_finding"],
            "gold":          row["output_disease"],
            "pred":          pred,
        })
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(df)}")
    return pd.DataFrame(rows)

print(f"Running inference on {len(test_df)} test samples ...")
ft_results = run_batch_ft(test_df)
ft_metrics = evaluate(ft_results["pred"].tolist(), ft_results["gold"].tolist())
print_metrics(ft_metrics, label=f"Task 2 — {MODEL_ID} (fine-tuned)")

# %% [markdown]
# ## 10. Compare Baseline vs Fine-Tuned

# %%
# Load Task 1 results if available
task1_path = "../results/task1_inference_results.xlsx"
comparison_rows = []

if Path(task1_path).exists():
    task1_xl = pd.read_excel(task1_path, sheet_name=None)
    for sheet, df_t1 in task1_xl.items():
        m = evaluate(df_t1["pred"].tolist(), df_t1["gold"].tolist())
        comparison_rows.append({
            "Model":       sheet,
            "Setting":     "Zero-shot",
            "Precision":   round(m["precision"],   4),
            "Recall":      round(m["recall"],       4),
            "F1":          round(m["f1"],           4),
            "Jaccard":     round(m["jaccard"],      4),
            "Exact Match": round(m["exact_match"],  4),
        })

comparison_rows.append({
    "Model":       MODEL_ID.split("/")[-1],
    "Setting":     "Fine-tuned (QLoRA)",
    "Precision":   round(ft_metrics["precision"],   4),
    "Recall":      round(ft_metrics["recall"],       4),
    "F1":          round(ft_metrics["f1"],           4),
    "Jaccard":     round(ft_metrics["jaccard"],      4),
    "Exact Match": round(ft_metrics["exact_match"],  4),
})

summary_df = pd.DataFrame(comparison_rows)
print("\n=== Model Comparison ===")
print(summary_df.to_string(index=False))

# %% [markdown]
# ## 11. Save All Results to Excel

# %%
output_path = "../results/task2_inference_results.xlsx"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

from metrics import entity_prf1, parse_labels

ft_results["f1"] = ft_results.apply(
    lambda r: entity_prf1(parse_labels(r["pred"]), parse_labels(r["gold"]))[2],
    axis=1,
).round(4)

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    ft_results.to_excel(writer, sheet_name="ft_predictions", index=False)
    summary_df.to_excel(writer, sheet_name="summary", index=False)

print(f"Saved → {output_path}")

# %% [markdown]
# ## 12. Failure Case Analysis (Fine-Tuned Model)

# %%
def analyze_failures(result_df, n=10):
    rows = []
    for _, row in result_df.iterrows():
        pred = parse_labels(row["pred"])
        gold = parse_labels(row["gold"])
        _, _, f1 = entity_prf1(pred, gold)
        rows.append({
            "case_id":  row["case_id"],
            "f1":       round(f1, 3),
            "gold":     row["gold"],
            "pred":     row["pred"],
            "missed":   ", ".join(sorted(gold - pred)),
            "spurious": ", ".join(sorted(pred - gold)),
        })
    return pd.DataFrame(rows).sort_values("f1").head(n)

fail_df = analyze_failures(ft_results)
print(f"\nTop-{len(fail_df)} hardest cases (fine-tuned model):")
print(fail_df.to_string(index=False))
