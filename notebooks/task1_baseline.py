# ============================================================
# Task 1: Zero-Shot Baseline with Qwen3 Prompt Engineering
# Compatible with Google Colab (T4 GPU, 15 GB VRAM)
#
# How to use in Colab:
#   1. Upload this file and train-test-data.xlsx to Colab
#   2. Run sections sequentially, or copy into notebook cells
# ============================================================

# %% [markdown]
# ## 1. Install Dependencies

# %%
# Run this cell first in Colab
# !pip install -q transformers>=4.51.0 accelerate bitsandbytes pandas openpyxl scikit-learn

# %% [markdown]
# ## 2. Imports & Configuration

# %%
import os, re, sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Add src/ to path if running from notebooks/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from metrics import evaluate, print_metrics, parse_labels

# Models to evaluate (4B first, 8B if memory allows)
MODELS = {
    "Qwen3-4B": "Qwen/Qwen3-4B",
    "Qwen3-8B": "Qwen/Qwen3-8B",   # comment out if T4 OOMs
}

DATA_PATH  = "train-test-data.xlsx"   # adjust path as needed
TEST_SIZE  = 236                       # held-out test split (1000 train / 236 test)
RANDOM_SEED = 42
MAX_NEW_TOKENS = 64

print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available(),
      "—", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# %% [markdown]
# ## 3. Load Data

# %%
df = pd.read_excel(DATA_PATH)
assert {"input_finding", "output_disease"}.issubset(df.columns), \
    f"Expected columns not found. Got: {df.columns.tolist()}"

df["input_finding"]  = df["input_finding"].astype(str).str.strip()
df["output_disease"] = df["output_disease"].fillna("").astype(str).str.strip()

# Reproducible 80/20 split — same seed used in Task 2
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_SEED, shuffle=True)
train_df = train_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

print(f"Train: {len(train_df)}  |  Test: {len(test_df)}")
print("Sample:", test_df["output_disease"].iloc[0])

# %% [markdown]
# ## 4. Disease Vocabulary (84 labels)
#
# Providing the complete label set is the core prompt engineering decision:
# the model cannot hallucinate disease names it was never trained to predict.

# %%
ALL_LABELS = [
    "Abdominal abscess", "Abdominal aortic aneurysm", "Abdominal lipiditis",
    "Abdominal mass", "Absent kidney", "Accessory spleen", "Adrenal calcification",
    "Adrenal hyperplasia", "Adrenal mass", "Adrenal metastasis", "Adrenal nodule",
    "Annular pancreas", "Appendicitis", "Arteriosclerosis", "Ascites",
    "Atherosclerosis", "Basal cell carcinoma", "Bile duct cancer", "Bile duct carcinoma",
    "Bile duct dilatation", "Bile duct stone", "Cecal epiploic appendagitis",
    "Cholecystitis", "Cirrhosis", "Colitis", "Colorectal cancer", "Diverticulitis",
    "Enteritis", "Enteritis possible", "Epiploic appendagitis", "Fatty liver",
    "Gallbladder mass", "Gallstone", "Gastric cancer", "Gastric lymphoma",
    "Gastrointestinal bleed", "Gastrointestinal perforation", "Hepatitis", "Hernia",
    "Hydronephrosis", "Inflammatory bowel disease", "Intestinal obstruction",
    "Kidney infarction", "Kidney stone", "Liver abscess", "Liver calcification",
    "Liver cyst", "Liver lesion", "Lymphadenopathy", "Neuroblastoma",
    "No acute abnormality", "Omental inflammation", "Ovarian cyst", "Ovarian mass",
    "Pancreatic atrophy", "Pancreatic calcification", "Pancreatic cancer",
    "Pancreatic cyst", "Pancreatic duct dilatation", "Pancreatic pseudocyst",
    "Pancreatitis", "Peritoneal metastasis", "Peritonitis", "Pheochromocytoma",
    "Pneumoperitoneum", "Polycystic kidney", "Polycystic kidneys", "Portal hypertension",
    "Portal vein thrombosis", "Renal atrophy", "Renal calcification", "Renal cyst",
    "Renal malformation", "Renal mass", "Retroperitoneal fibrosis", "Ruptured kidney",
    "Small intestinal lymphoma", "Splenic calcification", "Splenic cyst",
    "Splenic infarction", "Splenic metastasis", "Splenomegaly", "Traumatic injury",
    "Varices",
]
LABEL_STR = ", ".join(ALL_LABELS)

# Build a normalized lookup for snapping model output to canonical labels
LABEL_NORM_MAP = {l.strip().lower(): l for l in ALL_LABELS}

def snap_to_vocabulary(raw_output: str) -> str:
    """
    Map raw model output to canonical label names.
    Tries exact match (case-insensitive) first; keeps unmatched labels as-is.
    """
    parts = re.split(r"[,，;；\n]", raw_output)
    snapped = []
    for p in parts:
        norm = p.strip().lower()
        if norm in LABEL_NORM_MAP:
            snapped.append(LABEL_NORM_MAP[norm])
        elif norm:
            snapped.append(p.strip())   # keep as-is if no match
    return ", ".join(snapped) if snapped else "No acute abnormality"

# %% [markdown]
# ## 5. Prompt Design
#
# Three design principles:
# 1. **Constrained vocabulary** — list all 84 labels so the model cannot hallucinate
# 2. **Few-shot examples** — 3 diverse examples (1, 2, and 3 labels) to anchor format
# 3. **Explicit rules** — no prose, exact comma-separated output

# %%
# Few-shot examples taken directly from the training split (indices 0-2)
FEWSHOT_EXAMPLES = [
    {
        "finding": (
            "The liver surface is regular with no apparent abnormality in size or morphology. "
            "Multiple high-density foci can be seen in the liver parenchyma, and multiple round "
            "low-density lesions can be seen in the left and right lobes. The common bile duct "
            "and intrahepatic bile duct are not dilated. The spleen is of normal size, morphology, "
            "and density. The position, morphology, and density of the pancreas are normal, and "
            "the pancreatic duct is not dilated. The gallbladder is not enlarged."
        ),
        "output": "Liver lesion",
    },
    {
        "finding": (
            "The liver and spleen are in normal position, size and shape. No abnormal density "
            "found in the parenchyma. The gallbladder is not enlarged and density is uniform. "
            "The pancreas surface is uneven with uniform density; pancreatic duct is slightly "
            "dilated. Slightly high-density foci are found in the parenchyma at the lower poles "
            "of both kidneys. Eccentric thickening of the distal colon wall with uneven enhancement."
        ),
        "output": "Kidney stone, Pancreatitis",
    },
    {
        "finding": (
            "The liver surface is smooth and the lobe proportions are coordinated. Multiple "
            "low-density round lesions are found in the liver. Gallbladder size, wall, and "
            "intracystic density are normal. No bile duct dilation. Spleen and pancreas appear "
            "normal. Multiple low-density circular lesions are seen in the spleen. Multiple "
            "low-density areas are seen within the right kidney."
        ),
        "output": "Liver cyst, Splenic cyst, Renal cyst",
    },
]

SYSTEM_PROMPT = f"""You are a radiologist specializing in CT/MRI diagnosis.

Your task: extract disease labels from the radiology finding below.

RULES:
1. Output ONLY labels from the approved vocabulary — no other words.
2. Separate multiple labels with a comma.
3. If no disease is present, output exactly: No acute abnormality
4. Do NOT explain, do NOT add punctuation beyond commas.

APPROVED VOCABULARY ({len(ALL_LABELS)} labels):
{LABEL_STR}"""

def build_user_message(finding: str, include_fewshot: bool = True) -> str:
    msg = ""
    if include_fewshot:
        msg += "--- Examples ---\n\n"
        for ex in FEWSHOT_EXAMPLES:
            msg += f"Finding: {ex['finding']}\nOutput: {ex['output']}\n\n"
        msg += "--- Now analyze this finding ---\n\n"
    msg += f"Finding: {finding.strip()}\nOutput:"
    return msg

# Quick sanity check
print(SYSTEM_PROMPT[:300])
print("...")
print(build_user_message(test_df["input_finding"].iloc[0])[-200:])

# %% [markdown]
# ## 6. Model Loading

# %%
def load_model(model_id: str):
    """Load a Qwen3 model in 4-bit quantization for inference."""
    print(f"\nLoading {model_id} ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"Loaded. Memory: {torch.cuda.memory_allocated()/1e9:.1f} GB used")
    return tokenizer, model

# %% [markdown]
# ## 7. Inference

# %%
@torch.no_grad()
def run_inference(tokenizer, model, finding: str) -> str:
    """Run one inference call and return the cleaned disease label string."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": build_user_message(finding)},
    ]
    # enable_thinking=False: skip chain-of-thought for faster, deterministic output
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Decode only newly generated tokens (not the prompt)
    new_tokens = out[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    # Strip any residual <think> blocks
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    return snap_to_vocabulary(response)


def run_batch(tokenizer, model, df: pd.DataFrame, desc: str = "") -> pd.DataFrame:
    """Run inference on an entire DataFrame and return results."""
    rows = []
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        pred = run_inference(tokenizer, model, row["input_finding"])
        rows.append({
            "case_id":       row["case_id"],
            "input_finding": row["input_finding"],
            "gold":          row["output_disease"],
            "pred":          pred,
        })
        if (i + 1) % 50 == 0:
            print(f"  {desc} {i+1}/{total}")
    return pd.DataFrame(rows)

# %% [markdown]
# ## 8. Run Evaluation

# %%
all_results = {}

for model_name, model_id in MODELS.items():
    tokenizer, model = load_model(model_id)

    print(f"\nRunning inference: {model_name} on {len(test_df)} test samples ...")
    result_df = run_batch(tokenizer, model, test_df, desc=model_name)
    all_results[model_name] = result_df

    metrics = evaluate(result_df["pred"].tolist(), result_df["gold"].tolist())
    print_metrics(metrics, label=f"Task 1 — {model_name} (zero-shot)")

    # Free memory before loading next model
    del model, tokenizer
    torch.cuda.empty_cache()

# %% [markdown]
# ## 9. Failure Case Analysis

# %%
def analyze_failures(result_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return the n worst predictions by F1 score for manual inspection."""
    from metrics import entity_prf1, parse_labels

    rows = []
    for _, row in result_df.iterrows():
        pred = parse_labels(row["pred"])
        gold = parse_labels(row["gold"])
        _, _, f1 = entity_prf1(pred, gold)
        missed   = gold - pred
        spurious = pred - gold
        rows.append({
            "case_id":  row["case_id"],
            "f1":       round(f1, 3),
            "gold":     row["gold"],
            "pred":     row["pred"],
            "missed":   ", ".join(sorted(missed)),
            "spurious": ", ".join(sorted(spurious)),
        })
    fail_df = pd.DataFrame(rows).sort_values("f1").head(n)
    return fail_df

# Show failure cases for first model
first_model = list(all_results.keys())[0]
fail_df = analyze_failures(all_results[first_model])
print(f"\nTop-{len(fail_df)} worst predictions ({first_model}):")
print(fail_df[["case_id", "f1", "gold", "pred", "missed", "spurious"]].to_string(index=False))

# %% [markdown]
# ## 10. Save Results to Excel

# %%
output_path = "../results/task1_inference_results.xlsx"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    for model_name, result_df in all_results.items():
        # Add per-sample F1 column
        from metrics import entity_prf1, parse_labels
        result_df["f1"] = result_df.apply(
            lambda r: entity_prf1(parse_labels(r["pred"]), parse_labels(r["gold"]))[2],
            axis=1,
        ).round(4)
        sheet_name = model_name.replace("/", "-")[:31]
        result_df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Saved → {output_path}")
