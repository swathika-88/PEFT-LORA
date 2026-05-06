# RoBERTa + PEFT LoRA Fine-Tuning  
## Task: Lucid Motors Sentiment Classification

Fine-tune **RoBERTa-base** to classify social-media comments, tweets, and reviews
about **Lucid Motors** (the EV company) into three sentiment classes:

| Label | Meaning |
|-------|---------|
| Positive | Praise, satisfaction, excitement about Lucid vehicles / brand |
| Neutral  | Factual statements, news, specifications, stock updates |
| Negative | Complaints, criticism, disappointment, delivery issues |

Uses **Parameter-Efficient Fine-Tuning (PEFT)** via **LoRA** — only ~0.5% of
model parameters are trained, making it fast even on CPU.

---

## 📁 Project Structure

```
LORA/
├── config.py          # All hyper-parameters and dataset settings
├── data_utils.py      # Dataset loading, text cleaning, tokenisation
├── model_utils.py     # RoBERTa + LoRA adapter setup utilities
├── trainer_utils.py   # WeightedLossTrainer, metrics, TrainingArguments
├── eval_utils.py      # Confusion matrix, training curves, F1 plots
├── train.py           # ← Main training script (run this)
├── inference.py       # Classify new comments after training
├── requirements.txt   # Python dependencies
└── README.md
```

---

## ⚙️ Setup

### 1. Create & activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

> **GPU users:** Install CUDA-enabled PyTorch first:
> ```powershell
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```
> Then set `FP16 = True` in `config.py` for faster training.

---

## 📦 Dataset Options

The pipeline tries three sources in order:

### Option A — Kaggle EV YouTube Comments *(Recommended)*
1. Download [EV Talk: YouTube Sentiments Unveiled](https://www.kaggle.com/) from Kaggle
2. Set `KAGGLE_CSV = "path/to/ev_comments.csv"` in `config.py`
3. The loader automatically filters rows mentioning **Lucid** / **LCID**

### Option B — Your Own CSV
Place a CSV at `./data/lucid_motors_sentiment.csv` with columns:

| `text` | `sentiment` |
|--------|------------|
| The Lucid Air range is incredible! | Positive |
| Lucid delivered 1,457 vehicles in Q2 | Neutral |
| Still waiting 9 months for delivery | Negative |

### Option C — Synthetic Dataset *(Default / No data needed)*
If no CSV is found, the pipeline **auto-generates** 1,500 realistic Lucid Motors
comments (500 per class) covering:
- Praise for range, design, build quality, technology
- Factual news, delivery numbers, stock prices, specs
- Complaints about software bugs, delays, service, QC issues

The synthetic data is also exported to `./data/lucid_motors_sentiment.csv`
for inspection.

---

## 🚀 Training

```powershell
python train.py
```

### What happens step-by-step:
1. Load dataset (Kaggle → local CSV → synthetic)
2. Clean text (remove URLs, @mentions, normalize whitespace)
3. Tokenize with RoBERTa's BPE tokenizer
4. Load `roberta-base` + inject LoRA adapters on `query`/`value` layers
5. Train with **weighted cross-entropy** + **early stopping** (patience=3)
6. Evaluate on test split → print Accuracy, F1, Precision, Recall
7. Generate evaluation plots
8. Save adapter checkpoint + merged full model

---

## Inference

**Single comment:**
```powershell
python inference.py --text "The Lucid Air has 516 miles of range – incredible!"
```

**Multiple texts:**
```powershell
python inference.py --text "Great car!" "Delivery delayed again." "LCID stock is flat."
```

**From a text file:**
```powershell
python inference.py --file comments.txt
```

**Interactive mode:**
```powershell
python inference.py
```

**Example output:**
```
  Input     : "The Lucid Air has 516 miles of range – incredible!"
  Sentiment : Positive    (confidence: 94.3%)
  All Scores:
    Positive   0.9430  █████████████████████████
    Neutral    0.0421  █
    Negative   0.0149
```

---

## Key Hyper-parameters (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_NAME` | `roberta-base` | Swap to `roberta-large` for better accuracy |
| `LORA_R` | `16` | LoRA rank (higher = more params, more capacity) |
| `LORA_ALPHA` | `32` | LoRA scaling (keep ≈ 2× rank) |
| `LORA_TARGET_MODULES` | `["query","value"]` | Also try `["query","key","value"]` |
| `LEARNING_RATE` | `2e-4` | Higher than full fine-tuning (standard for LoRA) |
| `NUM_EPOCHS` | `5` | With early stopping; typically converges in 3–4 |
| `TRAIN_BATCH_SIZE` | `16` | Reduce to `8` if running out of memory |
| `MAX_SEQ_LEN` | `128` | Covers most short social-media texts |

---

## Output Files

```
lora_roberta_lucid_motors/
├── lora_adapters/           # Tiny LoRA-only checkpoint (~3 MB)
├── merged_model/            # Full merged model (standard HF format)
├── plots/
│   ├── confusion_matrix.png    # Positive/Neutral/Negative confusion matrix
│   ├── training_curves.png     # Train loss, val loss, val macro-F1
│   └── per_class_f1.png        # Per-class F1 bar chart
├── train_results.json
└── test_results.json

data/
└── lucid_motors_sentiment.csv  # Exported dataset (for inspection)
train.log                       # Full training log
```

---

## 🧠 How LoRA Works

```
Standard Fine-Tuning:  W_new = W_0 + ΔW           (all params trainable)

LoRA:                  W_new = W_0 + (A × B)       (only A, B trainable)
                         where A ∈ R^(d×r), B ∈ R^(r×k), r << d
```

- **Rank `r`** controls adapter size (r=16 → adds only ~1M params to RoBERTa's 125M)
- Backbone is frozen → original knowledge preserved
- A×B can be merged at inference for zero extra latency

---

## 🛠️ Troubleshooting

| Issue | Fix |
|-------|-----|
| `CUDA out of memory` | Reduce `TRAIN_BATCH_SIZE` or `MAX_SEQ_LEN` |
| `Slow on CPU` | Normal — expects ~30 min/epoch on CPU with synthetic data |
| `Kaggle CSV columns not found` | Check column names; pipeline auto-detects common names |
| `Windows dataloader error` | Already handled: `dataloader_num_workers=0` |
| `FP16 error on CPU` | Keep `FP16 = False` (default) when not using CUDA |
