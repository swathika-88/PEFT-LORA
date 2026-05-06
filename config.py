"""
=============================================================
  Configuration  –  RoBERTa + PEFT LoRA
  Task: Lucid Motors Sentiment Classification
  Labels: Positive / Neutral / Negative
=============================================================
"""

# ─── Model ────────────────────────────────────────────────
MODEL_NAME        = "roberta-base"          # or "roberta-large"
MAX_SEQ_LEN       = 128
NUM_LABELS        = 3                       # Positive / Neutral / Negative

# ─── PEFT / LoRA ──────────────────────────────────────────
LORA_R            = 16
LORA_ALPHA        = 32
LORA_DROPOUT      = 0.1
LORA_BIAS         = "none"
LORA_TARGET_MODULES = ["query", "value"]

# ─── Training ─────────────────────────────────────────────
OUTPUT_DIR        = "./lora_roberta_lucid_motors"
LOGGING_DIR       = "./logs"
NUM_EPOCHS        = 5
TRAIN_BATCH_SIZE  = 16
EVAL_BATCH_SIZE   = 32
LEARNING_RATE     = 2e-4
WEIGHT_DECAY      = 0.01
WARMUP_RATIO      = 0.06
LR_SCHEDULER      = "cosine"
GRAD_ACCUMULATION = 1
FP16              = False   # set True on NVIDIA GPU
BF16              = False   # set True on Ampere+ GPU
SAVE_STRATEGY     = "epoch"
EVAL_STRATEGY     = "epoch"
LOAD_BEST_MODEL   = True
METRIC_FOR_BEST   = "f1"
SEED              = 42

# ─── Dataset ──────────────────────────────────────────────
# Priority order:
#   1. Kaggle "EV Talk" CSV     (set KAGGLE_CSV to your downloaded file)
#   2. Local CSV                (place at LOCAL_CSV_PATH)
#   3. Synthetic Lucid Motors   (auto-generated realistic comments)
KAGGLE_CSV        = ""                      # e.g. "ev_youtube_comments.csv"
LOCAL_CSV_PATH    = "./data/lucid_motors_sentiment.csv"
TEXT_COLUMN       = "text"
LABEL_COLUMN      = "sentiment"             # "Positive" | "Neutral" | "Negative"

# Label maps (fixed for 3-class sentiment)
LABEL2ID          = {"Negative": 0, "Neutral": 1, "Positive": 2}
ID2LABEL          = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Train / Val / Test proportions (used when splitting from a single CSV)
TRAIN_RATIO       = 0.70
VAL_RATIO         = 0.15
TEST_RATIO        = 0.15
