"""
=============================================================
  Data Utilities – Lucid Motors Sentiment Dataset
=============================================================
Handles:
  • Loading a real Kaggle EV CSV (if provided)
  • Loading a local CSV file (if present)
  • Generating a realistic synthetic Lucid Motors dataset
  • Train / Val / Test splitting
  • Tokenisation with RoBERTa tokenizer
  • Class-weight computation for imbalanced splits
  • Dataset statistics reporting
"""

import os
import logging
import re
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# 1. Synthetic Lucid Motors Dataset
# ──────────────────────────────────────────────────────────────

# Realistic social-media comments about Lucid Motors
_POSITIVE_TEMPLATES = [
    "The Lucid Air is hands down the best EV on the market right now.",
    "Just test drove the Lucid Air Grand Touring – absolutely blown away by the range!",
    "Lucid Motors is pushing the boundaries of electric vehicle technology.",
    "The interior of the Lucid Air is more luxurious than any Tesla I've seen.",
    "500+ miles of range is no joke. Lucid Air is a game changer.",
    "Lucid's battery technology is years ahead of the competition.",
    "Finally picked up my Lucid Air Pure and couldn't be happier with it!",
    "The Dream Edition is gorgeous. Lucid Motors really nailed the design.",
    "Lucid Air has the best efficiency of any production EV. Impressive engineering.",
    "Customer service at Lucid was phenomenal. Love this company's direction.",
    "The acceleration on the Lucid Air Sapphire is absolutely insane!",
    "Range anxiety is a thing of the past with Lucid Motors.",
    "Lucid Air's frunk and trunk space are incredible for a sedan.",
    "Never thought I'd say this, but the Lucid Air makes Tesla look outdated.",
    "Lucid Motors is proving that American EV startups can compete globally.",
    "The over-the-air updates keep improving the car every few weeks. Love it!",
    "Charging speed on the Lucid Air is among the fastest I've experienced.",
    "Lucid's ADAS system is incredibly smooth on highway drives.",
    "The ambient lighting and glass roof make every drive feel premium.",
    "Lucid Air is worth every penny. Best car purchase I've ever made.",
    "So impressed with the build quality – everything feels solid and premium.",
    "Lucid's thermal management system is top tier – great range in winter too.",
    "The Lucid Air is whisper quiet even at highway speeds. Incredible refinement.",
    "Gravity SUV is going to be a huge hit. Lucid is on fire right now.",
    "My Lucid Air just hit 100,000 miles with zero major issues. Reliability is real.",
]

_NEUTRAL_TEMPLATES = [
    "Lucid Motors delivered {n} vehicles in Q{q} this year.",
    "Lucid Air pricing starts around ${price}k for the base model.",
    "Lucid Motors announced a new service center opening in {city}.",
    "The Lucid Gravity SUV is expected to start deliveries in {year}.",
    "Lucid Motors currently employs around {n} people globally.",
    "Lucid Air competes in the ultra-luxury EV segment above $70,000.",
    "Lucid has a partnership with the Saudi Arabian sovereign wealth fund.",
    "Lucid Motors uses a 900V architecture for its charging system.",
    "The Lucid Air has a drag coefficient of 0.21, one of the lowest in production.",
    "Lucid Motors is headquartered in Newark, California.",
    "Lucid Air is available in Pure, Touring, Grand Touring, and Sapphire trims.",
    "Lucid Motors reported production figures of approximately {n} units last quarter.",
    "The Lucid Gravity offers 7-seat capacity with an SUV form factor.",
    "Lucid Motors stock (LCID) was trading at ${price} today.",
    "Lucid announced an expansion of its manufacturing facility in Arizona.",
    "The Lucid Air weighs approximately 4,800 lbs due to the large battery pack.",
    "Lucid offers a 4-year / 50,000-mile limited warranty on the Air.",
    "Lucid Motors was originally founded in 2007 as Atieva.",
    "The Lucid Air's infotainment uses a dual-screen setup with Android Automotive.",
    "Lucid continues to iterate on software features through regular OTA updates.",
    "Lucid delivered its {n}th vehicle at a ceremony in Arizona.",
    "Lucid Air's EPA-rated range is 516 miles for the Grand Touring trim.",
    "Lucid is opening new retail studios in select US cities this year.",
    "The Lucid Air Pure rear-wheel-drive starts at $69,900 before incentives.",
    "Lucid Motors has been expanding its charging network partnership with EVgo.",
]

_NEGATIVE_TEMPLATES = [
    "Lucid Motors keeps missing delivery targets quarter after quarter. Disappointing.",
    "The Lucid Air's software is glitchy and crashes randomly. Not what you expect at this price.",
    "Still waiting 8 months for my Lucid Air delivery. Communication is terrible.",
    "Lucid Motors burned through cash again. Hard to see a path to profitability.",
    "The touchscreen on my Lucid Air froze twice this week. Unacceptable for a $100k car.",
    "Lucid's service centers are nowhere near where most customers live.",
    "Production numbers are way behind schedule again. Lucid needs to get it together.",
    "Paid $150k for a Lucid Sapphire and the door trim already has gaps. Poor QC.",
    "Lucid Motors LCID stock keeps dropping. Investor confidence is at an all-time low.",
    "The charging network is not mature enough compared to Tesla Supercharger.",
    "My Lucid Air had to be towed twice in the first three months. Very frustrating.",
    "Lucid's customer support takes days to respond to urgent issues. Terrible.",
    "Huge recall on Lucid Air – not a great look for a premium brand.",
    "Lucid keeps overpromising and underdelivering. Tired of the hype.",
    "The Lucid Gravity SUV launch has been delayed yet again. No surprise at this point.",
    "Range drops significantly in cold weather. The EPA numbers are misleading.",
    "Lucid fired hundreds of employees again. Leadership seems directionless.",
    "The autopilot features lag far behind Tesla and Waymo at this price point.",
    "Lucid Air has massive panel gaps on delivery. Quality control is non-existent.",
    "Service appointment wait times are ridiculously long for Lucid customers.",
    "The app is broken more often than it works. Basic features keep failing.",
    "Lucid's resale value has plummeted. Not a smart financial investment.",
    "Supply chain issues keep impacting Lucid's production. When will this end?",
    "Lucid took my deposit 2 years ago and still no delivery date. Awful experience.",
    "The Gravity was announced with great fanfare but deliveries are practically zero.",
]


def _generate_synthetic_dataset(seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic 1,500-row Lucid Motors sentiment dataset.
    500 samples per class (balanced for demo purposes).
    """
    rng = np.random.default_rng(seed)
    rows = []

    fill_values = dict(
        n    = lambda: int(rng.integers(200, 4000)),
        q    = lambda: int(rng.integers(1, 5)),
        price= lambda: int(rng.integers(70, 200)),
        city = lambda: rng.choice(["Los Angeles", "New York", "Chicago",
                                    "Houston", "Miami", "Seattle", "Austin"]),
        year = lambda: rng.choice(["2024", "2025", "2026"]),
    )

    def _fill(tmpl: str) -> str:
        for key, fn in fill_values.items():
            if f"{{{key}}}" in tmpl:
                tmpl = tmpl.replace(f"{{{key}}}", str(fn()))
        return tmpl

    for label, templates in [
        ("Positive", _POSITIVE_TEMPLATES),
        ("Neutral",  _NEUTRAL_TEMPLATES),
        ("Negative", _NEGATIVE_TEMPLATES),
    ]:
        for _ in range(500):
            tmpl = rng.choice(templates)
            text = _fill(tmpl)
            # Add minor paraphrasing noise
            if rng.random() < 0.3:
                text = text + " " + rng.choice([
                    "Thoughts?", "Anyone else?", "#LucidMotors",
                    "#LucidAir", "#EV", "What do you think?",
                    "Really.", "Seriously.", "Just my 2 cents.",
                ])
            rows.append({"text": text, "sentiment": label})

    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────
# 2. Text Cleaning
# ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Lightweight cleaning for social-media text:
      • Collapse whitespace
      • Remove URLs
      • Remove excessive punctuation repetition
      • Normalise mentions/hashtags (keep hashtag text, strip @)
    """
    text = re.sub(r"http\S+|www\.\S+", "", text)          # remove URLs
    text = re.sub(r"@\w+", "", text)                       # remove @mentions
    text = re.sub(r"([!?.]){3,}", r"\1\1", text)           # max 2 repeated punct
    text = re.sub(r"\s+", " ", text).strip()               # collapse whitespace
    return text


# ──────────────────────────────────────────────────────────────
# 3. Dataset Loading Pipeline
# ──────────────────────────────────────────────────────────────

def load_lucid_motors_dataset() -> DatasetDict:
    """
    Load the Lucid Motors sentiment dataset.

    Priority:
      1. Kaggle CSV (config.KAGGLE_CSV) — must have 'text' & 'sentiment' cols
         or will try to auto-detect text/label columns.
      2. Local CSV  (config.LOCAL_CSV_PATH)
      3. Synthetic  (auto-generated realistic Lucid Motors comments)

    Returns
    -------
    DatasetDict with splits: "train", "validation", "test"
    """
    df = None

    # ── 1. Kaggle CSV ──────────────────────────────────────
    if config.KAGGLE_CSV and os.path.exists(config.KAGGLE_CSV):
        logger.info(f"[OK] Loading Kaggle CSV: {config.KAGGLE_CSV}")
        df = pd.read_csv(config.KAGGLE_CSV)
        df = _prepare_kaggle_df(df)

    # ── 2. Local CSV ────────────────────────────────────────
    elif os.path.exists(config.LOCAL_CSV_PATH):
        logger.info(f"[OK] Loading local CSV: {config.LOCAL_CSV_PATH}")
        df = pd.read_csv(config.LOCAL_CSV_PATH)
        df = _prepare_local_df(df)

    # ── 3. Synthetic ────────────────────────────────────────
    else:
        logger.warning(
            "No CSV found. Generating a SYNTHETIC Lucid Motors dataset "
            "(1,500 realistic comments – Positive / Neutral / Negative)."
        )
        df = _generate_synthetic_dataset(seed=config.SEED)

    # Clean text
    df[config.TEXT_COLUMN] = df[config.TEXT_COLUMN].astype(str).apply(clean_text)

    # Encode labels
    df["label"] = df[config.LABEL_COLUMN].map(config.LABEL2ID)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    return _split_and_wrap(df)


def _prepare_kaggle_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a Kaggle EV YouTube comments CSV to our schema.
    Filters for Lucid Motors mentions if the dataset is multi-brand.
    """
    # Auto-detect text column
    text_col = _find_column(df, ["comment", "text", "content", "body", "review"])
    # Auto-detect label column
    label_col = _find_column(df, ["sentiment", "label", "polarity", "rating"])

    if text_col is None or label_col is None:
        raise ValueError(
            f"Could not find text/label columns in {config.KAGGLE_CSV}. "
            f"Available columns: {df.columns.tolist()}"
        )

    df = df[[text_col, label_col]].rename(
        columns={text_col: config.TEXT_COLUMN, label_col: config.LABEL_COLUMN}
    )

    # Filter for Lucid Motors if this is a multi-brand dataset
    lucid_mask = df[config.TEXT_COLUMN].str.lower().str.contains(
        r"lucid|lcid", regex=True, na=False
    )
    if lucid_mask.sum() > 50:
        logger.info(f"Filtering for Lucid Motors mentions: {lucid_mask.sum()} rows")
        df = df[lucid_mask].copy()

    # Normalize label strings
    df[config.LABEL_COLUMN] = df[config.LABEL_COLUMN].astype(str).str.strip().str.capitalize()
    return df


def _prepare_local_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and return a local CSV that already matches our schema."""
    required = {config.TEXT_COLUMN, config.LABEL_COLUMN}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Local CSV is missing columns: {missing}. "
            f"Expected {config.TEXT_COLUMN!r} and {config.LABEL_COLUMN!r}."
        )
    df[config.LABEL_COLUMN] = df[config.LABEL_COLUMN].astype(str).str.strip().str.capitalize()
    return df[[config.TEXT_COLUMN, config.LABEL_COLUMN]]


def _find_column(df: pd.DataFrame, candidates: list) -> Optional[str]:
    """Return the first column name in `candidates` that exists in df."""
    for col in candidates:
        matches = [c for c in df.columns if c.lower() == col.lower()]
        if matches:
            return matches[0]
    return None


def _split_and_wrap(df: pd.DataFrame) -> DatasetDict:
    """Stratified split → HuggingFace DatasetDict."""
    train_df, temp_df = train_test_split(
        df,
        test_size   = config.VAL_RATIO + config.TEST_RATIO,
        stratify    = df["label"],
        random_state= config.SEED,
    )
    val_ratio_adjusted = config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    val_df, test_df   = train_test_split(
        temp_df,
        test_size    = 1 - val_ratio_adjusted,
        stratify     = temp_df["label"],
        random_state = config.SEED,
    )
    logger.info(
        f"Split sizes -> train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    return DatasetDict(
        {
            "train":      Dataset.from_pandas(train_df.reset_index(drop=True)),
            "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
            "test":       Dataset.from_pandas(test_df.reset_index(drop=True)),
        }
    )


# ──────────────────────────────────────────────────────────────
# 4. Tokenisation
# ──────────────────────────────────────────────────────────────

def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = config.MAX_SEQ_LEN,
) -> DatasetDict:
    """Tokenise the text column and keep only model-required columns."""

    def _tokenize(batch):
        return tokenizer(
            batch[config.TEXT_COLUMN],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(_tokenize, batched=True)

    keep_cols = {"input_ids", "attention_mask", "label"}
    for split in tokenized:
        remove = [c for c in tokenized[split].column_names if c not in keep_cols]
        tokenized[split] = tokenized[split].remove_columns(remove)
        tokenized[split].set_format("torch")

    return tokenized


# ──────────────────────────────────────────────────────────────
# 5. Class Weights
# ──────────────────────────────────────────────────────────────

def compute_class_weights(
    dataset: DatasetDict,
    num_labels: int,
) -> Optional[np.ndarray]:
    """Compute balanced class weights from the training split."""
    try:
        train_labels = np.array(dataset["train"]["label"])
        classes      = np.arange(num_labels)
        weights      = compute_class_weight(
            class_weight="balanced", classes=classes, y=train_labels
        )
        logger.info(f"Class weights: {weights}")
        return weights.astype(np.float32)
    except Exception as e:
        logger.warning(f"Could not compute class weights: {e}")
        return None


# ──────────────────────────────────────────────────────────────
# 6. Dataset Statistics
# ──────────────────────────────────────────────────────────────

def print_dataset_stats(dataset: DatasetDict) -> None:
    """Print per-split size and sentiment label distribution."""
    id2label = config.ID2LABEL
    print("\n" + "=" * 60)
    print("  Lucid Motors Sentiment Dataset Statistics")
    print("=" * 60)
    for split_name, split_data in dataset.items():
        print(f"\n  [{split_name.upper()}]  {len(split_data):,} samples")
        if "label" in split_data.column_names:
            labels = np.array(split_data["label"])
            for uid, lname in sorted(id2label.items()):
                cnt = (labels == uid).sum()
                pct = 100 * cnt / len(labels)
                bar = "█" * int(pct / 2)
                print(f"    {lname:<12s} {cnt:>5,}  {pct:5.1f}%  {bar}")
    print("=" * 60 + "\n")


# ──────────────────────────────────────────────────────────────
# 7. Save Synthetic Dataset as CSV (utility)
# ──────────────────────────────────────────────────────────────

def export_synthetic_csv(path: str = config.LOCAL_CSV_PATH) -> None:
    """
    Generate and export the synthetic Lucid Motors dataset as a CSV.
    Useful for inspecting the data before training.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df = _generate_synthetic_dataset(seed=config.SEED)
    df.to_csv(path, index=False)
    logger.info(f"[OK] Synthetic dataset saved -> {path}")
