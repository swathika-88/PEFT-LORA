"""
=============================================================
  train.py  --  Main Fine-Tuning Script
  RoBERTa + PEFT LoRA  --  Lucid Motors Sentiment Classification
  Labels: Positive / Neutral / Negative
=============================================================

Usage
-----
  python train.py

All hyper-parameters are controlled via config.py.

Dataset source priority (set in config.py):
  1. Kaggle EV YouTube comments CSV (KAGGLE_CSV)
  2. Local CSV                       (LOCAL_CSV_PATH)
  3. Synthetic Lucid Motors dataset  (auto-generated, 1,500 comments)
"""

import io
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Force UTF-8 on Windows so emoji in print/log do not crash cp1252 ──
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import torch

import config
from data_utils import (
    load_lucid_motors_dataset,
    tokenize_dataset,
    compute_class_weights,
    print_dataset_stats,
    export_synthetic_csv,
)
from model_utils import (
    load_tokenizer,
    load_base_model,
    apply_lora,
    save_peft_model,
    merge_and_save,
    get_device,
)
from trainer_utils import build_training_args, build_trainer
from eval_utils import (
    print_classification_report,
    plot_confusion_matrix,
    plot_training_curves,
    plot_per_class_f1,
)

# ─── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("train.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    set_seed(config.SEED)
    device = get_device()

    # ── Banner ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RoBERTa + PEFT LoRA  –  Lucid Motors Sentiment Analysis")
    print("=" * 65)
    print(f"  Base Model  : {config.MODEL_NAME}")
    print(f"  LoRA Rank   : r={config.LORA_R}, α={config.LORA_ALPHA}")
    print(f"  Device      : {device}")
    print(f"  Epochs      : {config.NUM_EPOCHS}")
    print(f"  LR          : {config.LEARNING_RATE}")
    print(f"  Classes     : Negative / Neutral / Positive")
    print("=" * 65 + "\n")

    # ─────────────────────────────────────────────────────────
    # Step 1: Load Dataset
    # ─────────────────────────────────────────────────────────
    logger.info("Step 1/6 – Loading Lucid Motors sentiment dataset")
    dataset = load_lucid_motors_dataset()
    print_dataset_stats(dataset)

    # Also export synthetic data so users can inspect it
    if not os.path.exists(config.LOCAL_CSV_PATH):
        export_synthetic_csv(config.LOCAL_CSV_PATH)

    # ─────────────────────────────────────────────────────────
    # Step 2: Tokenise
    # ─────────────────────────────────────────────────────────
    logger.info("Step 2/6 – Tokenising with RoBERTa tokenizer")
    tokenizer = load_tokenizer()
    tokenized = tokenize_dataset(dataset, tokenizer)

    train_ds = tokenized["train"]
    val_ds   = tokenized["validation"]
    test_ds  = tokenized["test"]

    # ─────────────────────────────────────────────────────────
    # Step 3: Build Model
    # ─────────────────────────────────────────────────────────
    logger.info("Step 3/6 – Building RoBERTa + LoRA model")
    base_model = load_base_model(
        num_labels = config.NUM_LABELS,
        label2id   = config.LABEL2ID,
        id2label   = config.ID2LABEL,
    )
    peft_model = apply_lora(base_model)

    # ─────────────────────────────────────────────────────────
    # Step 4: Train
    # ─────────────────────────────────────────────────────────
    logger.info("Step 4/6 – Starting fine-tuning")
    class_weights = compute_class_weights(dataset, config.NUM_LABELS)
    training_args = build_training_args()
    trainer = build_trainer(
        model         = peft_model,
        training_args = training_args,
        train_dataset = train_ds,
        eval_dataset  = val_ds,
        num_labels    = config.NUM_LABELS,
        class_weights = class_weights,
        patience      = 3,
    )

    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    logger.info(f"Training complete - runtime: {metrics.get('train_runtime', 0):.1f}s")

    # ─────────────────────────────────────────────────────────
    # Step 5: Evaluate on Test Set
    # ─────────────────────────────────────────────────────────
    logger.info("Step 5/6 – Evaluating on test split")
    test_metrics = trainer.evaluate(test_ds)
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)

    print("\n" + "=" * 65)
    print("  Test Set Results  –  Lucid Motors Sentiment")
    print("=" * 65)
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k:<35s} {v:.4f}")
    print("=" * 65 + "\n")

    # ─────────────────────────────────────────────────────────
    # Step 6: Detailed Evaluation + Plots
    # ─────────────────────────────────────────────────────────
    logger.info("Step 6/6 – Generating evaluation plots")
    predictions = trainer.predict(test_ds)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids

    print_classification_report(y_true, y_pred, config.ID2LABEL)

    plots_dir = os.path.join(config.OUTPUT_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_confusion_matrix(
        y_true, y_pred, config.ID2LABEL,
        save_path=os.path.join(plots_dir, "confusion_matrix.png"),
    )
    plot_training_curves(
        trainer.state.log_history,
        save_path=os.path.join(plots_dir, "training_curves.png"),
    )
    plot_per_class_f1(
        y_true, y_pred, config.ID2LABEL,
        save_path=os.path.join(plots_dir, "per_class_f1.png"),
    )

    # ─────────────────────────────────────────────────────────
    # Step 7: Save Checkpoint
    # ─────────────────────────────────────────────────────────
    adapter_path = os.path.join(config.OUTPUT_DIR, "lora_adapters")
    save_peft_model(peft_model, adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info(f"[DONE] LoRA adapters saved -> {adapter_path}")

    merged_path = os.path.join(config.OUTPUT_DIR, "merged_model")
    merge_and_save(peft_model, merged_path)
    tokenizer.save_pretrained(merged_path)
    logger.info(f"[DONE] Merged model saved  -> {merged_path}")

    print("\n" + "=" * 65)
    print("  Fine-tuning complete!")
    print(f"  LoRA adapters  : {adapter_path}")
    print(f"  Merged model   : {merged_path}")
    print(f"  Plots          : {plots_dir}")
    print(f"  Exported data  : {config.LOCAL_CSV_PATH}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
