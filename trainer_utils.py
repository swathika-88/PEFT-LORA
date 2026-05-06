"""
=============================================================
  Trainer Utilities – Training & Evaluation Logic
=============================================================
Handles:
  • WeightedLossTrainer (handles class imbalance)
  • Custom compute_metrics (Accuracy, F1, Precision, Recall)
  • TrainingArguments construction
  • EarlyStopping callback
"""

import logging
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import evaluate
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

import config

logger = logging.getLogger(__name__)

# Pre-load evaluation metrics
_accuracy_metric  = evaluate.load("accuracy")
_f1_metric        = evaluate.load("f1")
_precision_metric = evaluate.load("precision")
_recall_metric    = evaluate.load("recall")


# ──────────────────────────────────────────────────────────────
# 1. Compute Metrics Callback
# ──────────────────────────────────────────────────────────────

def make_compute_metrics(num_labels: int):
    """
    Return a compute_metrics function compatible with HF Trainer.

    Uses macro-averaged F1 / Precision / Recall for multi-class
    classification to treat all dialogue-act types equally.
    """
    average = "binary" if num_labels == 2 else "macro"

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        acc = _accuracy_metric.compute(
            predictions=predictions, references=labels
        )
        f1 = _f1_metric.compute(
            predictions=predictions, references=labels, average=average
        )
        prec = _precision_metric.compute(
            predictions=predictions, references=labels, average=average
        )
        rec = _recall_metric.compute(
            predictions=predictions, references=labels, average=average
        )
        return {
            "accuracy":  acc["accuracy"],
            "f1":        f1["f1"],
            "precision": prec["precision"],
            "recall":    rec["recall"],
        }

    return compute_metrics


# ──────────────────────────────────────────────────────────────
# 2. Weighted Loss Trainer
# ──────────────────────────────────────────────────────────────

class WeightedLossTrainer(Trainer):
    """
    HuggingFace Trainer subclass that applies per-class weights
    to the cross-entropy loss.

    Useful when LUCID's dialogue-act distribution is skewed
    (e.g., 'inform' much more frequent than 'greet').
    """

    def __init__(self, class_weights: Optional[np.ndarray] = None, **kwargs):
        super().__init__(**kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(
                class_weights, dtype=torch.float
            )
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits

        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss_fn = nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ──────────────────────────────────────────────────────────────
# 3. TrainingArguments Factory
# ──────────────────────────────────────────────────────────────

def build_training_args(output_dir: str = config.OUTPUT_DIR) -> TrainingArguments:
    """
    Build and return HuggingFace TrainingArguments from config.py.
    """
    return TrainingArguments(
        output_dir                  = output_dir,
        num_train_epochs            = config.NUM_EPOCHS,
        per_device_train_batch_size = config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size  = config.EVAL_BATCH_SIZE,
        learning_rate               = config.LEARNING_RATE,
        weight_decay                = config.WEIGHT_DECAY,
        warmup_ratio                = config.WARMUP_RATIO,
        lr_scheduler_type           = config.LR_SCHEDULER,
        gradient_accumulation_steps = config.GRAD_ACCUMULATION,
        fp16                        = config.FP16,
        bf16                        = config.BF16,
        eval_strategy               = config.EVAL_STRATEGY,
        save_strategy               = config.SAVE_STRATEGY,
        load_best_model_at_end      = config.LOAD_BEST_MODEL,
        metric_for_best_model       = config.METRIC_FOR_BEST,
        greater_is_better           = True,
        logging_dir                 = config.LOGGING_DIR,
        logging_steps               = 50,
        report_to                   = ["tensorboard"],
        seed                        = config.SEED,
        save_total_limit            = 2,
        dataloader_num_workers      = 0,      # 0 = main process (safe on Windows)
    )


# ──────────────────────────────────────────────────────────────
# 4. Trainer Factory
# ──────────────────────────────────────────────────────────────

def build_trainer(
    model,
    training_args: TrainingArguments,
    train_dataset,
    eval_dataset,
    num_labels: int,
    class_weights: Optional[np.ndarray] = None,
    patience: int = 3,
) -> WeightedLossTrainer:
    """
    Construct and return a WeightedLossTrainer with EarlyStopping.

    Parameters
    ----------
    model          : PEFT-wrapped RoBERTa model
    training_args  : TrainingArguments from build_training_args()
    train_dataset  : Tokenized training split
    eval_dataset   : Tokenized validation split
    num_labels     : Number of classification labels
    class_weights  : Optional numpy array of per-class weights
    patience       : Early stopping patience (in eval steps)
    """
    compute_metrics = make_compute_metrics(num_labels)

    trainer = WeightedLossTrainer(
        class_weights = class_weights,
        model         = model,
        args          = training_args,
        train_dataset = train_dataset,
        eval_dataset  = eval_dataset,
        compute_metrics = compute_metrics,
        callbacks     = [
            EarlyStoppingCallback(early_stopping_patience=patience)
        ],
    )
    return trainer
