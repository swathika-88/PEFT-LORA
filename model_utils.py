"""
=============================================================
  Model Utilities – RoBERTa + PEFT LoRA Configuration
=============================================================
Handles:
  • Loading RoBERTa for sequence classification
  • Applying LoRA adapters via PEFT
  • Printing trainable parameter statistics
  • Saving / loading the merged model
"""

import logging
from typing import Dict, Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
    PeftConfig,
)

import config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# 1. Tokenizer
# ──────────────────────────────────────────────────────────────

def load_tokenizer(model_name: str = config.MODEL_NAME) -> PreTrainedTokenizerBase:
    """Load and return the RoBERTa tokenizer."""
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return tokenizer


# ──────────────────────────────────────────────────────────────
# 2. Base Model
# ──────────────────────────────────────────────────────────────

def load_base_model(
    model_name: str = config.MODEL_NAME,
    num_labels: int = 2,
    label2id: Optional[Dict[str, int]] = None,
    id2label: Optional[Dict[int, str]] = None,
) -> PreTrainedModel:
    """
    Load RoBERTa with a classification head.

    Parameters
    ----------
    model_name  : HuggingFace model identifier
    num_labels  : Number of target classes
    label2id    : Mapping label string → int
    id2label    : Mapping int → label string
    """
    logger.info(f"Loading base model: {model_name}  (num_labels={num_labels})")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,   # adapts head to num_labels
    )
    return model


# ──────────────────────────────────────────────────────────────
# 3. LoRA Configuration & PEFT Model
# ──────────────────────────────────────────────────────────────

def build_lora_config() -> LoraConfig:
    """
    Construct the LoRA configuration.

    Key LoRA hyper-parameters (all in config.py):
      r           – rank of the low-rank matrices
      lora_alpha  – scaling factor (effective LR ∝ α/r)
      lora_dropout– dropout on LoRA layers
      target_modules – which Linear layers to inject adapters into
    """
    lora_cfg = LoraConfig(
        task_type       = TaskType.SEQ_CLS,          # sequence classification
        r               = config.LORA_R,
        lora_alpha      = config.LORA_ALPHA,
        lora_dropout    = config.LORA_DROPOUT,
        bias            = config.LORA_BIAS,
        target_modules  = config.LORA_TARGET_MODULES,
        inference_mode  = False,
    )
    logger.info(
        f"LoRA config -> r={config.LORA_R}, alpha={config.LORA_ALPHA}, "
        f"dropout={config.LORA_DROPOUT}, targets={config.LORA_TARGET_MODULES}"
    )
    return lora_cfg


def apply_lora(model: PreTrainedModel) -> PeftModel:
    """
    Wrap the base model with LoRA adapters using PEFT.

    Only the adapter weights are trainable; all backbone
    parameters are frozen automatically.
    """
    lora_cfg = build_lora_config()
    peft_model = get_peft_model(model, lora_cfg)
    print_trainable_parameters(peft_model)
    return peft_model


# ──────────────────────────────────────────────────────────────
# 4. Parameter Statistics
# ──────────────────────────────────────────────────────────────

def print_trainable_parameters(model: PreTrainedModel) -> None:
    """
    Print the number of trainable vs. total parameters.
    Helps confirm that LoRA is correctly freezing the backbone.
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params     = sum(p.numel() for p in model.parameters())
    pct              = 100 * trainable_params / total_params

    print("\n" + "=" * 60)
    print("  LoRA Trainable Parameter Summary")
    print("=" * 60)
    print(f"  Trainable params : {trainable_params:>12,}  ({pct:.4f}%)")
    print(f"  Frozen params    : {total_params - trainable_params:>12,}")
    print(f"  Total params     : {total_params:>12,}")
    print("=" * 60 + "\n")


# ──────────────────────────────────────────────────────────────
# 5. Save / Load Utilities
# ──────────────────────────────────────────────────────────────

def save_peft_model(model: PeftModel, save_path: str) -> None:
    """
    Save only the LoRA adapter weights (tiny checkpoint).
    The backbone weights are NOT saved – they are loaded
    from HuggingFace Hub at inference time.
    """
    logger.info(f"Saving LoRA adapters → {save_path}")
    model.save_pretrained(save_path)


def load_peft_model(
    adapter_path: str,
    num_labels: int,
    label2id: Optional[Dict[str, int]] = None,
    id2label:  Optional[Dict[int, str]] = None,
) -> PeftModel:
    """
    Reload the PEFT model from a saved adapter checkpoint.
    Loads the base model first, then attaches the adapters.
    """
    peft_config = PeftConfig.from_pretrained(adapter_path)
    base_model  = load_base_model(
        model_name = peft_config.base_model_name_or_path,
        num_labels = num_labels,
        label2id   = label2id,
        id2label   = id2label,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    logger.info(f"[OK] LoRA model loaded from {adapter_path}")
    return model


def merge_and_save(model: PeftModel, save_path: str) -> None:
    """
    Merge LoRA adapter weights into the backbone and save the
    full model. The resulting checkpoint can be loaded with
    AutoModelForSequenceClassification.from_pretrained() directly.
    """
    logger.info("Merging LoRA adapters into base model …")
    merged = model.merge_and_unload()
    merged.save_pretrained(save_path)
    logger.info(f"[OK] Merged model saved -> {save_path}")


# ──────────────────────────────────────────────────────────────
# 6. Device Helper
# ──────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Return the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS backend.")
    else:
        device = torch.device("cpu")
        logger.info("No GPU found - running on CPU (will be slow).")
    return device
