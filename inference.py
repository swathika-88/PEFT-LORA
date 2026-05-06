"""
=============================================================
  inference.py  –  Lucid Motors Sentiment Classifier
=============================================================
Classifies text (tweets / YouTube comments / reviews) about
Lucid Motors into: Positive / Neutral / Negative

Usage
-----
  python inference.py --text "The Lucid Air has amazing range!"
  python inference.py --file comments.txt
  python inference.py   # interactive mode
"""

import argparse
import logging
import sys
from typing import List, Dict

import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig

import config
from model_utils import load_peft_model, get_device

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Label prefixes for display (ASCII-safe)
_SENTIMENT_PREFIX = {
    "Positive": "[+]",
    "Neutral":  "[ ]",
    "Negative": "[-]",
}


class LucidMotorsSentimentClassifier:
    """Classify Lucid Motors comments/reviews as Positive, Neutral, or Negative."""

    def __init__(self, adapter_path: str = None):
        self.adapter_path = adapter_path or f"{config.OUTPUT_DIR}/lora_adapters"
        self.device       = get_device()

        print(f"\nLoading model from: {self.adapter_path}")
        model_cfg  = AutoConfig.from_pretrained(self.adapter_path)
        self.id2label  = {int(k): v for k, v in model_cfg.id2label.items()}
        self.label2id  = {v: int(k) for k, v in model_cfg.id2label.items()}
        num_labels     = model_cfg.num_labels

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.adapter_path, use_fast=True
        )
        self.model = load_peft_model(
            adapter_path = self.adapter_path,
            num_labels   = num_labels,
            label2id     = self.label2id,
            id2label     = self.id2label,
        )
        self.model.eval()
        self.model.to(self.device)
        print("[OK] Model ready.\n")

    @torch.inference_mode()
    def predict(self, texts: List[str]) -> List[Dict]:
        """
        Classify a list of texts.

        Returns list of dicts:
            text        : original input
            sentiment   : "Positive" | "Neutral" | "Negative"
            confidence  : probability of predicted class
            scores      : dict of all label → probability
        """
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=config.MAX_SEQ_LEN,
            return_tensors="pt",
        ).to(self.device)

        logits   = self.model(**inputs).logits
        probs    = torch.softmax(logits, dim=-1).cpu().numpy()
        pred_ids = np.argmax(probs, axis=-1)

        results = []
        for i, text in enumerate(texts):
            label = self.id2label[pred_ids[i]]
            results.append({
                "text":       text,
                "sentiment":  label,
                "confidence": float(probs[i, pred_ids[i]]),
                "scores": {
                    self.id2label[j]: float(probs[i, j])
                    for j in sorted(self.id2label)
                },
            })
        return results

    def predict_one(self, text: str) -> Dict:
        return self.predict([text])[0]

    def _print_result(self, r: Dict) -> None:
        prefix = _SENTIMENT_PREFIX.get(r["sentiment"], "")
        print(f'\n  Input     : "{r["text"]}"')
        print(f'  Sentiment : {prefix} {r["sentiment"]:<10s}  '
              f'(confidence: {r["confidence"]:.1%})')
        print("  All Scores:")
        for label, score in sorted(r["scores"].items(),
                                    key=lambda x: x[1], reverse=True):
            bar = "|" * int(score * 25)
            p   = _SENTIMENT_PREFIX.get(label, "   ")
            print(f"    {p} {label:<10s} {score:.4f}  {bar}")

    def interactive(self) -> None:
        print("=" * 65)
        print("  Lucid Motors Sentiment Classifier  (type 'quit' to exit)")
        print("  Enter any comment, tweet, or review about Lucid Motors")
        print("=" * 65)
        while True:
            try:
                text = input("\n  Comment: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break
            if text.lower() in {"quit", "exit", "q"}:
                break
            if not text:
                continue
            self._print_result(self.predict_one(text))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lucid Motors Sentiment Classifier (RoBERTa + LoRA)"
    )
    parser.add_argument(
        "--adapter_path", type=str,
        default=f"{config.OUTPUT_DIR}/lora_adapters",
        help="Path to saved LoRA adapter directory.",
    )
    parser.add_argument(
        "--text", type=str, nargs="+",
        help="One or more texts to classify.",
    )
    parser.add_argument(
        "--file", type=str,
        help="Text file with one comment per line.",
    )
    return parser.parse_args()


def main():
    args       = parse_args()
    classifier = LucidMotorsSentimentClassifier(adapter_path=args.adapter_path)

    if args.text:
        for r in classifier.predict(args.text):
            classifier._print_result(r)
        return

    if args.file:
        if not os.path.exists(args.file):
            print(f"File not found: {args.file}")
            sys.exit(1)
        with open(args.file) as f:
            texts = [l.strip() for l in f if l.strip()]
        for r in classifier.predict(texts):
            classifier._print_result(r)
        return

    classifier.interactive()


import os
if __name__ == "__main__":
    main()
