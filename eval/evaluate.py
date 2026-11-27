"""Evaluate a released checkpoint on a labelled split.

Two protocols are reported:

  masked  (deployment, default)  the acquisition ion mode is known and used to
                                 suppress the four/five adduct classes of the
                                 opposite polarity at inference time.
  free    no polarity information is used anywhere.

The released weights are identical under both protocols; only the inference-time
output constraint differs.

Usage (from the repository root):
    python eval/evaluate.py --ckpt ckpt/MS2ADDUCT_ABL_tx_no_ion_aux_s1_000_Acc_0.854_F1_0.773
    python eval/evaluate.py --ckpt ... --protocol free --out results/eval_free.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import precision_recall_fscore_support, top_k_accuracy_score
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for _p in (HERE, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
from common import collect_logits, dataset_or_die, load_model


def main():
    ap = argparse.ArgumentParser(description="Evaluate MS2ADDUCT on a labelled split")
    ap.add_argument("--ckpt", required=True,
                    help="run directory or best_model.pth")
    ap.add_argument("--data", default=os.path.join(REPO_ROOT, "data/splits/test.pkl"))
    ap.add_argument("--protocol", choices=["masked", "free"], default="masked")
    ap.add_argument("--batch-size", type=int, default=int(config.BATCH_SIZE))
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else config.DEVICE
    model, run_dir = load_model(args.ckpt, device=device)

    ds = dataset_or_die(args.data)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)
    logits, labels, _ = collect_logits(
        model, loader, device, mask_polarity=(args.protocol == "masked"))

    n = config.NUM_ADDUCT_CLASSES
    preds = logits.argmax(dim=-1).numpy()
    acc = float(accuracy_score(labels, preds))
    macro_f1 = float(f1_score(labels, preds, average="macro",
                              labels=list(range(n)), zero_division=0))
    top3 = float(top_k_accuracy_score(labels, logits.numpy(), k=3,
                                      labels=range(n)))
    precision, recall, per_f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(n)), zero_division=0)
    cm = confusion_matrix(labels, preds, labels=list(range(n)))

    names = [config.IDX_TO_ADDUCT[i] for i in range(n)]
    print(f"checkpoint : {os.path.relpath(run_dir, REPO_ROOT)}")
    print(f"data       : {os.path.relpath(args.data, REPO_ROOT)}  (n={len(labels)})")
    print(f"protocol   : {args.protocol}")
    print()
    print("Adduct          n      P       R      F1")
    for i, name in enumerate(names):
        print(f"{name:12s} {support[i]:6d}  {precision[i]:.4f}  {recall[i]:.4f}  {per_f1[i]:.4f}")
    print()
    print(f"Accuracy   : {acc:.4f}")
    print(f"Macro F1   : {macro_f1:.4f}")
    print(f"Top-3 Acc  : {top3:.4f}")

    summary = {
        "checkpoint": os.path.relpath(run_dir, REPO_ROOT),
        "data": os.path.relpath(args.data, REPO_ROOT),
        "protocol": args.protocol,
        "n": int(len(labels)),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "top3_accuracy": top3,
        "per_class": {
            name: {
                "n": int(support[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(per_f1[i]),
            }
            for i, name in enumerate(names)
        },
        "confusion_matrix": cm.tolist(),
        "class_order": names,
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
