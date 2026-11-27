"""Predict adduct classes for spectra in a pickle file.

Input records follow the same dict format used for training/evaluation:

    {
      "mz": [...], "intensity": [...],
      "precursor_mz": 342.15,
      "normalized_collision_energy": 30.0,      # optional, default 0
      "ion_mode": "ESI+" | "ESI-" | ...,        # optional but recommended
      "adduct": "[M+H]+",                       # optional, only used for scoring
      "smiles": "...",                          # optional, ignored
    }

Any extra keys are copied to the output as-is only when named in --keep.

The deployment protocol uses the acquisition ion mode (when present) to
suppress adduct classes of the opposite polarity.  Pass --no-mask to disable.

Usage:
    python eval/predict.py --ckpt ckpt/MS2ADDUCT_ABL_tx_no_ion_aux_s1_000_Acc_0.854_F1_0.773 \
        --data my_spectra.pkl --out predictions.csv

    # 3-seed probability ensemble
    python eval/predict.py --ckpt ckpt/MS2ADDUCT_ABL_tx_no_ion_aux_s*/ \
        --data my_spectra.pkl --out predictions.csv
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for _p in (HERE, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
from common import (load_model, load_records, polarity_from_record,
                    preprocess_spectrum)

DEFAULT_KEEP = ["id", "name", "smiles", "precursor_mz", "adduct"]


def main():
    ap = argparse.ArgumentParser(description="Predict adducts from MS2 spectra")
    ap.add_argument("--ckpt", nargs="+", required=True,
                    help="one or more run directories / best_model.pth files")
    ap.add_argument("--data", required=True,
                    help="pickle file with a list of spectrum dicts")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--polarity", choices=["auto", "positive", "negative"],
                    default="auto",
                    help="auto = read ion_mode/polarity from each record; "
                         "explicit value overrides missing metadata")
    ap.add_argument("--no-mask", action="store_true",
                    help="disable inference-time polarity masking")
    ap.add_argument("--no-ensemble-vote", action="store_true",
                    help="with several checkpoints, report mean probability only "
                         "(default also reports majority vote)")
    ap.add_argument("--keep", nargs="*", default=DEFAULT_KEEP,
                    help="record keys copied to the output CSV")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else config.DEVICE
    records = load_records(args.data)
    if not records:
        raise SystemExit(f"{args.data}: empty record list")

    # Preprocess once; every checkpoint sees exactly the same tensors.
    prepared = [preprocess_spectrum(r) for r in records]
    mz = np.stack([p[0] for p in prepared])
    intensity = np.stack([p[1] for p in prepared])
    ce = np.asarray([p[2] for p in prepared], dtype=np.float32)
    precursor = np.asarray([float(r.get("precursor_mz", 0.0)) for r in records],
                           dtype=np.float32)

    pol = []
    for r in records:
        p = polarity_from_record(r)
        if p is None and args.polarity != "auto":
            p = 0 if args.polarity == "positive" else 1
        pol.append(-1 if p is None else p)   # -1 = unknown
    pol = np.asarray(pol, dtype=np.int64)

    models = []
    for ckpt in args.ckpt:
        model, run_dir = load_model(ckpt, device=device)
        # Masking is applied explicitly below (per record), so that batches
        # mixing spectra with and without known polarity stay correct.
        model.mask_output_by_polarity = False
        models.append((model, run_dir))
        print(f"loaded {os.path.relpath(run_dir, os.getcwd())}")

    ion_of_class = torch.tensor(config.GLOB_TO_ION, dtype=torch.long, device=device)
    probs_per_model = []
    with torch.no_grad():
        for model, _ in models:
            probs_all = []
            for start in range(0, len(records), args.batch_size):
                sl = slice(start, start + args.batch_size)
                b_mz = torch.from_numpy(mz[sl]).to(device)
                b_int = torch.from_numpy(intensity[sl]).to(device)
                b_ce = torch.from_numpy(ce[sl]).to(device).unsqueeze(-1)
                b_prec = torch.from_numpy(precursor[sl]).to(device)
                logits = model(b_mz, b_int, ce=b_ce, precursor_mz=b_prec)
                if not args.no_mask:
                    b_pol = torch.from_numpy(pol[sl]).to(device)
                    known = b_pol >= 0
                    if known.any():
                        keep = ion_of_class[None, :] == b_pol.view(-1, 1)
                        keep = keep | ~known.view(-1, 1)
                        logits = logits.masked_fill(~keep, -1e4)
                probs_all.append(torch.softmax(logits.float(), -1).cpu().numpy())
            probs_per_model.append(np.concatenate(probs_all))

    mean_probs = np.mean(probs_per_model, axis=0)
    pred_mean = mean_probs.argmax(axis=1)

    seed_preds = np.stack([p.argmax(axis=1) for p in probs_per_model], axis=1)
    if seed_preds.shape[1] >= 2 and not args.no_ensemble_vote:
        vote = []
        for row, fallback in zip(seed_preds, pred_mean):
            values, counts = np.unique(row, return_counts=True)
            vote.append(values[counts.argmax()] if counts.max() * 2 > len(row)
                        else fallback)
        pred_final = np.asarray(vote)
    else:
        pred_final = pred_mean

    names = [config.IDX_TO_ADDUCT[i] for i in range(config.NUM_ADDUCT_CLASSES)]
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    fields = ["record_id", "pred_adduct", "confidence"]
    fields += [f"prob_{name}" for name in names]
    if seed_preds.shape[1] > 1:
        fields += [f"pred_s{i + 1}" for i in range(seed_preds.shape[1])]
    if "pred_mean" not in fields and seed_preds.shape[1] > 1:
        fields.append("pred_meanprob")
    present_keep = [k for k in args.keep if any(k in r for r in records)]
    fields += present_keep

    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, r in enumerate(records):
            row = {
                "record_id": i,
                "pred_adduct": names[pred_final[i]],
                "confidence": float(mean_probs[i].max()),
            }
            for j, name in enumerate(names):
                row[f"prob_{name}"] = float(mean_probs[i, j])
            if seed_preds.shape[1] > 1:
                for j in range(seed_preds.shape[1]):
                    row[f"pred_s{j + 1}"] = names[seed_preds[i, j]]
                row["pred_meanprob"] = names[pred_mean[i]]
            for k in present_keep:
                row[k] = r.get(k)
            writer.writerow(row)

    print(f"\n{len(records)} spectra -> {args.out}")
    if seed_preds.shape[1] > 1:
        disagree = int(np.sum(np.any(seed_preds != pred_final[:, None], axis=1)))
        print(f"ensemble: {seed_preds.shape[1]} checkpoints, {disagree} spectra "
              "differ from at least one single-seed prediction")

    labelled = [i for i, r in enumerate(records)
                if r.get("adduct") in config.ADDUCT_TO_IDX]
    if labelled:
        true = np.asarray([config.ADDUCT_TO_IDX[records[i]["adduct"]]
                           for i in labelled])
        acc = float((pred_final[labelled] == true).mean())
        print(f"accuracy on {len(labelled)} labelled spectra: {acc:.4f}")


if __name__ == "__main__":
    main()
