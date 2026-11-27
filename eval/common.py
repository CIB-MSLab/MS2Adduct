"""Shared helpers for evaluation and prediction.

The release keeps the model code at the repository root so that ``config.py``
(relative data paths, global singleton) behaves exactly as during training.
"""
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import config
from dataloader import AdductDataset
from model_dreaMS_harmonictsfm import infer_delta_encoder
from model_dreaMS_harmonictsfm_2 import DreaMSAdductHarmonicTsfmClassifierV2

# Final-model defaults.  Training this architecture with these switches is what
# produced the released weights (Fourier delta + tri_xattn + CE token, no ion
# auxiliary head).  They also match the shipped config.py.
EXPECTED = {
    "DELTA_ENCODER": "fourier",
    "FUSION": "tri_xattn",
    "CE_FUSION": "cat",
    "CE_TOKEN_ENABLED": True,
    "ION_AUX_ENABLED": False,
    "PAIRDIFF_ENABLED": False,
    "HIERARCHICAL_HEADS": True,
    "DREAMS2_N_HEADS": 8,
    "DREAMS2_GRAPHORMER": False,
    "MAX_PEAKS": 100,
}


def resolve_ckpt(path):
    """Accept either a run directory or a ``best_model.pth`` path."""
    path = os.path.abspath(path)
    if path.endswith(".pth"):
        return os.path.dirname(path), path
    return path, os.path.join(path, "best_model.pth")


def check_config(strict=True):
    bad = {k: (getattr(config, k, "<missing>"), v)
           for k, v in EXPECTED.items()
           if getattr(config, k, None) != v}
    if bad and strict:
        raise SystemExit(
            "config.py does not match the released architecture (actual, expected):\n"
            + "\n".join(f"  config.{k} = {a!r}, expected {e!r}"
                        for k, (a, e) in bad.items()))
    return bad


def load_model(ckpt, device=None, strict_config=True):
    """Load a released checkpoint.

    The delta encoder architecture is inferred from the state dict (older
    checkpoints used hand-crafted harmonic bases); everything else comes from
    ``config.py``.  Returns ``(model, run_dir)``.
    """
    run_dir, ckpt_path = resolve_ckpt(ckpt)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"{ckpt_path} not found.\n"
            "Download the released weights into ckpt/<run>/best_model.pth "
            "(see ckpt/README.md and scripts/download_weights.sh).")
    check_config(strict=strict_config)

    device = device or config.DEVICE
    state = torch.load(ckpt_path, map_location=device)
    config.DELTA_ENCODER = infer_delta_encoder(state)

    model = DreaMSAdductHarmonicTsfmClassifierV2(
        ckpt_path=os.path.join(config.DREAMS_CKPT_DIR, "ssl_model.ckpt"),
        freeze_backbone=False,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, run_dir


@torch.no_grad()
def collect_logits(model, loader, device, mask_polarity=True):
    """Run a labelled ``AdductDataset`` loader; returns logits, labels, polarity."""
    model.mask_output_by_polarity = bool(mask_polarity)
    logits_all, labels_all, pols_all = [], [], []
    for batch in loader:
        mz = batch["mz"].to(device)
        intensity = batch["intensity"].to(device)
        ce = batch["ce"].to(device)
        precursor_mz = batch["precursor_mz"].squeeze(-1).to(device)
        polarity = batch["polarity"].to(device)
        logits = model(mz, intensity, ce=ce, precursor_mz=precursor_mz,
                       polarity=polarity)
        logits_all.append(logits.float().cpu())
        labels_all.extend(batch["label"].tolist())
        pols_all.extend(batch["polarity"].view(-1).tolist())
    return torch.cat(logits_all), np.asarray(labels_all), np.asarray(pols_all)


def load_records(pkl_path):
    import pickle
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def polarity_from_record(record):
    """Return 0 (positive), 1 (negative), or None when the record lacks it."""
    from dataloader import ion_mode_to_idx
    ion = record.get("ion_mode", record.get("polarity"))
    if ion is None or (isinstance(ion, str) and not ion.strip()):
        return None
    return ion_mode_to_idx(ion)


def preprocess_spectrum(record):
    """Same peak preprocessing as ``dataloader.AdductDataset`` (no augmentation)."""
    mz = np.asarray(record.get("mz", [100.0]), dtype=np.float32)
    intensity = np.asarray(record.get("intensity", [1e-5]), dtype=np.float32)
    if mz.size == 0:
        mz = np.array([100.0], dtype=np.float32)
        intensity = np.array([1e-5], dtype=np.float32)

    order = np.argsort(intensity)[::-1][:config.MAX_PEAKS]
    mz, intensity = mz[order], intensity[order]
    if intensity.max() > 0:
        intensity = intensity / intensity.max()
    pad = config.MAX_PEAKS - len(mz)
    if pad > 0:
        mz = np.pad(mz, (0, pad), constant_values=0.0)
        intensity = np.pad(intensity, (0, pad), constant_values=0.0)

    ce = record.get("normalized_collision_energy")
    ce = 0.0 if ce is None else float(ce)
    return mz, intensity, ce


def dataset_or_die(pkl_path):
    ds = AdductDataset(pkl_path)
    if len(ds) == 0:
        raise SystemExit(f"{pkl_path}: no records with a known adduct label. "
                         "For unlabelled spectra use eval/predict.py.")
    return ds
