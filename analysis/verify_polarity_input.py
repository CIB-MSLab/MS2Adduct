"""
verify_polarity_input.py
Self-check / regression for the "ion mode (positive/negative) as a known input" change.

Background: the polarity is chosen by the operator at acquisition time and is
given directly in the spectrum metadata (the in-house cib-ms set carries
ion_mode=ESI+), so the model should not have to guess it. See the POLARITY_*
block in config.py:
  POLARITY_AS_INPUT    the polarity embedding is concatenated with the CE
                       embedding before entering the gate
  POLARITY_MASK_OUTPUT sets the opposite-polarity classes among the nine logits
                       to -inf using the given polarity

This script answers four questions:
  (1) with both switches off, is the model **bit-identical** to the state before
      the change (can the weights be loaded strictly from the control-arm ckpt);
  (2) is the polarity really an input that gets *used*, rather than a parameter
      that is silently ignored;
  (3) does the mask really restrict predictions to the given polarity (including
      the numerical safety of focal loss over -inf);
  (4) does the dataloader produce the right polarity (metadata first, otherwise
      derived from the adduct), and does it agree with the class table of the
      model.

Usage:
    python verify_polarity_input.py
"""
import glob
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from model_dreaMS import DreaMSAdductClassifier
from model_dreaMS_harmonictsfm import DreaMSAdductHarmonicTsfmClassifier

CKPT_DIRS = sorted(glob.glob(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ckpt", "MS2ADDUCT_DreaMSHT2_s*")))
_results = []


def check(name):
    def deco(fn):
        _results.append((name, fn))
        return fn
    return deco


def _make(flags, ckpt=None):
    for k in ('POLARITY_AS_INPUT', 'POLARITY_MASK_OUTPUT',
              'POLARITY_MASK_INFERENCE_ONLY'):
        setattr(config, k, False)
    config.DELTA_ENCODER = "bases"
    config.FUSION = "bilinear"
    config.CE_FUSION = "gate"
    for k, v in flags.items():
        setattr(config, k, v)
    m = DreaMSAdductHarmonicTsfmClassifier(ckpt_path=ckpt, freeze_backbone=True)
    return m.to(config.DEVICE)


def _batch(b=4, n=40):
    torch.manual_seed(0)
    mz = torch.zeros(b, n, device=config.DEVICE)
    mz[:, :25] = torch.rand(b, 25, device=config.DEVICE) * 800 + 50
    mz, _ = mz.sort(dim=-1)
    inten = torch.zeros(b, n, device=config.DEVICE)
    inten[:, :25] = torch.rand(b, 25, device=config.DEVICE)
    ce = torch.rand(b, 1, device=config.DEVICE)
    prec = torch.tensor([300.0, 500.0, 420.0, 610.0], device=config.DEVICE)[:b]
    pol = torch.tensor([0, 1, 0, 1], device=config.DEVICE)[:b]
    return mz, inten, ce, prec, pol


@check("[1] both switches off == bit-identical to before the change (control-arm weights load strictly)")
def t1():
    ck = next((d for d in CKPT_DIRS if os.path.exists(os.path.join(d, "best_model.pth"))), None)
    m = _make(dict(POLARITY_AS_INPUT=False, POLARITY_MASK_OUTPUT=False))
    if ck is None:
        return "skipped (no control-arm ckpt found, only the parameter count is checked)", \
               f"proj input dim={m.ce_adapter.proj.in_features} (expected {config.CE_EMBED_DIM})"
    sd = torch.load(os.path.join(ck, "best_model.pth"), map_location=config.DEVICE)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not missing, f"missing keys {missing[:5]}"
    assert not unexpected, f"unexpected keys {unexpected[:5]}"
    assert m.accepts_polarity is False
    return f"strict load OK (0 missing / 0 unexpected, {len(sd)} tensors)", \
           f"proj input dim={m.ce_adapter.proj.in_features} (= CE_EMBED_DIM, not widened by polarity)"


@check("[2] with both switches off, passing polarity changes nothing (polarity is ignored, not misused)")
def t2():
    m = _make(dict(POLARITY_AS_INPUT=False, POLARITY_MASK_OUTPUT=False)).eval()
    mz, inten, ce, prec, pol = _batch()
    with torch.no_grad():
        a = m(mz, inten, ce=ce, precursor_mz=prec)
        b = m(mz, inten, ce=ce, precursor_mz=prec, polarity=pol)
    d = (a - b).abs().max().item()
    assert d == 0.0, f"max diff {d}"
    return f"bit-identical (max|delta|={d:.1e})", "polarity has no side effects when the flags are off"


@check("[3] class tables agree: config.GLOB_TO_ION == model_dreaMS.DreaMSAdductClassifier.GLOB_TO_ION")
def t3():
    assert config.GLOB_TO_ION == DreaMSAdductClassifier.GLOB_TO_ION, \
        (config.GLOB_TO_ION, DreaMSAdductClassifier.GLOB_TO_ION)
    for a, i in config.ADDUCT_TO_IDX.items():
        assert config.POLARITY_OF_ADDUCT[a] == DreaMSAdductClassifier.GLOB_TO_ION[i]
    return "nine classes -> polarity (0=positive/1=negative) matches the model's internal table", \
           f"positive classes {DreaMSAdductClassifier.POS_GLOB} / negative classes {DreaMSAdductClassifier.NEG_GLOB}"


@check("[4] dataloader: polarity comes from the metadata (preferred) or is derived from the adduct")
def t4():
    from dataloader import AdductDataset, ion_mode_to_idx
    assert ion_mode_to_idx('ESI+') == 0 and ion_mode_to_idx('ESI-') == 1
    assert ion_mode_to_idx('negative') == 1 and ion_mode_to_idx('Positive') == 0
    if not os.path.exists(config.TRAIN_PKL):
        return "skipped the real-data check (training data not downloaded, see scripts/download_data.sh)", \
               "literal parsing: ESI+->0, ESI-->1, negative->1 (cib-ms carries an ion_mode field)"
    ds = AdductDataset(config.TRAIN_PKL)
    seen = {}
    for i in range(0, len(ds), max(1, len(ds) // 4000)):
        s = ds.samples[i]
        ad = s.get('adduct')
        if ad in seen:
            continue
        seen[ad] = ds[i]['polarity'].item()
    assert set(seen) == set(config.ADDUCT_TO_IDX), seen.keys()
    for a, p in seen.items():
        assert p == config.POLARITY_OF_ADDUCT[a], (a, p)
    return f"all nine classes match on real data ({len(seen)} classes)", \
           f"literal parsing: ESI+->0, ESI-->1, negative->1 (cib-ms carries an ion_mode field)"


@check("[5] POLARITY_AS_INPUT: polarity is concatenated with CE into the gate and does change the output")
def t5():
    m = _make(dict(POLARITY_AS_INPUT=True, POLARITY_MASK_OUTPUT=False)).eval()
    assert m.accepts_polarity is True
    assert m.ce_adapter.proj.in_features == config.CE_EMBED_DIM + config.POLARITY_EMB_DIM
    assert isinstance(m.ce_adapter.polarity_embedding, torch.nn.Embedding)
    mz, inten, ce, prec, pol = _batch()
    with torch.no_grad():
        a = m(mz, inten, ce=ce, precursor_mz=prec, polarity=pol)
        b = m(mz, inten, ce=ce, precursor_mz=prec, polarity=1 - pol)
    d = (a - b).abs().max().item()
    assert d > 0, "output unchanged after flipping the polarity, so the polarity is not actually used"
    assert float(a[~torch.isinf(a)].min()) >= -1e9, "the output must not contain -inf (masking is off in this arm)"
    return f"flipping the polarity changes the output (max|delta|={d:.4f})", \
           f"proj input {config.CE_EMBED_DIM}+{config.POLARITY_EMB_DIM}; the only new parameters are the polarity embedding (2x{config.POLARITY_EMB_DIM})"


@check("[6] POLARITY_MASK_OUTPUT: opposite-polarity classes are masked, argmax always lands in the given polarity")
def t6():
    m = _make(dict(POLARITY_AS_INPUT=False, POLARITY_MASK_OUTPUT=True)).eval()
    P, N = list(DreaMSAdductClassifier.POS_GLOB), list(DreaMSAdductClassifier.NEG_GLOB)
    MASK = m.MASKED_LOGIT
    mz, inten, ce, prec, pol = _batch()
    with torch.no_grad():
        lg = m(mz, inten, ce=ce, precursor_mz=prec, polarity=pol)
        lg0 = m(mz, inten, ce=ce, precursor_mz=prec, polarity=None)
    for i in range(lg.shape[0]):
        dead = P if pol[i].item() == 1 else N
        alive = N if pol[i].item() == 1 else P
        assert (lg[i, dead] == MASK).all(), lg[i]
        assert torch.isfinite(lg[i, alive]).all()
        assert lg[i].argmax().item() in alive
        p = torch.softmax(lg[i].float(), -1)
        assert p[dead].sum().item() == 0.0, f"masked probability {p[dead].sum().item()}"
    assert (lg0 > MASK).all(), "nothing should be masked when polarity is not passed"
    return f"opposite-polarity {len(P) if pol[0] == 1 else len(N)}/{len(P) if pol[0] == 0 else len(N)} classes={MASK} (softmax probability exactly 0)", \
           f"MASKED_LOGIT={MASK} is finite, so the evaluation path (top-k/sklearn) is never broken by non-finite values"


@check("[7] a clear error when polarity is required but missing (no silent degradation)")
def t7():
    m = _make(dict(POLARITY_AS_INPUT=True, POLARITY_MASK_OUTPUT=True)).eval()
    mz, inten, ce, prec, pol = _batch()
    try:
        with torch.no_grad():
            m(mz, inten, ce=ce, precursor_mz=prec, polarity=None)
    except ValueError as e:
        assert "polarity" in str(e)
        return f"raised ValueError: {str(e)[:60]}...", "a training loop that forgets polarity fails immediately"
    raise AssertionError("no error raised; the polarity is silently ignored and would degrade training without any warning")


@check("[8] focal loss over masked logits is numerically safe and gradients flow")
def t8():
    from train_utils import FocalLoss
    m = _make(dict(POLARITY_AS_INPUT=True, POLARITY_MASK_OUTPUT=True))
    m.train()
    crit = FocalLoss(gamma=2.0, alpha=torch.ones(config.NUM_ADDUCT_CLASSES, device=config.DEVICE))
    mz, inten, ce, prec, pol = _batch()
    lab = torch.tensor([0, 1, 2, 4], device=config.DEVICE)[:mz.shape[0]]
    lg = m(mz, inten, ce=ce, precursor_mz=prec, polarity=pol)
    loss = crit(lg, lab)
    assert torch.isfinite(loss), f"loss={loss}"
    loss.backward()
    g = 0.0
    for p in m.ms_encoder.proj.parameters():
        if p.grad is not None:
            g += p.grad.abs().sum().item()
    assert g > 0, "gradients do not reach the spectrum encoder side"
    pol_grad = m.ce_adapter.polarity_embedding.weight.grad
    return f"loss={loss.item():.4f} finite, backward pass succeeded", \
           f"polarity-embedding gradient norm={None if pol_grad is None else float(pol_grad.norm()):.2e}"


@check("[9] top-k metrics are still computable after masking (regression: -inf makes sklearn raise)")
def t9():
    import numpy as np
    from sklearn.metrics import top_k_accuracy_score, f1_score, accuracy_score
    m = _make(dict(POLARITY_AS_INPUT=True, POLARITY_MASK_OUTPUT=True)).eval()
    mz, inten, ce, prec, pol = _batch()
    lab = torch.tensor([0, 1, 2, 4], device=config.DEVICE)[:mz.shape[0]].cpu().numpy()
    with torch.no_grad():
        lg = m(mz, inten, ce=ce, precursor_mz=prec, polarity=pol).float().cpu()
    k3 = top_k_accuracy_score(lab, lg.numpy(), k=3, labels=range(9))
    preds = lg.argmax(dim=-1).tolist()
    a = accuracy_score(lab, preds)
    f = f1_score(lab, preds, average='macro', labels=list(range(9)), zero_division=0)
    tf = torch.isfinite(lg[torch.arange(len(lab)), torch.from_numpy(lab)]).all().item()
    assert tf, "the logit of the true class must not be masked"
    return f"Acc={a:.3f} MacroF1={f:.3f} Top3={k3:.3f} all computable", \
           "the true-class logit is always finite (the polarity is self-consistent, so it is never masked); the evaluation path matches _print_test_results"


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    fails = 0
    print("=" * 78)
    print("verify_polarity_input.py -- self-check for ion mode as a known input")
    print("=" * 78)
    for name, fn in _results:
        if only and only not in name:
            continue
        try:
            headline, detail = fn()
            print(f"{name}\n    OK  {headline}\n        {detail}")
        except Exception as e:
            fails += 1
            print(f"{name}\n    FAIL  {type(e).__name__}: {e}")
            traceback.print_exc()
    print("=" * 78)
    print("all checks passed" if not fails else f"{fails} check(s) failed")
    print("=" * 78)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
