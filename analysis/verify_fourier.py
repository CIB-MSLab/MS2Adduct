"""Self-check for the dreaMS-style Fourier neutral-loss encoding.

Usage: python verify_fourier.py
"""
import os, sys, traceback
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from model_dreaMS_harmonictsfm import DreaMSAdductHarmonicTsfmClassifier

_results = []

def check(name):
    def deco(fn):
        _results.append((name, fn)); return fn
    return deco

def _reset(mode="bases", trainable=False):
    config.DELTA_ENCODER = mode
    config.FOURIER_STRATEGY = "lin_float_int"
    config.FOURIER_X_MIN = 1e-4
    config.FOURIER_X_MAX = 2000.0
    config.FOURIER_TRAINABLE = trainable
    config.FOURIER_NUM_FREQS = 5998
    config.PAIRDIFF_ENABLED = False

def _make():
    return DreaMSAdductHarmonicTsfmClassifier(ckpt_path=None, freeze_backbone=True
                                              ).to(config.DEVICE)

@check("[1] fourier by default; no fourier_enc when bases is requested explicitly")
def t_default():
    assert config.DELTA_ENCODER == "fourier"
    _reset("bases")
    m = _make()
    assert m.harmonic_enc.fourier_enc is None
    assert m.harmonic_enc.in_proj[1].in_features == 61 * 3 * 2

@check("[2] fourier: 13994-dim input, frequencies held fixed")
def t_fourier():
    _reset("fourier")
    m = _make()
    fe = m.harmonic_enc.fourier_enc
    assert fe is not None
    assert fe.num_features() == 13994, fe.num_features()
    assert m.harmonic_enc.in_proj[1].in_features == 13994
    assert fe.b.requires_grad is False

@check("[3] fourier forward is finite + padding invariance")
def t_forward():
    _reset("fourier")
    m = _make(); m.eval()
    B, N = 3, 20
    torch.manual_seed(0)
    mz = torch.zeros(B, N, device=config.DEVICE)
    mz[:, :12] = torch.sort(torch.rand(B, 12, device=config.DEVICE) * 800 + 50, dim=-1)[0]
    prec = torch.tensor([300.0, 500.0, 420.0], device=config.DEVICE)
    delta = prec.unsqueeze(-1) - mz
    mask = mz > 0
    with torch.no_grad():
        a = m.harmonic_enc(delta, mask)
        assert torch.isfinite(a).all()
        mz2 = mz.clone(); mz2[:, 12:] = torch.rand_like(mz2[:, 12:]) * 100
        b = m.harmonic_enc(prec.unsqueeze(-1) - mz2, mask)
    assert (a - b).abs().max().item() <= 1e-6, (a - b).abs().max().item()

@check("[4] toggling the switch changes the output (not silently ignored)")
def t_switch():
    _reset("bases"); mb = _make().eval()
    _reset("fourier"); mf = _make().eval()
    sd_b, sd_f = mb.state_dict(), mf.state_dict()
    shared = {k: v for k, v in sd_b.items()
              if k in sd_f and sd_f[k].shape == v.shape}
    mf.load_state_dict(shared, strict=False)
    B, N = 2, 10
    mz = torch.zeros(B, N, device=config.DEVICE); mz[:, :5] = 100.0
    prec = torch.tensor([300.0, 400.0], device=config.DEVICE)
    delta = prec.unsqueeze(-1) - mz
    with torch.no_grad():
        a = mb.harmonic_enc(delta, mz > 0)
        b = mf.harmonic_enc(delta, mz > 0)
    assert (a - b).abs().max().item() > 1e-6

@check("[5] frequencies receive gradients when trainable=True")
def t_trainable():
    _reset("fourier", trainable=True)
    m = _make()
    assert m.harmonic_enc.fourier_enc.b.requires_grad is True
    B, N = 2, 10
    mz = torch.zeros(B, N, device=config.DEVICE); mz[:, :5] = 100.0
    prec = torch.tensor([300.0, 400.0], device=config.DEVICE)
    out = m.harmonic_enc(prec.unsqueeze(-1) - mz, mz > 0)
    out.sum().backward()
    assert m.harmonic_enc.fourier_enc.b.grad is not None
    assert m.harmonic_enc.fourier_enc.b.grad.abs().sum().item() > 0

def main():
    fails = 0
    for name, fn in _results:
        try:
            fn(); print(f"  OK   {name}")
        except Exception as e:
            fails += 1; print(f"  FAIL {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print("all checks passed" if not fails else f"{fails} check(s) failed")
    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
