"""dreaMS 风格傅里叶中性丢失编码自检。用法: python verify_fourier.py"""
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

@check("[1] 默认 fourier; 显式 bases 时不构造 fourier_enc")
def t_default():
    assert config.DELTA_ENCODER == "fourier"
    _reset("bases")
    m = _make()
    assert m.harmonic_enc.fourier_enc is None
    assert m.harmonic_enc.in_proj[1].in_features == 61 * 3 * 2

# NOTE(2026-10-18): 原 brief 写的是 11994 维 (5997 频率)。该数字对应 x_max=1000
# (dreaMS 默认 max_mz)。lin_float_int 的实际公式为
#   range(2, ceil(1/x_min), 2) + range(2, ceil(x_max), 1)
# x_min=1e-4, x_max=2000 时为 4999 + 1998 = 6997 个频率 → sin+cos 后 13994 维。
# x_max=2000 是为覆盖现有 δ clamp(0, 2000), 故以实际维数为准。
@check("[2] fourier: 13994 维输入, 频率固定")
def t_fourier():
    _reset("fourier")
    m = _make()
    fe = m.harmonic_enc.fourier_enc
    assert fe is not None
    assert fe.num_features() == 13994, fe.num_features()
    assert m.harmonic_enc.in_proj[1].in_features == 13994
    assert fe.b.requires_grad is False

@check("[3] fourier 前向有限 + padding 不变性")
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

@check("[4] 切换开关改变输出 (不是被静默忽略)")
def t_switch():
    _reset("bases"); mb = _make().eval()
    _reset("fourier"); mf = _make().eval()
    # strict=False 只忽略缺键/多键, 不忽略 shape 不匹配; in_proj 维度不同, 先过滤。
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

@check("[5] trainable=True 时频率有梯度")
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
    print("全部通过" if not fails else f"{fails} 项失败")
    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
