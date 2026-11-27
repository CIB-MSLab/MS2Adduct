"""峰间差分中性丢失网络自检。用法: python verify_pairdiff.py"""
import os, sys, traceback
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from model_dreaMS_harmonictsfm import (
    DreaMSAdductHarmonicTsfmClassifier,
    raw_harmonic_components,
)

_results = []

def check(name):
    def deco(fn):
        _results.append((name, fn)); return fn
    return deco

@check("[1] 开关默认关闭")
def t_defaults():
    assert config.PAIRDIFF_ENABLED is False
    assert config.PAIRDIFF_BASE_SET == "both"
    assert config.PAIRDIFF_HARMONICS == 1
    assert config.PAIRDIFF_MIN_DELTA == 1.0
    assert config.PAIRDIFF_SHAM is False

@check("[2] 谐波原语显式参数 = 默认全局参数 (逐位)")
def t_helper():
    old_b, old_h = config.HARMONIC_BASE_FREQUENCIES, config.N_HARMONICS_PER_BASE
    try:
        config.HARMONIC_BASE_FREQUENCIES = [17.0, 43.0]
        config.N_HARMONICS_PER_BASE = 2
        x = torch.tensor([[1.0, 2.5, 9.0]])
        a = raw_harmonic_components(x)
        b = raw_harmonic_components(x, bases=[17.0, 43.0], n_harm=2)
        assert torch.equal(a, b), "显式参数与默认参数结果不一致"
        assert a.shape == (1, 3, 2 * 2 * 2), a.shape
    finally:
        config.HARMONIC_BASE_FREQUENCIES = old_b
        config.N_HARMONICS_PER_BASE = old_h

@check("[3] sham 基频: 等长/范围匹配/最小间隔/确定性")
def t_sham():
    from model_dreaMS_harmonictsfm import make_sham_bases
    true = config.HARMONIC_BASE_SETS["both"]
    s1 = make_sham_bases(true, seed=123)
    s2 = make_sham_bases(true, seed=123)
    assert s1 == s2, "同 seed 结果不稳定"
    assert len(s1) == len(true)
    assert s1 == sorted(s1)
    assert min(s1) >= 1.0 and max(s1) <= 200.0
    for v in s1:
        assert min(abs(v - t) for t in true) >= 0.05, f"sham {v} 离真基频太近"
    for i in range(len(s1)):
        for j in range(i + 1, len(s1)):
            assert s1[j] - s1[i] >= 0.05, (s1[i], s1[j])
    assert s1 != list(true)

def _reset_pairdiff(enabled=False, sham=False):
    config.DELTA_ENCODER = "bases"   # 默认已切 fourier; 本自检固定手工基频表架构
    config.PAIRDIFF_ENABLED = enabled
    config.PAIRDIFF_BASE_SET = "both"
    config.PAIRDIFF_HARMONICS = 1
    config.PAIRDIFF_MIN_DELTA = 1.0
    config.PAIRDIFF_SHAM = sham

def _make_model():
    return DreaMSAdductHarmonicTsfmClassifier(ckpt_path=None, freeze_backbone=True
                                              ).to(config.DEVICE)

def _batch(b=3, n=20):
    torch.manual_seed(0)
    mz = torch.zeros(b, n, device=config.DEVICE)
    mz[:, :12] = torch.sort(torch.rand(b, 12, device=config.DEVICE) * 800 + 50, dim=-1)[0]
    inten = torch.zeros(b, n, device=config.DEVICE)
    inten[:, :12] = torch.rand(b, 12, device=config.DEVICE)
    prec = torch.tensor([300.0, 500.0, 420.0], device=config.DEVICE)[:b]
    return mz, inten, prec

@check("[4] 关闭时不构造 pair_proj; 现有 ckpt 可 strict 加载")
def t_off():
    import glob
    _reset_pairdiff(False)
    m = _make_model()
    assert m.harmonic_enc.pair_proj is None
    cks = sorted(glob.glob(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ckpt", "MS2ADDUCT_DreaMSHT2_s*")))
    ck = next((d for d in cks if os.path.exists(os.path.join(d, "best_model.pth"))), None)
    if ck:
        sd = torch.load(os.path.join(ck, "best_model.pth"), map_location=config.DEVICE)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, (missing[:5], unexpected[:5])

@check("[5] 开启但零初始化 = 与关闭模型等价 (max|Δ| <= 1e-5)")
def t_zero_init():
    _reset_pairdiff(False); off = _make_model()
    _reset_pairdiff(True);  on = _make_model()
    on.load_state_dict(off.state_dict(), strict=False)
    mz, inten, prec = _batch()
    off.eval(); on.eval()
    with torch.no_grad():
        a = off(mz, inten, precursor_mz=prec)
        b = on(mz, inten, precursor_mz=prec)
    assert torch.isfinite(b).all()
    assert (a - b).abs().max().item() <= 1e-5, (a - b).abs().max().item()

@check("[6] padding 不变性 (只测 δ 编码器, 排除谱图主干的影响)")
def t_padding():
    _reset_pairdiff(True); m = _make_model()
    torch.nn.init.normal_(m.harmonic_enc.pair_proj[-1].weight, std=0.1)
    mz, inten, prec = _batch()
    m.eval()
    delta = prec.unsqueeze(-1) - mz
    mask = mz > 0
    mz2 = mz.clone(); mz2[:, 12:] = torch.rand_like(mz2[:, 12:]) * 100
    delta2 = prec.unsqueeze(-1) - mz2
    with torch.no_grad():
        a = m.harmonic_enc(delta, mask)
        b = m.harmonic_enc(delta2, mask)
    assert (a - b).abs().max().item() <= 1e-6, (a - b).abs().max().item()

@check("[7] 梯度能流到 pair_proj")
def t_grad():
    _reset_pairdiff(True); m = _make_model()
    mz, inten, prec = _batch()
    out = m(mz, inten, precursor_mz=prec)
    out.sum().backward()
    g = m.harmonic_enc.pair_proj[-1].weight.grad
    assert g is not None and g.abs().sum().item() > 0

@check("[8] 边偏置有限 (fp32 数值安全)")
def t_bias_finite():
    _reset_pairdiff(True); m = _make_model()
    torch.nn.init.normal_(m.harmonic_enc.pair_proj[-1].weight, std=0.1)
    mz, inten, prec = _batch()
    delta = prec.unsqueeze(-1) - mz
    mask = mz > 0
    with torch.no_grad():
        raw = delta.float()
        pd = (raw.unsqueeze(-2) - raw.unsqueeze(-1)).abs()
        feats = raw_harmonic_components(pd, bases=m.harmonic_enc.pair_bases,
                                        n_harm=m.harmonic_enc.pair_harm)
        bias = m.harmonic_enc.pair_proj(feats)
    assert torch.isfinite(bias).all()

@check("[9] 极端输入不产生 NaN")
def t_extreme():
    _reset_pairdiff(True); m = _make_model(); m.eval()
    # 全 padding
    mz = torch.zeros(2, 20, device=config.DEVICE)
    inten = torch.zeros(2, 20, device=config.DEVICE)
    prec = torch.tensor([300.0, 500.0], device=config.DEVICE)
    with torch.no_grad():
        assert torch.isfinite(m(mz, inten, precursor_mz=prec)).all()
    # 单峰 + 重复峰
    mz = torch.zeros(2, 20, device=config.DEVICE)
    mz[0, 0] = 100.0; mz[1, :5] = 100.0
    inten = torch.zeros(2, 20, device=config.DEVICE); inten[:, :5] = 1.0
    with torch.no_grad():
        assert torch.isfinite(m(mz, inten, precursor_mz=prec)).all()

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
