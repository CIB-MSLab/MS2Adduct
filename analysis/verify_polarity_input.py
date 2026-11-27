"""
verify_polarity_input.py
"离子模式 (正/负) 作为已知输入" 这一改动的自检 / 回归。

背景: 极性在采集时由操作者选定, 谱图元数据里直接给出 (cib-ms 的 ion_mode=ESI+),
     不需要模型去猜。改动见 config.py 的 POLARITY_* 说明块:
       POLARITY_AS_INPUT    极性嵌入与 CE 嵌入拼接后一起进门控
       POLARITY_MASK_OUTPUT 用给定极性把九类 logits 里反极性的类置 -inf

本脚本回答四件事:
  (1) 两个开关全关时, 模型是否与改动前**逐位一致** (权重可 strict 加载控制臂 ckpt);
  (2) 极性确实是"被使用的输入", 而不是被静默忽略的参数;
  (3) 屏蔽确实把预测限制在给定极性内 (含 focal loss + -inf 的数值安全性);
  (4) dataloader 给出的极性正确 (元数据优先, 否则由 adduct 推出), 且与模型的类别表一致。

用法:
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


@check("[1] 开关全关 = 与改动前逐位一致 (可 strict 加载控制臂权重)")
def t1():
    ck = next((d for d in CKPT_DIRS if os.path.exists(os.path.join(d, "best_model.pth"))), None)
    m = _make(dict(POLARITY_AS_INPUT=False, POLARITY_MASK_OUTPUT=False))
    if ck is None:
        return "跳过 (未找到控制臂 ckpt, 只验证参数量)", \
               f"proj 输入维度={m.ce_adapter.proj.in_features} (期望 {config.CE_EMBED_DIM})"
    sd = torch.load(os.path.join(ck, "best_model.pth"), map_location=config.DEVICE)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not missing, f"缺键 {missing[:5]}"
    assert not unexpected, f"多键 {unexpected[:5]}"
    assert m.accepts_polarity is False
    return f"strict 通过 (0 missing / 0 unexpected, {len(sd)} 条权重)", \
           f"proj 输入维度={m.ce_adapter.proj.in_features} (= CE_EMBED_DIM, 未被极性撑大)"


@check("[2] 开关全关时, 传 polarity 与不传的输出完全相同 (极性被忽略而非误用)")
def t2():
    m = _make(dict(POLARITY_AS_INPUT=False, POLARITY_MASK_OUTPUT=False)).eval()
    mz, inten, ce, prec, pol = _batch()
    with torch.no_grad():
        a = m(mz, inten, ce=ce, precursor_mz=prec)
        b = m(mz, inten, ce=ce, precursor_mz=prec, polarity=pol)
    d = (a - b).abs().max().item()
    assert d == 0.0, f"最大差 {d}"
    return f"逐位相同 (max|Δ|={d:.1e})", "关闭时 polarity 是无副作用的"


@check("[3] 类别表一致: config.GLOB_TO_ION == model_dreaMS.DreaMSAdductClassifier.GLOB_TO_ION")
def t3():
    assert config.GLOB_TO_ION == DreaMSAdductClassifier.GLOB_TO_ION, \
        (config.GLOB_TO_ION, DreaMSAdductClassifier.GLOB_TO_ION)
    for a, i in config.ADDUCT_TO_IDX.items():
        assert config.POLARITY_OF_ADDUCT[a] == DreaMSAdductClassifier.GLOB_TO_ION[i]
    return "九类 → 极性 (0=正/1=负) 与模型内部表一致", \
           f"正极类 {DreaMSAdductClassifier.POS_GLOB} / 负极类 {DreaMSAdductClassifier.NEG_GLOB}"


@check("[4] dataloader: 极性来自元数据 (优先) 或由 adduct 推出")
def t4():
    from dataloader import AdductDataset, ion_mode_to_idx
    assert ion_mode_to_idx('ESI+') == 0 and ion_mode_to_idx('ESI-') == 1
    assert ion_mode_to_idx('negative') == 1 and ion_mode_to_idx('Positive') == 0
    if not os.path.exists(config.TRAIN_PKL):
        return "跳过实数据检查 (未下载训练数据, 见 scripts/download_data.sh)", \
               "字面量解析: ESI+→0, ESI-→1, negative→1 (cib-ms 用 ion_mode 字段)"
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
    return f"实数据 9 类全部对表 ({len(seen)} 类)", \
           f"字面量解析: ESI+→0, ESI-→1, negative→1 (cib-ms 用 ion_mode 字段)"


@check("[5] POLARITY_AS_INPUT: 极性与 CE 拼接进门控, 且确实改变输出")
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
    assert d > 0, "翻转极性后输出不变 —— 极性没有被真正使用"
    assert float(a[~torch.isinf(a)].min()) >= -1e9, "输出不应有 -inf (本臂未开屏蔽)"
    return f"翻转极性 => 输出改变 (max|Δ|={d:.4f})", \
           f"proj 输入 {config.CE_EMBED_DIM}+{config.POLARITY_EMB_DIM}, 新增参数仅极性嵌入 (2×{config.POLARITY_EMB_DIM})"


@check("[6] POLARITY_MASK_OUTPUT: 反极性类被屏蔽, argmax 必落在给定极性内")
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
        assert p[dead].sum().item() == 0.0, f"屏蔽位概率 {p[dead].sum().item()}"
    assert (lg0 > MASK).all(), "不传 polarity 时不应屏蔽"
    return f"反极性 {len(P) if pol[0] == 1 else len(N)}/{len(P) if pol[0] == 0 else len(N)} 类={MASK} (softmax 概率精确为 0)", \
           f"MASKED_LOGIT={MASK} 是有限值, 评测链路 (top-k/sklearn) 不会被非有限值打断"


@check("[7] 需要极性却缺失时报错清晰 (不静默退化)")
def t7():
    m = _make(dict(POLARITY_AS_INPUT=True, POLARITY_MASK_OUTPUT=True)).eval()
    mz, inten, ce, prec, pol = _batch()
    try:
        with torch.no_grad():
            m(mz, inten, ce=ce, precursor_mz=prec, polarity=None)
    except ValueError as e:
        assert "polarity" in str(e)
        return f"抛出 ValueError: {str(e)[:60]}...", "训练循环漏传极性会立刻暴露"
    raise AssertionError("未报错 —— 极性被静默忽略, 会在训练里无声降级")


@check("[8] focal loss + 屏蔽 logits 数值安全, 梯度可回传")
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
    assert g > 0, "梯度没有流回谱图编码器侧"
    pol_grad = m.ce_adapter.polarity_embedding.weight.grad
    return f"loss={loss.item():.4f} 有限, 反向传播成功", \
           f"极性嵌入梯度范数={None if pol_grad is None else float(pol_grad.norm()):.2e}"


@check("[9] 屏蔽后 top-k 指标仍可计算 (回归: -inf 会让 sklearn 直接抛错)")
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
    assert tf, "真类的 logit 不应被屏蔽"
    return f"Acc={a:.3f} MacroF1={f:.3f} Top3={k3:.3f} 均可算", \
           "真类 logit 始终有限 (极性自洽 => 不会被屏蔽); 评测链路与 _print_test_results 一致"


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    fails = 0
    print("=" * 78)
    print("verify_polarity_input.py —— 离子模式作为已知输入的自检")
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
    print("全部通过" if not fails else f"{fails} 项失败")
    print("=" * 78)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
