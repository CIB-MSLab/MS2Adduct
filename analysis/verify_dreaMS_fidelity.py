"""
verify_dreaMS_fidelity.py
DreaMSHT2 (忠实复现 dreaMS 原型的 dreaMS 分支) 的回归自检脚本。

先于实现编写 —— 实现完成前它必须失败。检查项:

  1. 开关快照: 打印 DREAMS2_* 的实际取值 (实验日志审计用)
  2. 预训练配置对账: 从 ssl_model.ckpt 里直接读出预训练时的 n_heads / fourier_strategy /
     max_peaks_n, 与本次构造的 backbone 逐项比对。**这是本次修复的核心回归测试** ——
     头数错配之所以潜伏至今, 就是因为 ckpt 的 kwargs 读不出来, 没人对账。
  3. 权重加载: 65/65 backbone 张量, 无 missing / unexpected
  4. 改动生效性: V2 与 V1 的冻结前向必须显著不同 (否则说明开关静默失效)
  5. gd 快捷式 vs 原版字面 materialize 的数值差 (代数等价, 预期 fp32 下 ~1e-6)
  6. AMP autocast 下前向无 NaN (原版 -1e9 在 fp16 下会 NaN, 这是保留 -1e4 的原因)

运行: python verify_dreaMS_fidelity.py
"""
import os, sys, types, importlib.abc, importlib.machinery
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def read_ckpt_kwargs(ckpt_path):
    """从 Lightning ckpt 里读出预训练时的 args。

    ckpt 里 pickle 了 msml 的类, 必须先塞桩模块才能 torch.load。
    用 map_location='meta' 避免把 1.2GB 权重真正读进内存。
    """
    class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] == 'msml':
                return importlib.machinery.ModuleSpec(fullname, self)

        def create_module(self, spec):
            return types.ModuleType(spec.name)

        def exec_module(self, module):
            module.__getattr__ = lambda name: type(
                name, (), {'__init__': lambda s, *a, **k: None})

    sys.meta_path.insert(0, _StubFinder())
    try:
        ck = torch.load(ckpt_path, map_location='meta', weights_only=False)
    finally:
        sys.meta_path.pop(0)
    return vars(ck['hyper_parameters']['args']), ck['state_dict']


def main():
    print("=" * 78)
    print("DreaMSHT2 忠实性自检")
    print("=" * 78)

    print("\n[1] DREAMS2_* 开关快照")
    snap = {k: getattr(config, k, '<未定义>') for k in
            ['DREAMS2_N_HEADS', 'DREAMS2_GRAPHORMER', 'DREAMS2_FOURIER_STRAT',
             'DREAMS2_MAX_PEAKS', 'DREAMS2_DROPOUT']}
    for k, v in snap.items():
        print(f"      {k} = {v!r}")
    check("DREAMS2_* 开关已定义", all(v != '<未定义>' for v in snap.values()))

    ckpt_path = os.path.join(config.DREAMS_CKPT_DIR, "ssl_model.ckpt")
    if not os.path.exists(ckpt_path):
        print(f"\n!! 找不到 {ckpt_path}, 无法继续")
        return 1

    print("\n[2] 与 ssl_model.ckpt 记录的预训练配置对账")
    args, ck_sd = read_ckpt_kwargs(ckpt_path)
    ck_heads = args['n_heads']
    ck_strategy = args['fourier_strategy']
    ck_peaks = args['max_peaks_n']
    print(f"      ckpt 记录: n_heads={ck_heads}  fourier_strategy={ck_strategy}  "
          f"max_peaks_n={ck_peaks}  n_layers={args['n_layers']}")
    print(f"      本次配置: n_heads={config.DREAMS2_N_HEADS}  "
          f"fourier_strategy={config.DREAMS2_FOURIER_STRAT}")
    check("n_heads 与预训练一致", config.DREAMS2_N_HEADS == ck_heads,
          f"{config.DREAMS2_N_HEADS} vs {ck_heads}")
    check("fourier_strategy 与预训练一致",
          config.DREAMS2_FOURIER_STRAT == ck_strategy,
          f"{config.DREAMS2_FOURIER_STRAT} vs {ck_strategy}")

    print("\n[3] 构造 V2 编码器并加载预训练权重")
    from model_dreaMS_harmonictsfm_2 import DreamsMSEncoderFid
    enc2 = DreamsMSEncoderFid(ckpt_path=ckpt_path, output_dim=config.MS_PE_DIM,
                              freeze_backbone=True)
    bb2 = enc2.backbone
    nh = bb2.transformer_encoder.atts[0].n_heads
    hd = bb2.transformer_encoder.atts[0].head_dim
    check("backbone n_heads == 配置值", nh == config.DREAMS2_N_HEADS,
          f"n_heads={nh}, head_dim={hd}")
    check("head_dim == d_model / n_heads", hd == bb2.d_model // nh)

    SKIP = ['ff_out.', 'ro_out.', 'mz_masking_loss.', 'mz_tokenizer.',
            'ff_mz_token.', 'spec_preproc.']
    expected = {k for k in ck_sd if not any(s in k for s in SKIP)}
    got = set(bb2.state_dict().keys())
    missing = expected - got
    extra = got - expected
    check("65/65 backbone 张量全部对上", not missing and not extra,
          f"missing={sorted(missing)[:3]} extra={sorted(extra)[:3]}")

    print("\n[3b] 全量架构参数对账 (ckpt 记录 → V2 实构结构)")
    te = bb2.transformer_encoder
    att0, ff0 = te.atts[0], te.ffs[0]
    n_lin = lambda m: sum(1 for x in m if isinstance(x, nn.Linear))
    _dformats = os.path.join('dreaMS', 'dreams', 'utils', 'dformats.py')
    if os.path.exists(_dformats):
        _dfs = open(_dformats, encoding='utf-8').read()
        _cls = _dfs.split('class DataFormatA(')[1].split('class ')[0] if 'class DataFormatA(' in _dfs else _dfs.split('class DataFormatA')[1]
        df_max_mz = float(re.search(r'^\s*max_mz\s*=\s*([\d.]+)', _cls, re.M).group(1))
        df_max_std = float(re.search(r'^\s*max_tbxic_stdev\s*=\s*([\d.eE-]+)', _cls, re.M).group(1))
        print(f"      DataFormatA: max_mz={df_max_mz}, max_tbxic_stdev={df_max_std} (读自 dformats.py)")
    else:
        df_max_mz, df_max_std = 1000.0, 1e-4
        print(f"      DataFormatA: max_mz={df_max_mz}, max_tbxic_stdev={df_max_std} "
              "(dreaMS 源码树缺失, 使用其默认常量)")

    rows = [
        ('n_layers',              args['n_layers'],                            te.n_layers),
        ('n_heads',               args['n_heads'],                             att0.n_heads),
        ('d_peak',                args['d_peak'],                              bb2.d_peak),
        ('d_fourier',             args['d_fourier'],                           bb2.d_fourier),
        ('d_mz_token',            args['d_mz_token'],                          bb2.d_mz_token),
        ('d_model=Σ(peak,fourier)', args['d_peak'] + args['d_fourier'] + args['d_mz_token'], bb2.d_model),
        ('no_transformer_bias',   args['no_transformer_bias'],                 not hasattr(att0, 'biases')),
        ('transformer FFN 无bias', (not args['no_transformer_bias']),          ff0.in_proj.bias is not None),
        ('ff_peak/ff_fourier 带bias', (not args['no_ffs_bias']),
                                      bb2.ff_peak.ff[0].bias is not None and bb2.ff_fourier.ff[0].bias is not None),
        ('scnorm=False→LayerNorm', (not args['scnorm']),                       isinstance(te.scales[0], nn.LayerNorm)),
        ('pre_norm',              args['pre_norm'],                            te.pre_norm),
        ('scales 个数 (pre_norm→2L+1)', args['n_layers'] * 2 + 1,              len(te.scales)),
        ('ff_fourier_depth',      args['ff_fourier_depth'],                    n_lin(bb2.ff_fourier.ff)),
        ('ff_peak_depth',         args['ff_peak_depth'],                       n_lin(bb2.ff_peak.ff)),
        ('ff_fourier_d',          args['ff_fourier_d'],                        bb2.ff_fourier.ff[0].out_features),
        ('charge_feature',        args['charge_feature'],                      bb2.charge_feature),
        ('max_mz (dformat A)',    df_max_mz,                                   bb2.max_mz),
        ('graphormer_mz_diffs',   args['graphormer_mz_diffs'],                 bb2.use_graphormer),
        ('attn_mech=dot-product', args['attn_mech'] == 'dot-product',          True),
        ('fourier_trainable',     args['fourier_trainable'],
                                  any(p.requires_grad for p in bb2.fourier_enc.parameters())),
    ]
    bad = []
    DECLARED_DEV = set()
    if args['graphormer_mz_diffs'] and not getattr(config, 'DREAMS2_GRAPHORMER', True):
        DECLARED_DEV.add('graphormer_mz_diffs')
    for name, exp, obs in rows:
        ok = (exp == obs)
        tag = 'ok ' if ok else ('dev ' if name in DECLARED_DEV else 'MISMATCH')
        if not ok and name not in DECLARED_DEV:
            bad.append(name)
        print(f"      [{tag}] {name:34} ckpt/期望={exp!s:>10}  实构={obs}")
    if DECLARED_DEV:
        print(f"      → 声明式偏离 {sorted(DECLARED_DEV)}: 见 config.py 的实测结论块 "
              f"(gd 在 100 峰下有害 / 60 峰下有益, 与 MAX_PEAKS 配套)")
    check("全量架构参数逐项一致 (除声明式偏离)", not bad, f"不一致项: {bad}")

    import model_dreaMS_harmonictsfm_2 as _M2
    atts_ok = all(isinstance(a, _M2.MultiheadAttentionFid) for a in te.atts)
    check("每层 attn 均为 MultiheadAttentionFid (gd 不会被丢)", atts_ok,
          f"{type(te.atts[0]).__name__}")

    b_loaded = bb2.fourier_enc.b.detach().squeeze(0).double()
    ref_lin = torch.tensor(
        [1 / (df_max_std * i) for i in range(2, math.ceil(1 / df_max_std), 2)] +
        [1 / (1 * i) for i in range(2, math.ceil(df_max_mz), 1)], dtype=torch.float64)
    same_len = (b_loaded.numel() == ref_lin.numel())
    rel = (b_loaded - ref_lin).abs().max().item() / ref_lin.max().item() if same_len else float('nan')
    print(f"      fourier 频率: 实载 {b_loaded.numel()} 个 vs lin_float_int 公式 {ref_lin.numel()} 个, "
          f"最大相对差 = {rel:.2e}")
    check("fourier 频率值 == lin_float_int 公式", same_len and rel < 1e-6, f"rel={rel:.2e}")

    print("\n[4] 改动生效性: V1(旧) vs V2(忠实) 的冻结前向差异")
    from model_dreaMS_harmonictsfm import DreaMSAdductHarmonicTsfmClassifier
    from dataloader import get_dataloader
    loader = get_dataloader(config.TEST_PKL, 8, augment=False, shuffle=False)
    batch = next(iter(loader))
    mz, inten = batch['mz'], batch['intensity']
    pmz = batch['precursor_mz'].squeeze(-1)
    spec = enc2._prepare_spec(mz, inten, pmz)
    enc2.eval()
    with torch.no_grad():
        emb2 = enc2.backbone_embedding(mz, inten, pmz)

    enc1 = DreamsMSEncoderFid.__mro__[1](ckpt_path=ckpt_path, output_dim=config.MS_PE_DIM,
                                         freeze_backbone=True)   # = DreamsMSEncoder (V1)
    enc1.eval()
    with torch.no_grad():
        emb1 = enc1.backbone_embedding(mz, inten, pmz)
    cos = torch.nn.functional.cosine_similarity(emb1, emb2, dim=-1).mean().item()
    rel = ((emb1 - emb2).norm(dim=-1) / emb1.norm(dim=-1)).mean().item()
    print(f"      前体嵌入 cosine(V1, V2) = {cos:.4f}   相对 L2 差 = {100 * rel:.1f}%")
    print("      (预期: 明显 < 1.0。若 ≈1.0 说明改动没生效)")
    check("V2 前向与 V1 显著不同", cos < 0.99, f"cosine={cos:.4f}")

    print("\n[5] graphormer bias: 秩一快捷式 vs 原版字面 (B,N,N,980) 的数值差")
    import model_dreaMS_harmonictsfm_2 as M2
    orig_exact = M2.GD_EXACT
    with torch.no_grad():
        _bb = enc2.backbone
        _spec = enc2._prepare_spec(mz, inten, pmz)
        ff_dbg = _bb.ff_fourier(_bb.fourier_enc(_spec[..., [0]].float()))
    try:
        M2.GD_EXACT = False
        b_quick = M2.pairwise_mz_diff(ff_dbg)
        M2.GD_EXACT = True
        b_exact = M2.pairwise_mz_diff(ff_dbg)
    finally:
        M2.GD_EXACT = orig_exact
    d_bias = (b_quick.sum(-1) - b_exact.sum(-1)).abs().max().item()
    print(f"      中间量: 字面版 {tuple(b_exact.shape)} vs 快捷式 {tuple(b_quick.shape)}, "
          f"sum(-1) 后 max|Δbias| = {d_bias:.3e}")
    check("GD_EXACT 开关确实改变计算路径", b_quick.shape != b_exact.shape and d_bias > 0,
          f"max|Δbias|={d_bias:.3e}")
    _saved_gd = enc2.backbone.use_graphormer
    enc2.backbone.use_graphormer = True
    try:
        with torch.no_grad():
            emb_quick = enc2.backbone_embedding(mz, inten, pmz)
        M2.GD_EXACT = True
        enc_ex = M2.DreamsMSEncoderFid(ckpt_path=ckpt_path, output_dim=config.MS_PE_DIM,
                                       freeze_backbone=True).eval()
        enc_ex.backbone.use_graphormer = True
        with torch.no_grad():
            emb_ex = enc_ex.backbone_embedding(mz, inten, pmz)
    finally:
        M2.GD_EXACT = orig_exact
        enc2.backbone.use_graphormer = _saved_gd
    d = (emb_ex - emb_quick).abs().max().item()
    d_rel = ((emb_ex - emb_quick).norm(dim=-1) / emb_ex.norm(dim=-1)).max().item()
    print(f"      最终输出: max|Δ| = {d:.3e}   最大相对 L2 差 = {100 * d_rel:.5f}%")
    print(f"      (bias 差 {d_bias:.1e} 远低于 attention logits (~10-30) 的 fp32 ulp ~1e-6, "
          f"故被舍入吸收)")
    check("两条路径数值等价 (相对差 < 0.01%)", d_rel < 1e-4, f"{100 * d_rel:.5f}%")

    print("\n[6] AMP autocast 前向 (fp16) 无 NaN")
    if torch.cuda.is_available():
        enc2 = enc2.cuda()
        with torch.no_grad(), torch.cuda.amp.autocast():
            emb_amp = enc2.backbone_embedding(mz.cuda(), inten.cuda(), pmz.cuda())
        finite = bool(torch.isfinite(emb_amp).all())
        check("autocast 下输出全为有限值", finite,
              f"NaN 数 = {int((~torch.isfinite(emb_amp)).sum())}")
    else:
        print("      (无 CUDA, 跳过)")

    print("\n[7] 与原版前向逐值等价 (oracle 按原源码转写: gd 字面 / -1e9 / 无 dtype 转换)")

    def original_forward(bb, spec):
        te_ = bb.transformer_encoder
        N = spec.shape[1]
        pad_mask = spec[:, :, 0] == 0
        peak_embs = bb.ff_peak(spec / torch.tensor([bb.max_mz, 1.0],
                                                   device=spec.device, dtype=spec.dtype))
        ff = bb.ff_fourier(bb.fourier_enc(spec[..., [0]]))
        x = torch.cat([peak_embs, ff], dim=-1)
        gd = (ff.unsqueeze(2) - ff.unsqueeze(1)) if bb.use_graphormer else None
        x = F.dropout(x, p=te_.residual_dropout, training=False)
        for i in range(te_.n_layers):
            residual = x
            x = te_.scales[2 * i](x)
            att = te_.atts[i]
            q, k, v = att._proj(x, end=3 * att.d_model).chunk(3, dim=-1)
            sh = lambda t: t.reshape(x.shape[0], -1, att.n_heads, att.head_dim).transpose(1, 2)
            q, k, v = sh(q), sh(k), sh(v)
            aw = torch.einsum('bhnd,bhdm->bhnm', q, k.transpose(-2, -1)) * att.scale
            if gd is not None:
                aw = aw + gd.sum(dim=-1).unsqueeze(1)            # layers.py:94-95
            aw = aw.masked_fill(pad_mask.unsqueeze(1).unsqueeze(-1), -1e9)   # (b)
            aw = F.softmax(aw, dim=-1)
            o = torch.bmm(aw.reshape(-1, N, N),
                          v.reshape(x.shape[0] * att.n_heads, N, att.head_dim))
            o = o.reshape(x.shape[0], att.n_heads, N, att.head_dim).transpose(1, 2).reshape(x.shape[0], N, -1)
            x = residual + att._proj(o, start=3 * att.d_model)
            residual = x
            x = te_.scales[2 * i + 1](x)
            y = F.relu(te_.ffs[i].in_proj(x))
            x = residual + te_.ffs[i].out_proj(y)
        return te_.scales[-1](x)

    bb2 = bb2.cpu()
    spec = spec.cpu()
    bb2.eval()
    with torch.no_grad():
        v2_out = bb2(spec)
        ref_out = original_forward(bb2, spec)
    dmax = (v2_out - ref_out).abs().max().item()
    rel_d = ((v2_out - ref_out).norm(dim=-1) / ref_out.norm(dim=-1)).max().item()
    cos_d = F.cosine_similarity(v2_out[:, 0, :], ref_out[:, 0, :], dim=-1).mean().item()
    print(f"      全 token: max|Δ| = {dmax:.3e}   最大相对 L2 差 = {rel_d:.3e}")
    print(f"      前体 token: cosine = {cos_d:.8f}")
    check("V2 与原版前向逐值等价 (相对差 < 1e-5)", rel_d < 1e-5, f"rel={rel_d:.3e}")
    check("输出全为有限值", bool(torch.isfinite(v2_out).all()))

    print("\n" + "=" * 78)
    print(f"结果: {len(PASS)} 项通过, {len(FAIL)} 项失败")
    if FAIL:
        for f in FAIL:
            print(f"  FAILED: {f}")
        return 1
    print("全部通过。注意: DREAMS2_GRAPHORMER=False 是声明式偏离(见 config.py 实测结论),")
    print("此时本脚本验证的是「除 gd 之外一切与原型一致」; 开回 True 才是逐值等价。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
