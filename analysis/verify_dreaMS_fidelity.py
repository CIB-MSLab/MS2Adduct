"""
verify_dreaMS_fidelity.py
Regression self-check for DreaMSHT2 (the dreaMS branch that faithfully
reproduces the original dreaMS model).

Written before the implementation: it has to fail until the implementation is
complete. Checks:

  1. switch snapshot: print the actual DREAMS2_* values (for experiment-log audit)
  2. pretraining-config reconciliation: read n_heads / fourier_strategy /
     max_peaks_n straight out of ssl_model.ckpt and compare them item by item
     with the backbone built here. **This is the core regression test of the
     fix** -- the head-count mismatch went unnoticed for so long precisely
     because the ckpt kwargs could not be read and nobody reconciled them.
  3. weight loading: 65/65 backbone tensors, no missing / unexpected
  4. the change takes effect: the frozen forwards of V2 and V1 must differ
     significantly (otherwise a switch has silently stopped working)
  5. numeric difference between the gd shortcut and the literal materialisation
     of the original (algebraically equivalent, ~1e-6 expected in fp32)
  6. no NaN in the forward pass under AMP autocast (the original -1e9 turns
     into NaN in fp16, which is why -1e4 is kept)

Run: python verify_dreaMS_fidelity.py
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
    """Read the pretraining args out of the Lightning ckpt.

    The ckpt pickles msml classes, so stub modules must be installed before
    torch.load can run. map_location='meta' avoids loading the 1.2 GB of
    weights into memory.
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
    print("DreaMSHT2 fidelity self-check")
    print("=" * 78)

    print("\n[1] DREAMS2_* switch snapshot")
    snap = {k: getattr(config, k, '<undefined>') for k in
            ['DREAMS2_N_HEADS', 'DREAMS2_GRAPHORMER', 'DREAMS2_FOURIER_STRAT',
             'DREAMS2_MAX_PEAKS', 'DREAMS2_DROPOUT']}
    for k, v in snap.items():
        print(f"      {k} = {v!r}")
    check("DREAMS2_* switches are defined", all(v != '<undefined>' for v in snap.values()))

    ckpt_path = os.path.join(config.DREAMS_CKPT_DIR, "ssl_model.ckpt")
    if not os.path.exists(ckpt_path):
        print(f"\n!! {ckpt_path} not found, cannot continue")
        return 1

    print("\n[2] reconcile with the pretraining configuration recorded in ssl_model.ckpt")
    args, ck_sd = read_ckpt_kwargs(ckpt_path)
    ck_heads = args['n_heads']
    ck_strategy = args['fourier_strategy']
    ck_peaks = args['max_peaks_n']
    print(f"      ckpt records: n_heads={ck_heads}  fourier_strategy={ck_strategy}  "
          f"max_peaks_n={ck_peaks}  n_layers={args['n_layers']}")
    print(f"      current config: n_heads={config.DREAMS2_N_HEADS}  "
          f"fourier_strategy={config.DREAMS2_FOURIER_STRAT}")
    check("n_heads matches pretraining", config.DREAMS2_N_HEADS == ck_heads,
          f"{config.DREAMS2_N_HEADS} vs {ck_heads}")
    check("fourier_strategy matches pretraining",
          config.DREAMS2_FOURIER_STRAT == ck_strategy,
          f"{config.DREAMS2_FOURIER_STRAT} vs {ck_strategy}")

    print("\n[3] build the V2 encoder and load the pretrained weights")
    from model_dreaMS_harmonictsfm_2 import DreamsMSEncoderFid
    enc2 = DreamsMSEncoderFid(ckpt_path=ckpt_path, output_dim=config.MS_PE_DIM,
                              freeze_backbone=True)
    bb2 = enc2.backbone
    nh = bb2.transformer_encoder.atts[0].n_heads
    hd = bb2.transformer_encoder.atts[0].head_dim
    check("backbone n_heads == configured value", nh == config.DREAMS2_N_HEADS,
          f"n_heads={nh}, head_dim={hd}")
    check("head_dim == d_model / n_heads", hd == bb2.d_model // nh)

    SKIP = ['ff_out.', 'ro_out.', 'mz_masking_loss.', 'mz_tokenizer.',
            'ff_mz_token.', 'spec_preproc.']
    expected = {k for k in ck_sd if not any(s in k for s in SKIP)}
    got = set(bb2.state_dict().keys())
    missing = expected - got
    extra = got - expected
    check("all 65/65 backbone tensors match", not missing and not extra,
          f"missing={sorted(missing)[:3]} extra={sorted(extra)[:3]}")

    print("\n[3b] reconcile every architecture parameter (ckpt record -> the V2 actually built)")
    te = bb2.transformer_encoder
    att0, ff0 = te.atts[0], te.ffs[0]
    n_lin = lambda m: sum(1 for x in m if isinstance(x, nn.Linear))
    _dformats = os.path.join('dreaMS', 'dreams', 'utils', 'dformats.py')
    if os.path.exists(_dformats):
        _dfs = open(_dformats, encoding='utf-8').read()
        _cls = _dfs.split('class DataFormatA(')[1].split('class ')[0] if 'class DataFormatA(' in _dfs else _dfs.split('class DataFormatA')[1]
        df_max_mz = float(re.search(r'^\s*max_mz\s*=\s*([\d.]+)', _cls, re.M).group(1))
        df_max_std = float(re.search(r'^\s*max_tbxic_stdev\s*=\s*([\d.eE-]+)', _cls, re.M).group(1))
        print(f"      DataFormatA: max_mz={df_max_mz}, max_tbxic_stdev={df_max_std} (read from dformats.py)")
    else:
        df_max_mz, df_max_std = 1000.0, 1e-4
        print(f"      DataFormatA: max_mz={df_max_mz}, max_tbxic_stdev={df_max_std} "
              "(dreaMS source tree missing, using its default constants)")

    rows = [
        ('n_layers',              args['n_layers'],                            te.n_layers),
        ('n_heads',               args['n_heads'],                             att0.n_heads),
        ('d_peak',                args['d_peak'],                              bb2.d_peak),
        ('d_fourier',             args['d_fourier'],                           bb2.d_fourier),
        ('d_mz_token',            args['d_mz_token'],                          bb2.d_mz_token),
        ('d_model=Σ(peak,fourier)', args['d_peak'] + args['d_fourier'] + args['d_mz_token'], bb2.d_model),
        ('no_transformer_bias',   args['no_transformer_bias'],                 not hasattr(att0, 'biases')),
        ('transformer FFN without bias', (not args['no_transformer_bias']),          ff0.in_proj.bias is not None),
        ('ff_peak/ff_fourier with bias', (not args['no_ffs_bias']),
                                      bb2.ff_peak.ff[0].bias is not None and bb2.ff_fourier.ff[0].bias is not None),
        ('scnorm=False -> LayerNorm', (not args['scnorm']),                       isinstance(te.scales[0], nn.LayerNorm)),
        ('pre_norm',              args['pre_norm'],                            te.pre_norm),
        ('number of scales (pre_norm -> 2L+1)', args['n_layers'] * 2 + 1,              len(te.scales)),
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
        print(f"      [{tag}] {name:34} ckpt/expected={exp!s:>10}  built={obs}")
    if DECLARED_DEV:
        print(f"      -> declared deviations {sorted(DECLARED_DEV)}: see the measured-conclusion block in config.py "
              f"(gd hurts at 100 peaks and helps at 60, coupled with MAX_PEAKS)")
    check("every architecture parameter matches (apart from the declared deviations)", not bad, f"mismatches: {bad}")

    import model_dreaMS_harmonictsfm_2 as _M2
    atts_ok = all(isinstance(a, _M2.MultiheadAttentionFid) for a in te.atts)
    check("every layer's attention is MultiheadAttentionFid (gd is not dropped)", atts_ok,
          f"{type(te.atts[0]).__name__}")

    b_loaded = bb2.fourier_enc.b.detach().squeeze(0).double()
    ref_lin = torch.tensor(
        [1 / (df_max_std * i) for i in range(2, math.ceil(1 / df_max_std), 2)] +
        [1 / (1 * i) for i in range(2, math.ceil(df_max_mz), 1)], dtype=torch.float64)
    same_len = (b_loaded.numel() == ref_lin.numel())
    rel = (b_loaded - ref_lin).abs().max().item() / ref_lin.max().item() if same_len else float('nan')
    print(f"      fourier frequencies: {b_loaded.numel()} loaded vs {ref_lin.numel()} from the lin_float_int formula, "
          f"max relative difference = {rel:.2e}")
    check("fourier frequency values == the lin_float_int formula", same_len and rel < 1e-6, f"rel={rel:.2e}")

    print("\n[4] the change takes effect: frozen-forward difference between V1 (old) and V2 (faithful)")
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
    print(f"      precursor embedding cosine(V1, V2) = {cos:.4f}   relative L2 difference = {100 * rel:.1f}%")
    print("      (expected: clearly < 1.0. A value near 1.0 means the change did not take effect)")
    check("V2 forward differs significantly from V1", cos < 0.99, f"cosine={cos:.4f}")

    print("\n[5] graphormer bias: numeric difference between the rank-one shortcut and the literal original (B,N,N,980)")
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
    print(f"      intermediate: literal {tuple(b_exact.shape)} vs shortcut {tuple(b_quick.shape)}, "
          f"max|delta bias| after sum(-1) = {d_bias:.3e}")
    check("the GD_EXACT switch really changes the computation path", b_quick.shape != b_exact.shape and d_bias > 0,
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
    print(f"      final output: max|delta| = {d:.3e}   max relative L2 difference = {100 * d_rel:.5f}%")
    print(f"      (the bias difference {d_bias:.1e} is far below the fp32 ulp ~1e-6 of the attention logits (~10-30), "
          f"so it is absorbed by rounding)")
    check("the two paths are numerically equivalent (relative difference < 0.01%)", d_rel < 1e-4, f"{100 * d_rel:.5f}%")

    print("\n[6] AMP autocast forward (fp16) has no NaN")
    if torch.cuda.is_available():
        enc2 = enc2.cuda()
        with torch.no_grad(), torch.cuda.amp.autocast():
            emb_amp = enc2.backbone_embedding(mz.cuda(), inten.cuda(), pmz.cuda())
        finite = bool(torch.isfinite(emb_amp).all())
        check("all outputs are finite under autocast", finite,
              f"NaN count = {int((~torch.isfinite(emb_amp)).sum())}")
    else:
        print("      (no CUDA, skipped)")

    print("\n[7] value-for-value equivalence with the original forward (oracle transcribed from the original source: literal gd / -1e9 / no dtype casts)")

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
    print(f"      all tokens: max|delta| = {dmax:.3e}   max relative L2 difference = {rel_d:.3e}")
    print(f"      precursor token: cosine = {cos_d:.8f}")
    check("V2 is value-for-value equivalent to the original forward (relative difference < 1e-5)", rel_d < 1e-5, f"rel={rel_d:.3e}")
    check("all outputs are finite", bool(torch.isfinite(v2_out).all()))

    print("\n" + "=" * 78)
    print(f"result: {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"  FAILED: {f}")
        return 1
    print("all checks passed. Note: DREAMS2_GRAPHORMER=False is a declared deviation (see the measured conclusions in config.py);")
    print('here the script verifies "everything matches the original apart from gd"; only with True does it become value-for-value equivalent.')
    return 0


if __name__ == "__main__":
    sys.exit(main())
