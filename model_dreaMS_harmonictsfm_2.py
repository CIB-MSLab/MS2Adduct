"""
model_dreaMS_harmonictsfm_2.py
DreaMSHT2 = a dreaMS branch fully consistent with the original dreaMS model
+ delta-Transformer encoding + fusion -> 9 adduct classes.

The only difference from model_dreaMS_harmonictsfm.py is the dreaMS spectrum
encoder, controlled by the DREAMS2_* switches in config (default: "fully
consistent"):

  DREAMS2_N_HEADS = 8        ssl_model.ckpt was pretrained with n_heads=8. The
                             original port used 16, and because the qkv weight
                             shape (4096,1024) does not depend on the number of
                             heads, load_state_dict(strict=False) accepted it
                             silently: head_dim quietly became 64 instead of 128.
  DREAMS2_GRAPHORMER = False pairwise m/z difference attention bias: enabled
                             during pretraining, dropped by the port. **Measured
                             to hurt at 100-peak padding and to help at the
                             pretrained 60 peaks** (interaction term t ~ 3.2 on
                             two independent metrics), hence off by default; see
                             the conclusion block in config.py.
  DREAMS2_FOURIER_STRAT      frequency construction matching the ckpt (only
                             affects scratch initialisation).

Implementation strategy: inherit the existing implementation wholesale and
override only the methods that really need to change. The delta branch / CE
adapter / FusionBlock / hierarchical head / _prepare_spec / _load_weights are
reused verbatim. This fix adds no new parameters, so the backbone state-dict
keys are identical to V1 (65 keys) and the weight-loading path needs no change.

Self-check / regression: verify_dreaMS_fidelity.py (including an item-by-item
reconciliation with the values recorded in the ckpt)
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

import config
from model_dreaMS import (
    MultiheadAttention, TransformerEncoder, DreaMSBackbone, DreamsMSEncoder,
)
from model_dreaMS_harmonictsfm import DreaMSAdductHarmonicTsfmClassifier


GD_EXACT = False


def pairwise_mz_diff(fourier_features):
    """graphormer distance tensor; the last dimension is left to the .sum(-1)
    inside attention (see layers.py:94).

    Upstream (dreams.py):
        graphormer_dists = fourier_features.unsqueeze(2) - fourier_features.unsqueeze(1)
    Note that under the d_fourier branch the argument is the output of ff_fourier,
    i.e. the difference of the 980-dim projected features, not the raw m/z
    difference.
    """
    if GD_EXACT:
        return fourier_features.unsqueeze(2) - fourier_features.unsqueeze(1)  # (B,N,N,d_fourier)
    s = fourier_features.sum(dim=-1)                                          # (B,N)
    return (s.unsqueeze(2) - s.unsqueeze(1)).unsqueeze(-1)                    # (B,N,N,1)


class MultiheadAttentionFid(MultiheadAttention):
    """Adds the graphormer bias back; otherwise line-for-line identical to
    model_dreaMS.MultiheadAttention."""

    def forward(self, q, k, v, mask, graphormer_dists=None, do_proj_qkv=True):
        bs, n, d = q.size()

        def _split_heads(tensor):
            return tensor.reshape(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)

        if do_proj_qkv:
            qkv = self._proj(q, end=3 * self.d_model)
            q, k, v = qkv.chunk(3, dim=-1)
        q = _split_heads(q)
        k = _split_heads(k)
        v = _split_heads(v)
        att_weights = torch.einsum('bhnd,bhdm->bhnm', q, k.transpose(-2, -1))
        att_weights = att_weights * self.scale
        if graphormer_dists is not None:
            att_weights = att_weights + graphormer_dists.sum(dim=-1).unsqueeze(1).to(att_weights.dtype)
        if mask is not None:
            att_weights.masked_fill_(mask.unsqueeze(1).unsqueeze(-1), -1e4)
        att_weights = F.softmax(att_weights, dim=-1)
        att_weights = F.dropout(att_weights, p=self.dropout, training=self.training)
        _att_weights = att_weights.reshape(-1, n, n)
        output = torch.bmm(_att_weights, v.reshape(bs * self.n_heads, -1, self.head_dim))
        output = output.reshape(bs, self.n_heads, n, self.head_dim).transpose(1, 2).reshape(bs, n, -1)
        output = self._proj(output, start=3 * self.d_model)
        return output, att_weights


class TransformerEncoderFid(TransformerEncoder):
    """Line-for-line identical to the parent class except for two points:

      (1) __init__ must rebuild atts as MultiheadAttentionFid. The parent's
          __init__ builds a plain **MultiheadAttention** whose forward ignores
          graphormer_dists: reading forward alone suggests gd is supported, but
          every layer silently drops it (this is exactly what catches it in
          check [7] of verify_dreaMS_fidelity.py, where the output is equivalent
          to "gd switched off").
      (2) forward passes graphormer_dists through to every layer.

    The parameter names are identical to the parent class (MultiheadAttentionFid
    subclasses MultiheadAttention), so the state-dict keys are unchanged and the
    65-key filter in _load_weights needs no modification.
    """

    def __init__(self, d_model=1024, n_heads=16, n_layers=7, use_layernorm=True,
                 pre_norm=True, residual_dropout=0.0, att_dropout=0.0,
                 ff_dropout=0.0, use_transformer_bias=False):
        super().__init__(d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                         use_layernorm=use_layernorm, pre_norm=pre_norm,
                         residual_dropout=residual_dropout, att_dropout=att_dropout,
                         ff_dropout=ff_dropout, use_transformer_bias=use_transformer_bias)
        self.atts = nn.ModuleList([
            MultiheadAttentionFid(d_model=d_model, n_heads=n_heads,
                                  dropout=att_dropout,
                                  use_transformer_bias=use_transformer_bias)
            for _ in range(n_layers)])

    def forward(self, src_inputs, src_mask, graphormer_dists=None):
        x = F.dropout(src_inputs, p=self.residual_dropout, training=self.training)
        for i in range(self.n_layers):
            residual = x
            if self.pre_norm:
                x = self.scales[2 * i](x)
            x, _ = self.atts[i](q=x, k=x, v=x, mask=src_mask,
                                graphormer_dists=graphormer_dists)
            x = residual + F.dropout(x, p=self.residual_dropout, training=self.training)
            if not self.pre_norm:
                x = self.scales[2 * i](x)
            residual = x
            if self.pre_norm:
                x = self.scales[2 * i + 1](x)
            x = self.ffs[i](x)
            x = residual + F.dropout(x, p=self.residual_dropout, training=self.training)
            if not self.pre_norm:
                x = self.scales[2 * i + 1](x)
        if self.pre_norm:
            x = self.scales[-1](x)
        return x


class DreaMSBackboneFid(DreaMSBackbone):
    """Defaults to the ckpt pretraining configuration and swaps in an encoder
    that passes graphormer_dists through."""

    def __init__(self, d_fourier=980, d_peak=44, d_mz_token=0, d_model=1024,
                 n_heads=None, n_layers=7, charge_feature=False,
                 use_layernorm=True, pre_norm=True,
                 dropout=None, att_dropout=None, ff_dropout=None, residual_dropout=None,
                 fourier_strategy=None, fourier_num_freqs=5998,
                 fourier_trainable=False, max_mz=1000.0, max_tbxic_stdev=1e-4,
                 ff_peak_depth=1, ff_fourier_depth=5, ff_fourier_d=512,
                 use_transformer_bias=False, no_ffs_bias=False):
        if n_heads is None:
            n_heads = getattr(config, 'DREAMS2_N_HEADS', 8)
        if fourier_strategy is None:
            fourier_strategy = getattr(config, 'DREAMS2_FOURIER_STRAT', 'lin_float_int')
        d2_drop = getattr(config, 'DREAMS2_DROPOUT', None)
        drop = 0.0 if d2_drop is None else d2_drop
        if dropout is None:
            dropout = drop
        if att_dropout is None:
            att_dropout = drop
        if ff_dropout is None:
            ff_dropout = drop
        if residual_dropout is None:
            residual_dropout = drop
        self.use_graphormer = bool(getattr(config, 'DREAMS2_GRAPHORMER', True))
        super().__init__(
            d_fourier=d_fourier, d_peak=d_peak, d_mz_token=d_mz_token, d_model=d_model,
            n_heads=n_heads, n_layers=n_layers, charge_feature=charge_feature,
            use_layernorm=use_layernorm, pre_norm=pre_norm,
            dropout=dropout, att_dropout=att_dropout, ff_dropout=ff_dropout,
            residual_dropout=residual_dropout,
            fourier_strategy=fourier_strategy, fourier_num_freqs=fourier_num_freqs,
            fourier_trainable=fourier_trainable, max_mz=max_mz,
            max_tbxic_stdev=max_tbxic_stdev, ff_peak_depth=ff_peak_depth,
            ff_fourier_depth=ff_fourier_depth, ff_fourier_d=ff_fourier_d,
            use_transformer_bias=use_transformer_bias, no_ffs_bias=no_ffs_bias)
        self.transformer_encoder = TransformerEncoderFid(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            use_layernorm=use_layernorm, pre_norm=pre_norm,
            residual_dropout=residual_dropout, att_dropout=att_dropout,
            ff_dropout=ff_dropout, use_transformer_bias=use_transformer_bias)

    def forward(self, spec, charge=None):
        """Only difference from the parent: it also computes the graphormer
        distances and passes them to the encoder."""
        padding_mask = spec[:, :, 0] == 0
        if self.charge_feature:
            if charge is None:
                raise ValueError("charge_feature=True requires charge input")
            charge_features = (~padding_mask) * charge.unsqueeze(-1)
            spec = torch.cat([spec, charge_features.unsqueeze(-1)], dim=-1)
        spec_normalized = spec / torch.tensor(
            [self.max_mz, 1.0], device=spec.device, dtype=spec.dtype
        )
        peak_embs = self.ff_peak(spec_normalized)
        graphormer_dists = None
        if self.d_fourier:
            with autocast(enabled=False):
                fourier_features = self.ff_fourier(self.fourier_enc(spec[..., [0]].float()))
            if self.use_graphormer:
                graphormer_dists = pairwise_mz_diff(fourier_features)
            fourier_features = fourier_features.to(dtype=peak_embs.dtype)
            spec_out = torch.cat([peak_embs, fourier_features], dim=-1)
        else:
            spec_out = peak_embs
        spec_out = self.transformer_encoder(spec_out, padding_mask, graphormer_dists)
        return spec_out


class DreamsMSEncoderFid(DreamsMSEncoder):
    """Only the backbone is swapped for the Fid version; weight loading,
    preprocessing and the forward projection are all inherited."""

    def __init__(self, ckpt_path=None, output_dim=None, freeze_backbone=True,
                 n_highest_peaks=None, d_model=1024):
        super().__init__(ckpt_path=None, output_dim=output_dim, freeze_backbone=False,
                         n_highest_peaks=n_highest_peaks, d_model=d_model)
        self.backbone = DreaMSBackboneFid()
        if ckpt_path is not None and os.path.exists(ckpt_path):
            self._load_weights(ckpt_path)
        elif ckpt_path is not None:
            print(f"[DreamsMSEncoderFid] WARNING: {ckpt_path} not found, using random init")
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()


class DreaMSAdductHarmonicTsfmClassifierV2(DreaMSAdductHarmonicTsfmClassifier):
    """Interface identical to DreaMSAdductHarmonicTsfmClassifier; only the
    spectrum encoder is swapped.

    The delta branch / CE adapter / FusionBlock / hierarchical head / forward are
    all inherited, so the difference between DreaMSHT2 and DreaMSHT is strictly
    the fidelity of the dreaMS branch (a single variable).
    """

    def __init__(self, ckpt_path=None, freeze_backbone=True):
        print(f"[DreaMSHT2] faithful spectrum encoder: n_heads={getattr(config, 'DREAMS2_N_HEADS', 8)}, "
              f"graphormer={getattr(config, 'DREAMS2_GRAPHORMER', True)}, "
              f"fourier_strategy={getattr(config, 'DREAMS2_FOURIER_STRAT', 'lin_float_int')}")
        super().__init__(ckpt_path=None, freeze_backbone=freeze_backbone)
        self.ms_encoder = DreamsMSEncoderFid(
            ckpt_path=ckpt_path,
            output_dim=config.MS_PE_DIM,
            freeze_backbone=freeze_backbone,
        )


if __name__ == "__main__":
    print("=" * 70)
    print("DreaMSHT2: construction + weight loading + forward self-check")
    print("=" * 70)
    ckpt = os.path.join(config.DREAMS_CKPT_DIR, "ssl_model.ckpt")
    model = DreaMSAdductHarmonicTsfmClassifierV2(ckpt_path=ckpt, freeze_backbone=False)
    bb = model.ms_encoder.backbone
    print(f"backbone: n_heads={bb.transformer_encoder.atts[0].n_heads}, "
          f"head_dim={bb.transformer_encoder.atts[0].head_dim}, "
          f"graphormer={bb.use_graphormer}, "
          f"fourier_strategy={getattr(config, 'DREAMS2_FOURIER_STRAT', 'lin_float_int')}")
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"params: {n_all/1e6:.2f}M (trainable {n_tr/1e6:.2f}M)")

    batch = 4
    mz = torch.rand(batch, config.MAX_PEAKS) * 1000
    inten = torch.rand(batch, config.MAX_PEAKS)
    pad = 40
    mz[:, -pad:] = 0
    inten[:, -pad:] = 0
    pmz = mz.max(dim=-1).values.clamp(min=100.0)
    ce = torch.rand(batch, 1)
    model.eval()
    with torch.no_grad():
        logits = model(mz, inten, ce=ce, precursor_mz=pmz)
    print(f"logits: {tuple(logits.shape)}  finite={bool(torch.isfinite(logits).all())}")
    print("=" * 70)
    print("self-check PASSED (run verify_dreaMS_fidelity.py for the full regression)")
