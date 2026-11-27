"""
model_dreaMS_harmonictsfm.py
DreaMS MS/MS encoder + a delta (neutral-loss) branch encoded by a
**Transformer** + searchable fusion.

Differences from the earlier delta-MLP implementation (delta through an MLP
followed by mean pooling):
  - delta branch: per-peak raw harmonic features (210) -> small Transformer
    (self-attention across peaks models the co-occurrence of neutral losses)
    -> [CLS] 256
  - fusion: config.FUSION in {concat, hadamard, gated, ffn}, enumerated and
    selected automatically
    concat    [d; h]                        (2*dim, baseline same as harmonic-MLP)
    hadamard  [d; h; d*h]                   (3*dim, explicit per-dimension interaction)
    gated     g=sigmoid(W[d;h]); g*d+(1-g)*h  (dim, soft routing)
    ffn       d + h + FFN([d;h]), then LN   (dim, non-linear residual fusion)
"""
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import config
from model import EnergyGate
from model_dreaMS import (
    DreaMSAdductClassifier,
    DreamsMSEncoder,
    FourierFeatures,
)
from fusion import FusionBlock, CEAdapter


def infer_delta_encoder(state_dict) -> str:
    """Infer the architecture of the per-peak delta branch from trained weights.

    The presence of `harmonic_enc.fourier_enc.b` means "fourier"; otherwise
    "bases". After the default moved to "fourier", this lets the loading path of
    an old ckpt fall back to the bases architecture automatically.
    """
    return "fourier" if any("harmonic_enc.fourier_enc.b" in k for k in state_dict) else "bases"


# ════════════════════════════════════════════════════════════════════
# Raw harmonic components of δ (shared primitive, computed in fp32)
# ════════════════════════════════════════════════════════════════════

def raw_harmonic_components(delta, bases=None, n_harm=None):
    """delta: (...,) or (B,N) or (B,N,N)
    When bases/n_harm are None the config globals are used (bit-identical to the
    previous behaviour)."""
    bases = config.HARMONIC_BASE_FREQUENCIES if bases is None else bases
    n_harm = getattr(config, 'N_HARMONICS_PER_BASE', 3) if n_harm is None else n_harm
    delta = delta.unsqueeze(-1)
    components = []
    for dk in bases:
        for h in range(n_harm):
            freq = 2 * math.pi / (dk / (2 ** h))
            components.append(torch.sin(freq * delta))
            components.append(torch.cos(freq * delta))
    return torch.cat(components, dim=-1)


def make_sham_bases(bases, seed=123):
    """Chemically meaningless base frequencies with a matched frequency range:
    each true base frequency is jittered by +-0.5 Da while keeping a 0.05 Da
    minimum separation."""
    rng = random.Random(seed)
    true = list(bases)
    out = []
    for b in true:
        for _ in range(10000):
            v = min(200.0, max(1.0, b + rng.uniform(-0.5, 0.5)))
            if min(abs(v - t) for t in true) < 0.05:
                continue
            if any(abs(v - u) < 0.05 for u in out):
                continue
            out.append(v)
            break
        else:
            raise RuntimeError(f"base frequency {b}: could not find a sham value")
    return sorted(out)


# ════════════════════════════════════════════════════════════════════
# δ Transformer encoder
# ════════════════════════════════════════════════════════════════════

class HarmonicTsfmEncoder(nn.Module):
    """
    Per-peak delta harmonic features -> input_proj -> self-attention Transformer
    -> [CLS] pooling. Computed in fp32 (AMP safe).
    """

    def __init__(self, dim_out: int = 256):
        super().__init__()
        n_bases = len(config.HARMONIC_BASE_FREQUENCIES)
        n_harm = getattr(config, 'N_HARMONICS_PER_BASE', 3)
        self.delta_encoder_mode = getattr(config, 'DELTA_ENCODER', 'bases')
        if self.delta_encoder_mode not in ('bases', 'fourier'):
            raise ValueError(
                "DELTA_ENCODER only supports 'bases' or 'fourier', "
                f"got {self.delta_encoder_mode!r}")
        self.delta_token_mode = getattr(config, 'DELTA_TOKEN_MODE', 'peak')
        if self.delta_token_mode not in ('peak', 'pair', 'peak_top5'):
            raise ValueError(
                "DELTA_TOKEN_MODE only supports 'peak'/'pair'/'peak_top5', "
                f"got {self.delta_token_mode!r}")
        self.anchor_k = int(getattr(config, 'DELTA_ANCHOR_K', 5))
        self.fourier_enc = None
        if self.delta_encoder_mode == 'fourier':
            self.fourier_enc = FourierFeatures(
                strategy=getattr(config, 'FOURIER_STRATEGY', 'lin_float_int'),
                x_min=float(getattr(config, 'FOURIER_X_MIN', 1e-4)),
                x_max=float(getattr(config, 'FOURIER_X_MAX', 2000.0)),
                trainable=bool(getattr(config, 'FOURIER_TRAINABLE', False)),
                funcs='both',
                num_freqs=int(getattr(config, 'FOURIER_NUM_FREQS', 5998)),
            )
            d_raw = self.fourier_enc.num_features()
        else:
            d_raw = n_bases * n_harm * 2
        dim = getattr(config, 'HSFM_DIM', 256)

        self.in_proj = nn.Sequential(
            nn.LayerNorm(d_raw),
            nn.Linear(d_raw, dim),
            nn.GELU(),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=getattr(config, 'HSFM_HEADS', 4),
            dim_feedforward=dim * getattr(config, 'HSFM_FFN_MULT', 4),
            dropout=getattr(config, 'HSFM_DROPOUT', 0.1),
            batch_first=True,
            activation='relu',
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=getattr(config, 'HSFM_LAYERS', 2))
        self.out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_out),
        )
        self.output_scale = nn.Parameter(torch.tensor(0.5))

        self.use_pairdiff = bool(getattr(config, 'PAIRDIFF_ENABLED', False))
        if self.use_pairdiff and self.delta_token_mode == 'pair':
            raise ValueError(
                "PAIRDIFF_ENABLED and DELTA_TOKEN_MODE='pair' are mutually exclusive: "
                "the peak pairs already are the tokens, so adding a pairwise attention "
                "edge bias would encode the same thing twice")
        self.pair_proj = None
        self.pair_bases = []
        self.pair_harm = 1
        self.pair_min_delta = float(getattr(config, 'PAIRDIFF_MIN_DELTA', 1.0))
        self.n_heads = getattr(config, 'HSFM_HEADS', 4)
        if self.use_pairdiff:
            bases = list(config.HARMONIC_BASE_SETS[
                getattr(config, 'PAIRDIFF_BASE_SET', 'both')])
            if getattr(config, 'PAIRDIFF_SHAM', False):
                bases = make_sham_bases(bases, int(getattr(config, 'PAIRDIFF_SHAM_SEED', 123)))
            self.pair_bases = bases
            self.pair_harm = int(getattr(config, 'PAIRDIFF_HARMONICS', 1))
            d_pair = len(bases) * self.pair_harm * 2
            self.pair_proj = nn.Sequential(
                nn.LayerNorm(d_pair),
                nn.Linear(d_pair, self.n_heads),
            )
            nn.init.zeros_(self.pair_proj[-1].weight)
            nn.init.zeros_(self.pair_proj[-1].bias)

    def forward(self, delta: torch.Tensor, mask: torch.Tensor,
                intensity: torch.Tensor = None) -> torch.Tensor:
        """delta:(B,N) mask: valid peak mask(B,N) intensity:(B,N) -> (B, dim_out) fp32

        intensity is only used to pick the anchor peaks when
        DELTA_TOKEN_MODE='peak_top5'.
        """
        from torch import autocast
        device_type = 'cuda' if delta.is_cuda else 'cpu'
        with autocast(device_type=device_type, enabled=False):
            raw = delta.float()
            delta_c = raw.clamp(0.0, 2000.0)
            if self.delta_token_mode == 'pair':
                return self._forward_pair(raw, mask)
            if self.delta_token_mode == 'peak_top5':
                return self._forward_anchor(raw, mask, intensity)
            if self.fourier_enc is not None:
                tok = self.in_proj(self.fourier_enc(delta_c.unsqueeze(-1)))
            else:
                tok = self.in_proj(raw_harmonic_components(delta_c))
            B, N, D = tok.shape
            cls = self.cls_token.expand(B, -1, -1)
            tok = torch.cat([cls, tok], dim=1)
            kp = ~torch.cat([torch.ones(B, 1, dtype=torch.bool, device=delta.device),
                             mask], dim=1)

            pair_bias = None
            if self.pair_proj is not None:
                pd = (raw.unsqueeze(-2) - raw.unsqueeze(-1)).abs()
                feats = raw_harmonic_components(pd, bases=self.pair_bases,
                                                n_harm=self.pair_harm)
                bias = self.pair_proj(feats)                  # (B,N,N,H)
                valid = mask.unsqueeze(-2) & mask.unsqueeze(-1)
                valid = valid & (pd > self.pair_min_delta)
                bias = bias * valid.unsqueeze(-1)
                bias = bias.permute(0, 3, 1, 2).reshape(
                    B * self.n_heads, N, N)
                pair_bias = torch.nn.functional.pad(bias, (1, 0, 1, 0))

            out = self.encoder(tok, mask=pair_bias, src_key_padding_mask=kp)
            h = self.out(out[:, 0, :]) * self.output_scale            # (B,dim_out)
        return h

    def _forward_pair(self, raw: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Peak-pair tokens: one token per valid peak pair, encoding
        |delta_i - delta_j|.

        Each sample is forwarded on its own: the number of pairs C(k,2) varies
        with the peak count of the sample, and padding to the largest pair count
        in the batch (up to 4,950) would inflate both memory use and wasted
        attention computation. Every pair is a valid key, so no padding mask is
        needed.
        """
        outs = []
        for b in range(raw.shape[0]):
            d = raw[b][mask[b]]
            k = d.numel()
            if k >= 2:
                ii, jj = torch.triu_indices(k, k, 1, device=d.device)
                pd = (d[ii] - d[jj]).abs().clamp(0.0, 2000.0)
            else:
                pd = d.new_zeros(0)
            tok = self.in_proj(self.fourier_enc(pd.unsqueeze(-1))).unsqueeze(0)
            cls = self.cls_token.expand(1, -1, -1)
            tok = torch.cat([cls, tok], dim=1)
            out = checkpoint(self.encoder, tok, use_reentrant=False) \
                if self.training else self.encoder(tok)
            outs.append(self.out(out[:, 0, :]) * self.output_scale)
        return torch.cat(outs, dim=0)

    def _forward_anchor(self, raw: torch.Tensor, mask: torch.Tensor,
                        intensity: torch.Tensor) -> torch.Tensor:
        """Precursor neutral loss + inter-peak neutral losses anchored on the
        top-K peaks by intensity.

        Token set = delta_i of every valid peak (as in 'peak' mode)
                   + |delta_a - delta_j| for every anchor peak a and every peak j
                     with **m/z smaller than a**.
        Anchors are the top-K valid peaks by intensity (K = DELTA_ANCHOR_K); the
        pairing is one-directional (mz_j < mz_a <=> delta_j > delta_a), so at
        k=100 there are at most about K*k/2 inter-peak tokens. Compared with
        'pair', this keeps only the neutral losses descending from K strong peaks
        instead of C(k,2)=4,950.

        Padded to the largest token count in the batch and forwarded in a single
        pass (at most 596) rather than looping per sample: a per-sample loop
        produces many small kernels at batch=26 and measured 3.7x the original
        peak mode.
        """
        if intensity is None:
            raise ValueError(
                "DELTA_TOKEN_MODE='peak_top5' requires intensity to select the anchor peaks")
        B, N = raw.shape
        device = raw.device
        vals_list, lens = [], []
        for b in range(B):
            idx = mask[b].nonzero(as_tuple=True)[0]
            d = raw[b][idx]
            it = intensity[b][idx].float()
            k = d.numel()
            parts = [d]
            if k > 1:
                K = min(self.anchor_k, k)
                top = torch.topk(it, K).indices
                pd = (d[top].unsqueeze(1) - d.unsqueeze(0)).abs()   # (K,k)
                keep = d.unsqueeze(0) > d[top].unsqueeze(1)         # (K,k)
                parts.append(pd[keep])
            v = torch.cat(parts).clamp(0.0, 2000.0)
            vals_list.append(v)
            lens.append(v.numel())

        L = max(lens)
        vals = raw.new_zeros(B, L)
        for b, v in enumerate(vals_list):
            vals[b, :v.numel()] = v
        if self.fourier_enc is not None:
            tok = self.in_proj(self.fourier_enc(vals.unsqueeze(-1)))
        else:
            tok = self.in_proj(raw_harmonic_components(vals))
        cls = self.cls_token.expand(B, -1, -1)
        tok = torch.cat([cls, tok], dim=1)                       # (B, 1+L, dim)
        pad = torch.arange(L, device=device)[None, :] >= \
            torch.tensor(lens, device=device)[:, None]
        kp = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=device),
                        pad], dim=1)
        out = checkpoint(self.encoder, tok, src_key_padding_mask=kp,
                         use_reentrant=False) \
            if self.training else self.encoder(tok, src_key_padding_mask=kp)
        return self.out(out[:, 0, :]) * self.output_scale


# ════════════════════════════════════════════════════════════════════
# Classifier: DreaMS + δ-Transformer + fusion
# ════════════════════════════════════════════════════════════════════

class DreaMSAdductHarmonicTsfmClassifier(DreaMSAdductClassifier):
    """Interface identical to DreaMSAdductHarmonicClassifier; only the delta
    encoding and the fusion differ."""

    MASKED_LOGIT = -1e4

    def __init__(self, ckpt_path=None, freeze_backbone=True):
        nn.Module.__init__(self)
        self.ms_encoder = DreamsMSEncoder(
            ckpt_path=ckpt_path,
            output_dim=config.MS_PE_DIM,
            freeze_backbone=freeze_backbone,
        )
        pol_in = bool(getattr(config, 'POLARITY_AS_INPUT', False))
        pol_mask = bool(getattr(config, 'POLARITY_MASK_OUTPUT', False))
        self.mask_output_by_polarity = pol_mask
        self.mask_start_epoch = int(getattr(config, 'POLARITY_MASK_START_EPOCH', 0))
        self.mask_inference_only = bool(
            getattr(config, 'POLARITY_MASK_INFERENCE_ONLY', False))
        self.current_epoch = 0
        self.accepts_polarity = pol_in or pol_mask or self.mask_inference_only
        pol_kw = dict(use_polarity=pol_in,
                      polarity_embed_dim=getattr(config, 'POLARITY_EMB_DIM', 8))
        ce_mode = getattr(config, 'CE_FUSION', 'gate')
        use_tri = getattr(config, 'FUSION', 'concat') in (
            'tri_cat', 'tri_attn', 'tri_xattn')
        self.use_tri = use_tri
        d_mode = ce_mode if ce_mode in ('gate', 'film', 'cat', 'none') else 'none'
        self.ce_adapter = CEAdapter(
            feat_dim=config.MS_PE_DIM,
            mode=d_mode,
            num_ce_bins=getattr(config, 'CE_NUM_BINS', 64),
            ce_embed_dim=getattr(config, 'CE_EMBED_DIM', 32),
            aux_dim=config.MS_PE_DIM if use_tri else getattr(config, 'CE_AUX_DIM', 64),
            **pol_kw,
        )
        delta_mode = {'gate_delta': 'gate', 'film_delta': 'film'}.get(ce_mode)
        self.ce_adapter_delta = CEAdapter(
            feat_dim=getattr(config, 'HARMONIC_EMB_DIM', config.MS_PE_DIM),
            mode=delta_mode,
            num_ce_bins=getattr(config, 'CE_NUM_BINS', 64),
            ce_embed_dim=getattr(config, 'CE_EMBED_DIM', 32),
            **pol_kw,
        ) if delta_mode else None
        self.harmonic_enc = HarmonicTsfmEncoder(
            dim_out=getattr(config, 'HARMONIC_EMB_DIM', config.MS_PE_DIM))
        self.fusion = FusionBlock(config.MS_PE_DIM, getattr(config, 'FUSION', 'concat'))

        feat_dim = self.fusion.out_dim + (0 if use_tri else self.ce_adapter.aux_dim)
        dropout = config.DROPOUT_RATE
        use_hierarchical = getattr(config, 'HIERARCHICAL_HEADS', False)
        if use_hierarchical:
            self.ion_mode_head = (nn.Linear(feat_dim, 2)
                                  if getattr(config, 'ION_AUX_ENABLED', True) else None)
            self.positive_head = nn.Sequential(
                nn.Linear(feat_dim, 128), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(128, len(self.POS_GLOB)))
            self.negative_head = nn.Sequential(
                nn.Linear(feat_dim, 128), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(128, len(self.NEG_GLOB)))
            self.classifier = None
            self._last_ion_logits = None
        else:
            self.classifier = nn.Sequential(
                nn.Linear(feat_dim, 128), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(128, config.NUM_ADDUCT_CLASSES))
            self.ion_mode_head = None
            self.positive_head = None
            self.negative_head = None

    def _precursor(self, mz, precursor_mz):
        if precursor_mz is None:
            return mz.max(dim=-1)[0]
        precursor_mz = precursor_mz.squeeze(-1)
        return torch.where(precursor_mz > 0, precursor_mz, mz.max(dim=-1)[0])

    def set_epoch(self, epoch):
        """Called once per epoch by the training loop; only used for the
        POLARITY_MASK_START_EPOCH warm-up check."""
        self.current_epoch = int(epoch)

    def _apply_polarity_mask(self, logits, polarity):
        """Mask the opposite-polarity classes out of the nine-class logits using
        the **given** ion mode.

        Note that this is entirely different from gating on the **predicted**
        polarity, which measured as a zero effect (analyze_polarity.py: of the
        1630 cross-polarity errors, the polarity head itself was wrong 96.3% of
        the time). Here the true polarity known at acquisition time is used, so
        this is equivalent to the oracle-polarity upper bound in that script.

        During training the mask can start late (mask_start_epoch): masking
        makes the gradient of the opposite-polarity logits permanently zero, so
        the warm-up phase first lets the nine classes compete freely, avoiding
        cutting off the only effective learning signal of the polarity input.
        """
        if polarity is None or not getattr(self, 'mask_output_by_polarity', False):
            return logits
        if self.training and (self.mask_inference_only
                              or self.current_epoch < self.mask_start_epoch):
            return logits
        ion = torch.tensor(self.GLOB_TO_ION, device=logits.device)[None, :]
        keep = ion == polarity.view(-1, 1).long()
        return logits.masked_fill(~keep, self.MASKED_LOGIT)

    def forward(self, mz, intensity, ce=None, precursor_mz=None,
                precomputed_emb=None, polarity=None):
        feat = self.ms_encoder(mz, intensity,
                               precursor_mz=precursor_mz,
                               precomputed_emb=precomputed_emb)
        aux = None
        if ce is not None and self.ce_adapter_delta is None:
            feat, aux = self.ce_adapter.apply(feat, ce, polarity)
            if not getattr(config, 'CE_TOKEN_ENABLED', True):
                aux = None

        if mz is not None:
            p = self._precursor(mz, precursor_mz)
            delta = p.unsqueeze(-1) - mz                      # (B,N)
            mask = mz > 0.0
            h = self.harmonic_enc(delta, mask, intensity)     # (B,dim) fp32
            if self.ce_adapter_delta is not None and ce is not None:
                h, _ = self.ce_adapter_delta.apply(h, ce, polarity)
        else:
            h = torch.zeros_like(feat)
        fused = self.fusion(feat, h, aux if self.use_tri else None)
        if aux is not None and not self.use_tri:
            fused = torch.cat([fused, aux.to(dtype=fused.dtype)], dim=-1)

        if self.classifier is not None:
            return self._apply_polarity_mask(self.classifier(fused), polarity)

        if self.ion_mode_head is not None:
            self._last_ion_logits = self.ion_mode_head(fused)
        pos_logits = self.positive_head(fused)
        neg_logits = self.negative_head(fused)
        logits = torch.full((fused.shape[0], config.NUM_ADDUCT_CLASSES),
                            self.MASKED_LOGIT, device=pos_logits.device,
                            dtype=pos_logits.dtype)
        logits[:, self.POS_GLOB] = pos_logits
        logits[:, self.NEG_GLOB] = neg_logits
        return self._apply_polarity_mask(logits, polarity)


if __name__ == "__main__":
    print("=" * 60)
    import os
    for fusion in ['concat', 'hadamard', 'gated', 'ffn']:
        config.FUSION = fusion
        torch.manual_seed(0)
        model = DreaMSAdductHarmonicTsfmClassifier(
            ckpt_path=os.path.join(config.DREAMS_CKPT_DIR, "ssl_model.ckpt"),
            freeze_backbone=True)
        model.train()
        B, N = 2, 40
        mz = torch.zeros(B, N); mz[:, :20] = torch.rand(B, 20) * 800 + 50
        intensity = torch.zeros(B, N); intensity[:, :20] = torch.rand(B, 20)
        prec = torch.tensor([500.0, 300.0])
        logits = model(mz, intensity, precursor_mz=prec)
        loss = logits.sum(); loss.backward()
        n = sum(p.numel() for p in model.harmonic_enc.parameters())
        print(f"fusion={fusion:<8s} out={tuple(logits.shape)} "
              f"harm_params={n/1e3:.0f}K OK")
    print("=" * 60)
