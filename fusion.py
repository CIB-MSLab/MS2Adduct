"""
fusion.py
Fusion of the dreaMS spectrum features d with the delta-branch features h
(shared by the MLP and the Transformer encoders).

FUSION in {concat, hadamard, gated, ffn}:
  concat    [d; h]                        (2*dim)
  hadamard  [d; h; d*h]                   (3*dim, explicit per-dimension interaction)
  gated     g=sigmoid(W[d;h]); g*d+(1-g)*h  (dim, soft routing)
  ffn       d + h + FFN([d;h]), then LN   (dim, non-linear residual fusion)
"""
import math
import torch
import torch.nn as nn


class CEAdapter(nn.Module):
    """Ways to inject collision energy (CE) into the dreaMS spectrum features d.

    mode:
      gate  (default, the original EnergyGate): a sigmoid gate vector multiplies
            d elementwise
      film  :  scale=1+tanh(...), shift=... -> d' = scale*d + shift
      cat   :  leaves d unmodulated; projects a CE descriptor vector aux that
               the classifier concatenates onto the fused features
      none  :  CE is not used

    use_polarity: concatenates the ion-mode (positive/negative) embedding with
      the CE embedding and projects them jointly, i.e. "polarity as an input,
      fed in together with CE". The ion mode is known at acquisition time (see
      the notes in config), so this adds a conditioning path to the gate or to
      the descriptor vector rather than another prediction head.
    """

    def __init__(self, feat_dim: int, mode: str = 'gate',
                 num_ce_bins: int = 64, ce_embed_dim: int = 32, aux_dim: int = 64,
                 use_polarity: bool = False, n_polarity: int = 2,
                 polarity_embed_dim: int = 8):
        super().__init__()
        self.mode = mode
        self.num_ce_bins = num_ce_bins
        self.aux_dim = aux_dim if mode == 'cat' else 0
        self.ce_embedding = nn.Embedding(num_ce_bins, ce_embed_dim)
        self.use_polarity = use_polarity
        emb_dim = ce_embed_dim
        if use_polarity:
            self.polarity_embedding = nn.Embedding(n_polarity, polarity_embed_dim)
            emb_dim += polarity_embed_dim
        else:
            self.polarity_embedding = None
        if mode == 'gate':
            self.act = nn.Sigmoid()
            self.proj = nn.Linear(emb_dim, feat_dim)
        elif mode == 'film':
            self.proj = nn.Linear(emb_dim, 2 * feat_dim)
            self.ln = nn.LayerNorm(feat_dim)
        elif mode == 'cat':
            self.proj = nn.Linear(emb_dim, aux_dim)
        elif mode != 'none':
            raise ValueError(f"Unknown CE_FUSION: {mode}")

    def _embed(self, energy: torch.Tensor,
               polarity: torch.Tensor = None) -> torch.Tensor:
        energy = torch.clamp(energy, 0.0, 150.0)
        bin_idx = torch.floor(energy / 150.0 * self.num_ce_bins).long()
        bin_idx = torch.where(bin_idx >= self.num_ce_bins,
                              torch.tensor(self.num_ce_bins - 1, device=energy.device),
                              bin_idx).squeeze(-1)
        emb = self.ce_embedding(bin_idx)
        if not self.use_polarity:
            return emb
        if polarity is None:
            raise ValueError(
                "POLARITY_AS_INPUT=True but the forward pass received no polarity; "
                "the training/evaluation loop must pass batch['polarity'] into the model")
        emb = torch.cat([emb, self.polarity_embedding(polarity.long().view(-1))], dim=-1)
        return emb

    def apply(self, feat: torch.Tensor, energy: torch.Tensor,
              polarity: torch.Tensor = None):
        """Return (feat_mod, aux). aux is (B, aux_dim), always None outside cat mode."""
        emb = self._embed(energy, polarity)
        if self.mode == 'none':
            return feat, None
        if self.mode == 'gate':
            gate = self.act(self.proj(emb))
            return feat * gate, None
        if self.mode == 'film':
            p = self.proj(emb)
            scale = 1.0 + torch.tanh(p[..., :feat.shape[-1]])
            shift = p[..., feat.shape[-1]:]
            return self.ln(scale * feat + shift), None
        # cat
        return feat, self.proj(emb)


class FusionBlock(nn.Module):
    def __init__(self, dim: int, fusion: str = 'concat'):
        super().__init__()
        self.fusion = fusion
        self.dim = dim
        if fusion == 'concat':
            self.out_dim = 2 * dim
        elif fusion == 'hadamard':
            self.out_dim = 3 * dim
        elif fusion == 'gated':
            self.out_dim = dim
            self.gate = nn.Linear(2 * dim, dim)
        elif fusion == 'ffn':
            self.out_dim = dim
            self.proj = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU())
            self.ln = nn.LayerNorm(dim)
        elif fusion == 'gated_cat':
            self.out_dim = 3 * dim
            self.gate = nn.Linear(2 * dim, dim)
        elif fusion == 'bilinear':
            self.out_dim = dim
            self.proj = nn.Linear(dim, dim)
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
            self.ln = nn.LayerNorm(dim)
        elif fusion == 'film':
            self.out_dim = 2 * dim
            self.scale = nn.Sequential(nn.Linear(dim, dim), nn.Tanh())
            self.shift = nn.Linear(dim, dim)
            nn.init.zeros_(self.shift.weight)
            nn.init.zeros_(self.shift.bias)
        elif fusion == 'coattn':
            self.out_dim = dim
            self.mha = nn.MultiheadAttention(dim, 4, batch_first=True)
            self.norm = nn.LayerNorm(dim)
        elif fusion == 'tri_cat':
            self.out_dim = 3 * dim
        elif fusion == 'tri_attn':
            self.out_dim = dim
            self.mha = nn.MultiheadAttention(dim, 4, batch_first=True)
            self.norm = nn.LayerNorm(dim)
        elif fusion == 'tri_xattn':
            self.out_dim = dim
            self.mha = nn.MultiheadAttention(dim, 4, batch_first=True)
            self.q_norm = nn.LayerNorm(dim)
            self.kv_norm = nn.LayerNorm(dim)
            self.norm = nn.LayerNorm(dim)
        else:
            raise ValueError(f"Unknown FUSION: {fusion}")

    def forward(self, d: torch.Tensor, h: torch.Tensor,
                aux: torch.Tensor = None) -> torch.Tensor:
        h = h.to(dtype=d.dtype)
        cat = torch.cat([d, h], dim=-1)
        if self.fusion == 'concat':
            return cat
        if self.fusion == 'hadamard':
            return torch.cat([d, h, d * h], dim=-1)
        if self.fusion == 'gated':
            g = torch.sigmoid(self.gate(cat))
            return g * d + (1 - g) * h
        if self.fusion == 'ffn':
            return self.ln(d + h + self.proj(cat))
        if self.fusion == 'gated_cat':
            g = torch.sigmoid(self.gate(cat))
            return torch.cat([d, h, g * (d * h)], dim=-1)
        if self.fusion == 'bilinear':
            return self.ln(d + h + self.proj(d * h))
        if self.fusion == 'film':
            s = 1.0 + self.scale(h)
            dmod = s * d + self.shift(h)
            return torch.cat([dmod, h], dim=-1)
        if self.fusion in ('tri_cat', 'tri_attn', 'tri_xattn'):
            if self.fusion == 'tri_xattn' and aux is None:
                q = self.q_norm(d).unsqueeze(1)                   # (B,1,dim)
                kv = self.kv_norm(h.unsqueeze(1))                 # (B,1,dim)
                a, _ = self.mha(q, kv, kv)
                return self.norm(d + a.squeeze(1))                # (B,dim)
            if aux is None:
                raise ValueError(f"{self.fusion} requires a third input aux (the CE vector)")
            aux = aux.to(dtype=d.dtype)
            if self.fusion == 'tri_cat':
                return torch.cat([d, h, aux], dim=-1)
            if self.fusion == 'tri_xattn':
                q = self.q_norm(d).unsqueeze(1)                   # (B,1,dim)
                kv = self.kv_norm(torch.stack([h, aux], dim=1))   # (B,2,dim)
                a, _ = self.mha(q, kv, kv)
                return self.norm(d + a.squeeze(1))                # (B,dim)
            t = torch.stack([d, h, aux], dim=1)        # (B,3,dim)
            t = self.norm(t)
            a, _ = self.mha(t, t, t)
            return (t + a).mean(dim=1)                 # (B,dim)
        # coattn
        t = torch.stack([d, h], dim=1)          # (B,2,dim)
        t = self.norm(t)
        a, _ = self.mha(t, t, t)
        return (t + a).mean(dim=1)               # (B,dim)
