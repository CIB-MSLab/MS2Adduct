"""
model_dreaMS_harmonic.py
DreaMS MS/MS encoder + 中性丢失 δ = precursor_mz − m/z 的 Harmonic 编码.

动机:
  DreaMS/Transformer 只在峰内部建模 (每个峰相对前体的质量差 δ 没有显式喂给模型)。
  这里把每个峰的中性丢失 δ 用「化学基频谐波」sin(2π·δ/Δ_k) 编码,
  mean-pool 到全局向量后 concat 进分类头 —— 与任何峰编码器正交的额外信息。

与 model_dreaMS.py 的关系:
  - 复用 DreamsMSEncoder (预训练 backbone + proj → MS_PE_DIM) 与 EnergyGate
  - 分层/扁平分类头逻辑保持一致 (与 DreaMS_016/017 相同), 仅把分类头输入扩为
    MS_PE_DIM(谱图) + HARMONIC_EMB_DIM(中性丢失) = 2*MS_PE_DIM

参考: MSEcoding/model.py HarmonicSeriesEncoder + 36 个化学中性丢失基频。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from model import EnergyGate
from model_dreaMS import (
    DreaMSAdductClassifier,
    DreamsMSEncoder,
)
from fusion import FusionBlock


# ════════════════════════════════════════════════════════════════════
# Harmonic Series Encoder (δ = precursor_mz − m/z)
# ════════════════════════════════════════════════════════════════════

class HarmonicSeriesEncoder(nn.Module):
    """
    用化学基频 Δ_k 编码中性丢失 δ。
    sin(2π·δ/Δ_k) 隐式做 δ mod Δ_k; 每基频再叠 2^h 谐波提高分辨率。
    """

    def __init__(self, out_dim: int, n_harmonics_per_base: int = 3):
        super().__init__()
        self.base_frequencies = config.HARMONIC_BASE_FREQUENCIES
        self.n_harmonics = n_harmonics_per_base
        self.n_bases = len(self.base_frequencies)
        d_raw = self.n_bases * n_harmonics_per_base * 2

        self.harmonic_proj = nn.Sequential(
            nn.LayerNorm(d_raw),          # 稳定原始谐波特征尺度
            nn.Linear(d_raw, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
        # 初始缩放 < 1: 避免 harmonic 一开始压过谱图特征
        self.output_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        """delta: (B, N) → (B, N, out_dim). 固定 fp32 计算 (AMP 安全)。"""
        from torch import autocast
        device_type = 'cuda' if delta.is_cuda else 'cpu'
        with autocast(device_type=device_type, enabled=False):
            delta = delta.float().clamp(0.0, 2000.0).unsqueeze(-1)
            components = []
            for dk in self.base_frequencies:
                for h in range(self.n_harmonics):
                    wavelength = dk / (2 ** h)
                    freq = 2 * math.pi / wavelength
                    components.append(torch.sin(freq * delta))
                    components.append(torch.cos(freq * delta))
            out = self.harmonic_proj(torch.cat(components, dim=-1))
            out = out * self.output_scale
        return out


# ════════════════════════════════════════════════════════════════════
# DreaMS + Harmonic 分类器
# ════════════════════════════════════════════════════════════════════

class DreaMSAdductHarmonicClassifier(DreaMSAdductClassifier):
    """
    与 DreaMSAdductClassifier 相同的接口, 额外 concat 全局中性丢失特征。
    不调用父类 __init__ (分类头输入维不同); 复用父类的模式映射与标签工具。
    """

    def __init__(self, ckpt_path=None, freeze_backbone=True):
        nn.Module.__init__(self)          # 绕过 DreaMSAdductClassifier.__init__

        self.ms_encoder = DreamsMSEncoder(
            ckpt_path=ckpt_path,
            output_dim=config.MS_PE_DIM,
            freeze_backbone=freeze_backbone,
        )
        self.ce_gate = EnergyGate(
            num_ce_bins=getattr(config, 'CE_NUM_BINS', 64),
            embed_dim=getattr(config, 'CE_EMBED_DIM', 32),
            output_dim=config.MS_PE_DIM,
            gate_type=getattr(config, 'CE_GATE_TYPE', 'sigmoid'),
        )
        self.harmonic_enc = HarmonicSeriesEncoder(
            out_dim=getattr(config, 'HARMONIC_EMB_DIM', config.MS_PE_DIM),
            n_harmonics_per_base=getattr(config, 'N_HARMONICS_PER_BASE', 3),
        )
        fusion = getattr(config, 'FUSION', 'concat')
        # 旧 MLP/harmonic 路径没有第三路 CE token, 三方融合在这里不适用;
        # 保持其历史默认 (bilinear), 不随 TSfM 的最优默认值变化。
        if fusion.startswith('tri_'):
            fusion = 'bilinear'
        self.fusion = FusionBlock(config.MS_PE_DIM, fusion)

        # 融合后送入分类头的维度 (concat=512, hadamard=768, gated/ffn=256)
        feat_dim = self.fusion.out_dim
        dropout = config.DROPOUT_RATE

        use_hierarchical = getattr(config, 'HIERARCHICAL_HEADS', False)
        if use_hierarchical:
            self.ion_mode_head = nn.Linear(feat_dim, 2)
            self.positive_head = nn.Sequential(
                nn.Linear(feat_dim, 128), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(128, len(self.POS_GLOB)),
            )
            self.negative_head = nn.Sequential(
                nn.Linear(feat_dim, 128), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(128, len(self.NEG_GLOB)),
            )
            self.classifier = None
            self._last_ion_logits = None
        else:
            self.classifier = nn.Sequential(
                nn.Linear(feat_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, config.NUM_ADDUCT_CLASSES),
            )
            self.ion_mode_head = None
            self.positive_head = None
            self.negative_head = None

    def _global_neutral_loss_feat(self, mz, intensity, precursor_mz):
        """(B,N) 峰 + precursor → (B, HARMONIC_EMB_DIM) mean-pooled δ 特征。"""
        mask = mz > 0.0
        if precursor_mz is None:
            precursor_mz = mz.max(dim=-1)[0]
        else:
            precursor_mz = precursor_mz.squeeze(-1)
            # 个别样本 precursor_mz 缺失(≤0)时回退到谱内最大峰
            fallback = mz.max(dim=-1)[0]
            precursor_mz = torch.where(precursor_mz > 0, precursor_mz, fallback)

        delta = precursor_mz.unsqueeze(-1) - mz            # (B, N)
        h = self.harmonic_enc(delta)                        # (B, N, H)
        h = h * mask.unsqueeze(-1).float()
        h_count = mask.sum(dim=-1, keepdim=True).clamp(min=1.0).float()
        return (h.sum(dim=1) / h_count).to(delta.dtype)     # (B, H)

    def forward(self, mz, intensity, ce=None, precursor_mz=None, precomputed_emb=None):
        feat = self.ms_encoder(mz, intensity,
                               precursor_mz=precursor_mz,
                               precomputed_emb=precomputed_emb)
        gate = None
        if ce is not None:
            gate = self.ce_gate(ce)
            feat = feat * gate

        if mz is not None:
            h_global = self._global_neutral_loss_feat(mz, intensity, precursor_mz)
            if gate is not None and getattr(config, 'HARMONIC_CE_GATE', False):
                h_global = h_global * gate.float()
        else:
            # cache / precomputed 模式没有原始峰 → 零特征占位
            h_global = torch.zeros_like(feat)
        feat = self.fusion(feat, h_global)   # (B, fusion.out_dim)

        if self.classifier is not None:
            return self.classifier(feat)

        # ── Hierarchical forward (与 model_dreaMS.py 一致) ──
        ion_logits = self.ion_mode_head(feat)               # (B, 2)
        pos_logits = self.positive_head(feat)               # (B, 5)
        neg_logits = self.negative_head(feat)               # (B, 4)

        batch = feat.shape[0]
        # 头输出在 AMP 下可能 fp16 (fusion=ffn 的 LN 会回 fp32); 缓冲 dtype 跟随头输出
        logits = torch.full((batch, config.NUM_ADDUCT_CLASSES),
                            float('-inf'), device=pos_logits.device,
                            dtype=pos_logits.dtype)
        logits[:, self.POS_GLOB] = pos_logits
        logits[:, self.NEG_GLOB] = neg_logits

        self._last_ion_logits = ion_logits
        return logits


if __name__ == "__main__":
    print("=" * 60)
    print("Testing DreaMSAdductHarmonicClassifier forward/backward")
    print("=" * 60)
    import os
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = os.path.join(config.DREAMS_CKPT_DIR, "ssl_model.ckpt")
    model = DreaMSAdductHarmonicClassifier(
        ckpt_path=ckpt, freeze_backbone=True).to(device)
    model.train()
    B, N = 4, 100
    mz = torch.zeros(B, N, device=device)
    mz[:, :40] = torch.rand(B, 40, device=device) * 900 + 50
    intensity = torch.zeros(B, N, device=device)
    intensity[:, :40] = torch.rand(B, 40, device=device)
    precursor_mz = torch.tensor([500.0, 300.0, 800.0, 200.0], device=device)
    ce = torch.full((B, 1), 40.0, device=device)
    logits = model(mz, intensity, ce=ce, precursor_mz=precursor_mz)
    loss = logits.sum()
    loss.backward()
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_hm = sum(p.numel() for p in model.harmonic_enc.parameters())
    print(f"logits={tuple(logits.shape)}  trainable={n_tr/1e6:.2f}M  "
          f"harmonic_params={n_hm/1e3:.0f}K")
    print("Test PASSED!")
