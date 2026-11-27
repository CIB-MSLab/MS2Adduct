"""
model_dreaMS_harmonictsfm_2.py
DreaMSHT2 = 与 dreaMS 原型完全一致的 dreaMS 分支 + δ-Transformer 编码 + 融合 → 9 类加合物。

与 model_dreaMS_harmonictsfm.py 的唯一区别在 dreaMS 谱图编码器, 由 config 的
DREAMS2_* 开关控制 (默认即"完全一致"):

  DREAMS2_N_HEADS = 8        ssl_model.ckpt 是按 n_heads=8 预训练的。原移植版用 16,
                             而 qkv 权重形状 (4096,1024) 与头数无关, load_state_dict
                             (strict=False) 静默通过 —— head_dim 悄悄变成 64 而非 128。
  DREAMS2_GRAPHORMER = False 成对 m/z 差 attention bias: 预训练启用, 移植版丢掉。
                             **实测在 100 峰 padding 下有害、在预训练的 60 峰下有益**
                             (两个独立指标交互项 t≈3.2), 故默认关; 详见 config.py 的结论块。
  DREAMS2_FOURIER_STRAT      与 ckpt 一致的频率构造方式 (只影响 scratch 初始化)。

实现策略: 全部继承既有实现, 只覆盖真正需要改的方法。δ 分支 / CE adapter / FusionBlock /
分层头 / _prepare_spec / _load_weights 均原样复用 —— 本修复不引入任何新参数, 因此
backbone 的状态字典键与 V1 完全相同 (65 个), 权重加载路径无需改动。

自检 / 回归: verify_dreaMS_fidelity.py (含与 ckpt 记录值的逐项对账)
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
    """graphormer 距离张量, 末维留给 attention 内部的 .sum(-1) (对应 layers.py:94)。

    原版 (dreams.py):
        graphormer_dists = fourier_features.unsqueeze(2) - fourier_features.unsqueeze(1)
    注意 d_fourier 分支下传入的是 ff_fourier 的输出, 即 980 维投影后的特征差,
    不是原始 m/z 差。
    """
    if GD_EXACT:
        return fourier_features.unsqueeze(2) - fourier_features.unsqueeze(1)  # (B,N,N,d_fourier)
    s = fourier_features.sum(dim=-1)                                          # (B,N)
    return (s.unsqueeze(2) - s.unsqueeze(1)).unsqueeze(-1)                    # (B,N,N,1)


class MultiheadAttentionFid(MultiheadAttention):
    """加回 graphormer bias; 其余逐行与 model_dreaMS.MultiheadAttention 一致。"""

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
    """与父类逐行一致, 区别有二:

      (1) __init__ 必须重建 atts 为 MultiheadAttentionFid。父类的 __init__ 构造的是
          **普通 MultiheadAttention**, 其 forward 忽略 graphormer_dists —— 只看 forward
          会以为已经支持, 实际 gd 被每层静默丢弃 (verify_dreaMS_fidelity.py 第 [7] 项
          就是靠这一点抓出问题的: 输出等价于"关了 gd")。
      (2) forward 把 graphormer_dists 透传给每层。

    参数名与父类完全相同 (MultiheadAttentionFid 继承自 MultiheadAttention), 所以
    状态字典键不变, _load_weights 的 65 键过滤无需改动。
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
    """默认取 ckpt 的预训练配置, 并把 encoder 换成会传递 graphormer_dists 的版本。"""

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
        """与父类唯一区别: 额外算出 graphormer 距离并传给 encoder。"""
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
    """只把 backbone 换成 Fid 版; 权重加载 / 预处理 / 前向投影全部继承。"""

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
    """接口与 DreaMSAdductHarmonicTsfmClassifier 完全一致, 只换谱图编码器。

    δ 分支 / CE adapter / FusionBlock / 分层头 / forward 全部继承, 因此 DreaMSHT2
    与 DreaMSHT 的差异严格只有 dreaMS 分支的忠实度这一项 (单变量)。
    """

    def __init__(self, ckpt_path=None, freeze_backbone=True):
        print(f"[DreaMSHT2] 忠实版谱图编码器: n_heads={getattr(config, 'DREAMS2_N_HEADS', 8)}, "
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
    print("DreaMSHT2: 构造 + 权重加载 + 前向自检")
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
    print(f"logits: {tuple(logits.shape)}  有限值={bool(torch.isfinite(logits).all())}")
    print("=" * 70)
    print("自检 PASSED (完整回归请跑 verify_dreaMS_fidelity.py)")
