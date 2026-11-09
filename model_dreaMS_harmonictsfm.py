"""
model_dreaMS_harmonictsfm.py
DreaMS MS/MS encoder + δ(中性丢失) 分支用 **Transformer** 编码 + 可搜索的融合方式。

与 model_dreaMS_harmonic.py (δ 走 MLP + mean-pool) 的区别:
  - δ 分支: 逐峰 raw 谐波特征(210) → 小 Transformer(自注意力跨峰建模中性丢失共现) → [CLS] 256
  - 融合: config.FUSION ∈ {concat, hadamard, gated, ffn} 自动枚举选优
    concat    [d; h]                      (2*dim, 基线同 harmonic-MLP)
    hadamard  [d; h; d⊙h]                 (3*dim, 显式逐维交互)
    gated     g=σ(W[d;h]);  g⊙d+(1-g)⊙h   (dim, 软路由)
    ffn       d + h + FFN([d;h]) 后 LN     (dim, 非线性残差融合)
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
    """从已训练权重判断节点 δ 分支的架构。

    有 `harmonic_enc.fourier_enc.b` 即 "fourier"; 否则 "bases"。
    用于默认值切到 "fourier" 后, 旧 ckpt 的加载路径自动回到 bases 架构。
    """
    return "fourier" if any("harmonic_enc.fourier_enc.b" in k for k in state_dict) else "bases"


# ════════════════════════════════════════════════════════════════════
# Raw harmonic components of δ (shared primitive, computed in fp32)
# ════════════════════════════════════════════════════════════════════

def raw_harmonic_components(delta, bases=None, n_harm=None):
    """delta: (...,) 或 (B,N) 或 (B,N,N)
    bases/n_harm 为 None 时沿用 config 全局量 (与旧行为逐位一致)。"""
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
    """频率范围匹配的无化学含义基频: 每个真基频 ±0.5 Da 抖动, 保持 0.05 Da 间隔。"""
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
            raise RuntimeError(f"无法为基频 {b} 找到 sham 值")
    return sorted(out)


# ════════════════════════════════════════════════════════════════════
# δ Transformer encoder
# ════════════════════════════════════════════════════════════════════

class HarmonicTsfmEncoder(nn.Module):
    """
    逐峰 δ 谐波特征 → input_proj → 自注意力 Transformer → [CLS] 汇聚。
    固定 fp32 计算 (AMP 安全)。
    """

    def __init__(self, dim_out: int = 256):
        super().__init__()
        n_bases = len(config.HARMONIC_BASE_FREQUENCIES)
        n_harm = getattr(config, 'N_HARMONICS_PER_BASE', 3)
        self.delta_encoder_mode = getattr(config, 'DELTA_ENCODER', 'bases')
        if self.delta_encoder_mode not in ('bases', 'fourier'):
            raise ValueError(
                "DELTA_ENCODER 只支持 'bases' 或 'fourier', "
                f"收到 {self.delta_encoder_mode!r}")
        # δ token 的定义 (2026-10-20):
        #   "peak": 每个峰一个 token, δ_i = precursor_mz − mz_i (原行为)
        #   "pair": 每个峰对一个 token, pd_ij = |δ_i − δ_j| = |mz_i − mz_j|
        #   "peak_top5": 全部峰 δ + 强度前 K 个峰锚定的峰间差
        # 峰对模式逐样本前向 (对数随峰数变化, 最多 C(100,2)=4,950), 避免 padding
        # 到批内最大对数带来的显存浪费; 参数与 "peak" 完全相同, ckpt 可 strict 互载。
        self.delta_token_mode = getattr(config, 'DELTA_TOKEN_MODE', 'peak')
        if self.delta_token_mode not in ('peak', 'pair', 'peak_top5'):
            raise ValueError(
                "DELTA_TOKEN_MODE 只支持 'peak'/'pair'/'peak_top5', "
                f"收到 {self.delta_token_mode!r}")
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
                "PAIRDIFF_ENABLED 与 DELTA_TOKEN_MODE='pair' 互斥: 峰对已经是 token 本身, "
                "再加峰对 attention 边偏置属于重复编码")
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
            # 零初始化: 训练起点与关闭时函数等价; 只有有用时梯度才让它偏离
            nn.init.zeros_(self.pair_proj[-1].weight)
            nn.init.zeros_(self.pair_proj[-1].bias)

    def forward(self, delta: torch.Tensor, mask: torch.Tensor,
                intensity: torch.Tensor = None) -> torch.Tensor:
        """delta:(B,N) mask: valid peak mask(B,N) intensity:(B,N) → (B, dim_out) fp32

        intensity 只在 DELTA_TOKEN_MODE='peak_top5' 时用于挑选锚点峰。
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
                # FourierFeatures 需要最后一维为 1 (与 dreaMS 的 spec[..., [0]] 一致)
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
                # 用 clamp 前的 raw δ: |δi-δj| 严格等于 |mzi-mzj| (含峰高于前体的情形)
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
        """峰对 token: 每个有效峰对一个 token, 编码 |δ_i − δ_j|。

        逐样本单独前向: 对数 C(k,2) 随样本峰数变化, padding 到批内最大对数
        (最多 4,950) 会同时放大显存与无用的注意力计算。所有峰对都是有效
        key, 因此不需要 padding mask。
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
            # 大谱最多 4,950 个 token, 4 头注意力分数约 0.4 GB/样本; checkpoint
            # 只保留各层输入, 反向时重算, 避免一个 batch 里多个大谱把显存打爆
            out = checkpoint(self.encoder, tok, use_reentrant=False) \
                if self.training else self.encoder(tok)
            outs.append(self.out(out[:, 0, :]) * self.output_scale)
        return torch.cat(outs, dim=0)

    def _forward_anchor(self, raw: torch.Tensor, mask: torch.Tensor,
                        intensity: torch.Tensor) -> torch.Tensor:
        """前体中性丢失 + 强度前 K 峰锚定的峰间中性丢失。

        token 集合 = 全部有效峰的 δ_i (与 'peak' 模式相同)
                   + 对每个锚点峰 a 与每个 **m/z 小于 a** 的峰 j 的 |δ_a − δ_j|。
        锚点按 intensity 取前 K 个有效峰 (K = DELTA_ANCHOR_K); 单向配对
        (mz_j < mz_a ⇔ δ_j > δ_a), 因此 k=100 时峰间 token 最多约 K×k/2。
        与 'pair' 相比只保留 K 个强峰向下的中性丢失, 而不是 C(k,2)=4,950。

        批内 padding 到最大 token 数后一次前向 (最多 596), 而不是逐样本循环:
        逐样本会在 batch=26 时产生大量小 kernel, 实测 3.7× 于原峰模式。
        """
        if intensity is None:
            raise ValueError(
                "DELTA_TOKEN_MODE='peak_top5' 需要 intensity 来选择锚点峰")
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
                # 只与比锚点 m/z 更小的峰配对: mz_j < mz_a ⇔ δ_j > δ_a。
                # 严格大于同时排除 j=a 与 m/z 完全相同的重复峰。
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
# Fusion blocks 见 fusion.py (FusionBlock, 两种编码器共用)
# ════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════
# Classifier: DreaMS + δ-Transformer + fusion
# ════════════════════════════════════════════════════════════════════

class DreaMSAdductHarmonicTsfmClassifier(DreaMSAdductClassifier):
    """接口与 DreaMSAdductHarmonicClassifier 一致; 仅 δ 编码方式与融合不同。"""

    # 屏蔽位的常数: 用 -1e4 而非 -inf, 理由与两处既有实现一致 ——
    #   (1) model_dreaMS.MultiheadAttention 的 attention mask 同样取 -1e4 (fp16 下
    #       -1e9 → -inf 会导致softmax出 NaN);
    #   (2) sklearn.metrics.top_k_accuracy_score 会对分数矩阵做全有限性检查, 传入
    #       -inf 直接抛 ValueError —— 而那正是 _print_test_results 的指标之一。
    # 数值上等价: 真实 logits 量级在 ±100 内, softmax 里 exp(-1e4-max) 在 fp32/fp16
    # 下都下溢为**精确的 0**, 故该类的概率与梯度均为 0, argmax 也永远不会选中它。
    MASKED_LOGIT = -1e4

    def __init__(self, ckpt_path=None, freeze_backbone=True):
        nn.Module.__init__(self)
        self.ms_encoder = DreamsMSEncoder(
            ckpt_path=ckpt_path,
            output_dim=config.MS_PE_DIM,
            freeze_backbone=freeze_backbone,
        )
        # 离子模式 (正/负) 在采集时已知: 见 config.py 的说明块。
        #   pol_in   — 极性嵌入与 CE 嵌入拼接后一起进门控
        #   pol_mask — 用给定极性把反极性类屏蔽掉 (不再让极性参与 argmax 竞争)
        pol_in = bool(getattr(config, 'POLARITY_AS_INPUT', False))
        pol_mask = bool(getattr(config, 'POLARITY_MASK_OUTPUT', False))
        self.mask_output_by_polarity = pol_mask
        # 热身: 前 mask_start_epoch-1 轮不施加训练期屏蔽 (0 = 全程屏蔽)。
        # 动机见 config.POLARITY_MASK_START_EPOCH 的说明; 验证/评测不受影响。
        self.mask_start_epoch = int(getattr(config, 'POLARITY_MASK_START_EPOCH', 0))
        self.mask_inference_only = bool(
            getattr(config, 'POLARITY_MASK_INFERENCE_ONLY', False))
        self.current_epoch = 0
        # 训练/评测循环据此决定是否把 batch['polarity'] 传进来
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
            # 三方融合要求 CE 向量与 d/δ 同维; 其余模式用较小的 aux
            aux_dim=config.MS_PE_DIM if use_tri else getattr(config, 'CE_AUX_DIM', 64),
            **pol_kw,
        )
        # CE 只作用于 δ 分支的模式 (gate_delta / film_delta): d 分支不被 CE 触碰
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
            # 极性不需要预测 (采集时已知): ION_AUX_ENABLED=False 时不构造该头,
            # 训练循环的 _hierarchical() 随即跳过辅助 loss。两个同极性内的分类头保留。
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
        """训练循环每轮调用一次; 只用于 POLARITY_MASK_START_EPOCH 的预热判断。"""
        self.current_epoch = int(epoch)

    def _apply_polarity_mask(self, logits, polarity):
        """按**给定**的离子模式把九类 logits 里反极性的类屏蔽掉。

        注意这与"用**预测**出的极性去门控"完全不同 —— 后者实测零影响
        (analyze_polarity.py: 1630 个跨极性错误里 96.3% 极性头自己也判错)。
        这里用的是采集时已知的真值极性, 因此等价于该脚本里的 oracle 极性上界。

        训练期可延迟生效 (mask_start_epoch): 屏蔽会让反极性 logits 的梯度恒为 0,
        预热期先让九类自由竞争, 避免掐掉极性输入唯一有效的学习信号。
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
            # CE token 消融 (config.CE_TOKEN_ENABLED=False): 模块与参数保留, 但不把
            # CE 描述向量送进融合。tri_xattn 的 KV 因此从 [δ, CE] 缩成 [δ]。
            if not getattr(config, 'CE_TOKEN_ENABLED', True):
                aux = None

        if mz is not None:
            p = self._precursor(mz, precursor_mz)
            delta = p.unsqueeze(-1) - mz                      # (B,N)
            mask = mz > 0.0
            h = self.harmonic_enc(delta, mask, intensity)     # (B,dim) fp32
            if self.ce_adapter_delta is not None and ce is not None:
                h, _ = self.ce_adapter_delta.apply(h, ce, polarity)   # CE 只碰 δ
        else:
            h = torch.zeros_like(feat)                        # cache 模式无 δ
        fused = self.fusion(feat, h, aux if self.use_tri else None)
        if aux is not None and not self.use_tri:
            fused = torch.cat([fused, aux.to(dtype=fused.dtype)], dim=-1)

        if self.classifier is not None:
            return self._apply_polarity_mask(self.classifier(fused), polarity)

        if self.ion_mode_head is not None:
            self._last_ion_logits = self.ion_mode_head(fused)
        pos_logits = self.positive_head(fused)
        neg_logits = self.negative_head(fused)
        # 头输出在 AMP 下可能是 fp16; logits 缓冲 dtype 跟随头输出, 避免 Index put dtype 不匹配
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
