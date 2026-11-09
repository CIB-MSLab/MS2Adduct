"""
model_dreaMS.py
DreaMS MS/MS encoder replacing CSU_MSMSEncoder.
Supports 3 modes: cache (precomputed embeddings), finetune (pretrained + trainable), scratch (random init).
"""
import io, os, re, sys, math, pickle, zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import pathlib, platform
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
import config


class FourierFeatures(nn.Module):
    def __init__(self, strategy='voronov_et_al', x_min=1e-4, x_max=1000.0,
                 trainable=False, funcs='both', num_freqs=5997):
        super().__init__()
        self.funcs = funcs
        self.trainable = trainable
        if strategy == 'voronov_et_al':
            self.b = torch.tensor(
                [1 / (x_min * (x_max / x_min) ** (2 * i / (num_freqs - 2)))
                 for i in range(1, num_freqs)],
            )
        elif strategy == 'lin_float_int':
            self.b = torch.tensor(
                [1 / (x_min * i) for i in range(2, math.ceil(1 / x_min), 2)] +
                [1 / (1 * i) for i in range(2, math.ceil(x_max), 1)],
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        self.b = self.b.unsqueeze(0)
        self.b = nn.Parameter(self.b, requires_grad=self.trainable)
        self.register_parameter('Fourier frequencies', self.b)

    def forward(self, x):
        x = 2 * math.pi * x @ self.b
        if self.funcs == 'both':
            x = torch.cat((torch.cos(x), torch.sin(x)), dim=-1)
        return x

    def num_features(self):
        return self.b.shape[1] if self.funcs != 'both' else 2 * self.b.shape[1]


class FeedForward(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, depth=None, act_last=True,
                 act=nn.ReLU, bias=True, dropout=0.0):
        super().__init__()
        if isinstance(hidden_dim, int):
            assert depth is not None
            hidden_dim = [hidden_dim] * depth
        else:
            depth = len(hidden_dim)
        layers = []
        for l in range(depth):
            d1 = hidden_dim[l - 1] if l != 0 else in_dim
            d2 = hidden_dim[l] if l != depth - 1 else out_dim
            layers.append(nn.Linear(d1, d2, bias=bias))
            if l != depth - 1:
                layers.append(nn.Dropout(p=dropout))
            if l != depth - 1 or act_last:
                layers.append(act())
        self.ff = nn.Sequential(*layers)

    def forward(self, x):
        return self.ff(x)


class ScaleNorm(nn.Module):
    def __init__(self, scale, eps=1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm


class MultiheadAttention(nn.Module):
    def __init__(self, d_model=1024, n_heads=16, dropout=0.0,
                 use_transformer_bias=False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.weights = nn.Parameter(torch.Tensor(4 * d_model, d_model))
        if use_transformer_bias:
            self.biases = nn.Parameter(torch.Tensor(4 * d_model))
        mean, std = 0, (2 / (5 * d_model)) ** 0.5
        nn.init.normal_(self.weights, mean=mean, std=std)
        if use_transformer_bias:
            nn.init.constant_(self.biases, 0.)

    def _proj(self, x, start=0, end=None):
        weight = self.weights[start:end, :]
        bias = None if not hasattr(self, 'biases') else self.biases[start:end]
        return F.linear(x, weight=weight, bias=bias)

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
        if mask is not None:
            att_weights.masked_fill_(mask.unsqueeze(1).unsqueeze(-1), -1e4)
        att_weights = F.softmax(att_weights, dim=-1)
        att_weights = F.dropout(att_weights, p=self.dropout, training=self.training)
        _att_weights = att_weights.reshape(-1, n, n)
        output = torch.bmm(_att_weights, v.reshape(bs * self.n_heads, -1, self.head_dim))
        output = output.reshape(bs, self.n_heads, n, self.head_dim).transpose(1, 2).reshape(bs, n, -1)
        output = self._proj(output, start=3 * self.d_model)
        return output, att_weights


class TransformerFFN(nn.Module):
    def __init__(self, d_model=1024, dropout=0.0, use_transformer_bias=False):
        super().__init__()
        self.ff_dim = 4 * d_model
        self.in_proj = nn.Linear(d_model, self.ff_dim, bias=use_transformer_bias)
        self.out_proj = nn.Linear(self.ff_dim, d_model, bias=use_transformer_bias)
        mean, std = 0, (2 / (self.ff_dim + d_model)) ** 0.5
        nn.init.normal_(self.in_proj.weight, mean=mean, std=std)
        nn.init.normal_(self.out_proj.weight, mean=mean, std=std)
        if use_transformer_bias:
            nn.init.constant_(self.in_proj.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        self.dropout = dropout

    def forward(self, x):
        y = F.relu(self.in_proj(x))
        y = F.dropout(y, p=self.dropout, training=self.training)
        return self.out_proj(y)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model=1024, n_heads=16, n_layers=7, use_layernorm=True,
                 pre_norm=True, residual_dropout=0.0, att_dropout=0.0,
                 ff_dropout=0.0, use_transformer_bias=False):
        super().__init__()
        self.residual_dropout = residual_dropout
        self.n_layers = n_layers
        self.pre_norm = pre_norm
        self.atts = nn.ModuleList([
            MultiheadAttention(d_model=d_model, n_heads=n_heads, dropout=att_dropout,
                               use_transformer_bias=use_transformer_bias)
            for _ in range(n_layers)
        ])
        self.ffs = nn.ModuleList([
            TransformerFFN(d_model=d_model, dropout=ff_dropout,
                           use_transformer_bias=use_transformer_bias)
            for _ in range(n_layers)
        ])
        num_scales = n_layers * 2 + 1 if pre_norm else n_layers * 2
        if use_layernorm:
            self.scales = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_scales)])
        else:
            self.scales = nn.ModuleList([ScaleNorm(d_model ** 0.5) for _ in range(num_scales)])

    def forward(self, src_inputs, src_mask, graphormer_dists=None):
        x = F.dropout(src_inputs, p=self.residual_dropout, training=self.training)
        for i in range(self.n_layers):
            residual = x
            if self.pre_norm:
                x = self.scales[2 * i](x)
            x, _ = self.atts[i](q=x, k=x, v=x, mask=src_mask)
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


class DreaMSBackbone(nn.Module):
    def __init__(self, d_fourier=980, d_peak=44, d_mz_token=0, d_model=1024,
                 n_heads=16, n_layers=7, charge_feature=False,
                 use_layernorm=True, pre_norm=True,
                 dropout=0.0, att_dropout=0.0, ff_dropout=0.0,
                 residual_dropout=0.0,
                 fourier_strategy='voronov_et_al', fourier_num_freqs=5998,
                 fourier_trainable=False, max_mz=1000.0, max_tbxic_stdev=1e-4,
                 ff_peak_depth=1, ff_fourier_depth=5, ff_fourier_d=512,
                 use_transformer_bias=False, no_ffs_bias=False):
        super().__init__()
        self.d_fourier = d_fourier
        self.d_peak = d_peak
        self.d_mz_token = d_mz_token
        self.d_model = d_model
        self.n_layers = n_layers
        self.charge_feature = charge_feature
        self.max_mz = max_mz
        token_dim = 2
        if charge_feature:
            token_dim += 1
        if d_fourier:
            self.fourier_enc = FourierFeatures(
                strategy=fourier_strategy, num_freqs=fourier_num_freqs,
                x_min=max_tbxic_stdev, x_max=max_mz,
                trainable=fourier_trainable,
            )
            self.ff_fourier = FeedForward(
                in_dim=self.fourier_enc.num_features(), out_dim=d_fourier,
                dropout=dropout, depth=ff_fourier_depth,
                hidden_dim=ff_fourier_d, bias=not no_ffs_bias,
            )
        self.ff_peak = FeedForward(
            in_dim=token_dim, hidden_dim=d_peak, out_dim=d_peak,
            depth=ff_peak_depth, dropout=dropout, bias=not no_ffs_bias,
        )
        self.transformer_encoder = TransformerEncoder(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            use_layernorm=use_layernorm, pre_norm=pre_norm,
            residual_dropout=residual_dropout, att_dropout=att_dropout,
            ff_dropout=ff_dropout, use_transformer_bias=use_transformer_bias,
        )

    def forward(self, spec, charge=None):
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
        if self.d_fourier:
            with autocast(enabled=False):
                fourier_features = self.ff_fourier(self.fourier_enc(spec[..., [0]].float()))
            fourier_features = fourier_features.to(dtype=peak_embs.dtype)
            spec_out = torch.cat([peak_embs, fourier_features], dim=-1)
        else:
            spec_out = peak_embs
        spec_out = self.transformer_encoder(spec_out, padding_mask)
        return spec_out


def _load_state_dict_from_lightning_ckpt(ckpt_path):
    _dreams_root = os.path.dirname(os.path.abspath(__file__))
    if _dreams_root not in sys.path:
        sys.path.insert(0, _dreams_root)
    print(f"[Loader] Loading checkpoint...")
    with zipfile.ZipFile(ckpt_path, 'r') as zf:
        data_bytes = zf.read('archive/data.pkl')
        _storage_cache = {}
        def persistent_load(pid):
            if not isinstance(pid, tuple) or len(pid) < 5:
                raise pickle.UnpicklingError(f"Unsupported persistent id: {pid}")
            storage_key = pid[2]
            num_elements = pid[4]
            storage_cls = pid[1]
            cache_key = str(storage_cls)
            if cache_key not in _storage_cache:
                s0 = storage_cls(0)
                _storage_cache[cache_key] = (s0.element_size(), s0.dtype)
            el_size, dtype = _storage_cache[cache_key]
            data_path = f'archive/data/{storage_key}'
            raw = zf.read(data_path)
            nbytes = num_elements * el_size
            t = torch.frombuffer(bytearray(raw[:nbytes]), dtype=dtype)
            return torch.storage.TypedStorage(
                wrap_storage=t.untyped_storage(), dtype=dtype)

        class StubClass:
            def __init__(self, *args, **kwargs): pass
            def __setstate__(self, state): self.__dict__.update(state)

        class MSMLRedirectUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module.startswith('msml') or module == 'argparse' or module == 'pathlib':
                    return StubClass
                if module == 'torch._tensor':
                    import torch._tensor; return getattr(torch._tensor, name)
                if module == 'torch._utils':
                    import torch._utils; return getattr(torch._utils, name)
                if module == 'collections' and name == 'OrderedDict':
                    from collections import OrderedDict; return OrderedDict
                try:
                    __import__(module)
                    return getattr(sys.modules[module], name)
                except (ImportError, AttributeError):
                    return StubClass

        unpickler = MSMLRedirectUnpickler(io.BytesIO(data_bytes))
        unpickler.persistent_load = persistent_load
        ckpt = unpickler.load()
    state_dict = ckpt.get('state_dict', ckpt)
    print(f"[Loader] Loaded state_dict ({len(state_dict)} params)")
    return state_dict


class DreamsMSEncoder(nn.Module):
    def __init__(self, ckpt_path=None, output_dim=None, freeze_backbone=True,
                 n_highest_peaks=None, d_model=1024):
        super().__init__()
        if output_dim is None:
            output_dim = getattr(config, 'MS_PE_DIM', 256)
        if ckpt_path is None:
            print("[DreamsMSEncoder] ckpt_path=None, using random initialization")
        if n_highest_peaks is None:
            n_highest_peaks = getattr(config, 'MAX_PEAKS', 100)
        self.n_highest_peaks = n_highest_peaks
        self.max_mz = 1000.0
        self.output_dim = output_dim
        self.backbone = DreaMSBackbone()
        if ckpt_path is not None and os.path.exists(ckpt_path):
            self._load_weights(ckpt_path)
        elif ckpt_path is not None:
            print(f"[DreamsMSEncoder] WARNING: {ckpt_path} not found, using random init")
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def _load_weights(self, ckpt_path):
        state_dict = _load_state_dict_from_lightning_ckpt(ckpt_path)
        backbone_sd = {}
        skip_prefixes = ['ff_out.', 'ro_out.', 'mz_masking_loss.',
                         'mz_tokenizer.', 'ff_mz_token.', 'spec_preproc.']
        for key, tensor in state_dict.items():
            if any(k in key for k in skip_prefixes):
                continue
            new_key = key
            for prefix in ['model.', 'backbone.']:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    break
            backbone_sd[new_key] = tensor
        missing, unexpected = self.backbone.load_state_dict(backbone_sd, strict=False)
        if missing:
            print(f"[DreamsMSEncoder] Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"[DreamsMSEncoder] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
        print(f"[DreamsMSEncoder] Loaded {len(backbone_sd) - len(unexpected)} backbone params")

    def _prepare_spec(self, mz, intensity, precursor_mz=None):
        batch_size = mz.shape[0]
        spec = torch.stack([mz, intensity], dim=-1)
        if self.n_highest_peaks and spec.shape[1] > self.n_highest_peaks:
            _, top_idx = intensity.topk(
                min(self.n_highest_peaks, spec.shape[1]), dim=-1)
            top_idx = top_idx.unsqueeze(-1).expand(-1, -1, 2)
            spec = spec.gather(1, top_idx)
        spec = spec[torch.arange(batch_size).unsqueeze(-1),
                    spec[..., 0].argsort(dim=-1)]
        max_intens = spec[..., 1].max(dim=-1, keepdim=True)[0].clamp(min=1e-8)
        spec[..., 1] = spec[..., 1] / max_intens
        if precursor_mz is None:
            precursor_mz = spec[..., 0].max(dim=-1)[0]
        prec_token = torch.stack([
            precursor_mz, torch.full_like(precursor_mz, 1.1)
        ], dim=-1).unsqueeze(1)
        spec = torch.cat([prec_token, spec], dim=1)
        return spec

    def forward(self, mz, intensity, precursor_mz=None, precomputed_emb=None):
        if precomputed_emb is not None and precomputed_emb.shape[-1] > 2:
            return self.proj(precomputed_emb.to(
                dtype=self.proj[0].weight.dtype, device=self.proj[0].weight.device))
        spec = self._prepare_spec(mz, intensity, precursor_mz)
        frozen = all(not p.requires_grad for p in self.backbone.parameters())
        with torch.set_grad_enabled(not frozen):
            if frozen:
                with torch.no_grad():
                    token_embs = self.backbone(spec)
            else:
                token_embs = self.backbone(spec)
        spec_emb = token_embs[:, 0, :]
        return self.proj(spec_emb)

    @torch.no_grad()
    def backbone_embedding(self, mz, intensity, precursor_mz=None):
        """Return raw 1024-dim precursor token embedding (before proj)."""
        spec = self._prepare_spec(mz, intensity, precursor_mz)
        token_embs = self.backbone(spec)
        return token_embs[:, 0, :]


from model import EnergyGate


class DreaMSAdductClassifier(nn.Module):
    # ── Index mappings for hierarchical classifier ──
    # Positive mode adducts:  [M+H]+,  [M+Na]+,  [M+NH4]+, [M+K]+, [M]+
    POS_GLOB = [0, 2, 3, 5, 8]       # local → global index
    # Negative mode adducts:  [M-H]-,  [M+HCOOH-H]-, [M+CH3COOH-H]-, [M+Cl]-
    NEG_GLOB = [1, 4, 6, 7]          # local → global index
    # global_index → ion_mode (0=positive, 1=negative)
    GLOB_TO_ION = [0, 1, 0, 0, 1, 0, 1, 1, 0]
    # global_index → local_index within each mode (-1 = not in that mode)
    GLOB_TO_LOC = [
        [0, -1, 1, 2, -1, 3, -1, -1, 4],   # positive mode local indices
        [-1, 0, -1, -1, 1, -1, 2, 3, -1],  # negative mode local indices
    ]

    def __init__(self, ckpt_path=None, freeze_backbone=True):
        super().__init__()
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

        use_hierarchical = getattr(config, 'HIERARCHICAL_HEADS', False)
        if use_hierarchical:
            embed_dim = config.MS_PE_DIM
            dropout = config.DROPOUT_RATE
            self.ion_mode_head = nn.Linear(embed_dim, 2)
            self.positive_head = nn.Sequential(
                nn.Linear(embed_dim, 128), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(128, len(self.POS_GLOB)),
            )
            self.negative_head = nn.Sequential(
                nn.Linear(embed_dim, 128), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(128, len(self.NEG_GLOB)),
            )
            self.classifier = None
            self._last_ion_logits = None
        else:
            self.classifier = nn.Sequential(
                nn.Linear(config.MS_PE_DIM, 128),
                nn.ReLU(),
                nn.Dropout(config.DROPOUT_RATE),
                nn.Linear(128, config.NUM_ADDUCT_CLASSES),
            )
            self.ion_mode_head = None
            self.positive_head = None
            self.negative_head = None

    def forward(self, mz, intensity, ce=None, precursor_mz=None, precomputed_emb=None):
        feat = self.ms_encoder(mz, intensity,
                               precursor_mz=precursor_mz,
                               precomputed_emb=precomputed_emb)
        if ce is not None:
            gate = self.ce_gate(ce)
            feat = feat * gate

        if self.classifier is not None:
            return self.classifier(feat)

        # ── Hierarchical forward ──
        ion_logits = self.ion_mode_head(feat)           # (B, 2)
        pos_logits = self.positive_head(feat)            # (B, 5)
        neg_logits = self.negative_head(feat)            # (B, 4)

        # Assemble 9-class logits
        batch = feat.shape[0]
        logits = torch.full((batch, config.NUM_ADDUCT_CLASSES),
                            float('-inf'), device=feat.device, dtype=feat.dtype)
        logits[:, self.POS_GLOB] = pos_logits
        logits[:, self.NEG_GLOB] = neg_logits

        self._last_ion_logits = ion_logits
        return logits

    def get_ion_logits(self):
        """Retrieve ion_mode logits from the last forward() call."""
        return self._last_ion_logits

    @staticmethod
    def ion_mode_labels(labels):
        """Convert global 0..8 labels → ion_mode labels (0=positive, 1=negative)."""
        ion = torch.zeros_like(labels)
        for i, mode in enumerate(DreaMSAdductClassifier.GLOB_TO_ION):
            ion[labels == i] = mode
        return ion


if __name__ == "__main__":
    print("=" * 60)
    print("Testing DreamsMSEncoder construction + weight loading")
    print("=" * 60)
    ckpt = os.path.join(config.DREAMS_CKPT_DIR, "ssl_model.ckpt")
    encoder = DreamsMSEncoder(ckpt_path=ckpt, output_dim=256, freeze_backbone=True)
    mz = torch.rand(2, 50) * 1000
    intensity = torch.rand(2, 50)
    mz[:, -10:] = 0; intensity[:, -10:] = 0
    encoder.eval()
    with torch.no_grad():
        emb = encoder(mz, intensity)
    print(f"Input: mz={mz.shape}, intensity={intensity.shape}")
    print(f"Output embedding: {emb.shape}")
    print(f"Output norm: {emb.norm(dim=-1)}")
    print("=" * 60)
    print("Test PASSED!")
