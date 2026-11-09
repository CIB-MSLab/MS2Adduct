import math
import torch
import torch.nn as nn
import config


class FourierEmbedder(nn.Module):
    def __init__(self, spec_embed_dim=config.MS_PE_DIM, logmin=2.5, logmax=3.3):
        super().__init__()
        self.d = spec_embed_dim
        lambda_min = 10 ** (-logmin)
        lambda_max = 10 ** logmax
        index = torch.arange(math.ceil(self.d / 2)).float()
        exp_term = torch.pow(lambda_max / lambda_min, (2 * index) / (self.d - 2))
        freqs = 2 * math.pi * ((lambda_min * exp_term) ** (-1))
        self.register_buffer('freqs', freqs)
        self.ffn = nn.Sequential(
            nn.Linear(self.d + 1, self.d),
            nn.GELU(),
            nn.Linear(self.d, self.d)
        )

    def forward(self, mz, intensity):
        freq_input = mz.unsqueeze(-1) * self.freqs.unsqueeze(0).unsqueeze(0)
        embedded = torch.cat([torch.sin(freq_input), torch.cos(freq_input)], dim=-1)
        embedded = embedded[:, :, :self.d]
        embedded = embedded.type_as(intensity)
        x = torch.cat([embedded, intensity.unsqueeze(-1)], dim=-1)
        return self.ffn(x)


class ESAModule(nn.Module):
    def __init__(self, feature_dim=config.MS_PE_DIM, out_dim=config.MS_PE_DIM):
        super().__init__()
        self.ln_f = nn.LayerNorm(feature_dim)
        self.linear = nn.Linear(feature_dim, out_dim)
        self.linear1 = nn.Linear(out_dim, out_dim)

    def forward(self, hidden_states, padding_mask=None):
        logits = self.ln_f(hidden_states)
        cap_embes = self.linear(logits)
        features_in = self.linear1(cap_embes)

        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(-1).expand_as(features_in)
            features_in = features_in.masked_fill(padding_mask, 0.0)

        return torch.sum(features_in, dim=1)


class CSU_MSMSEncoder(nn.Module):
    def __init__(self, pe_dim=config.MS_PE_DIM, output_dim=config.MS_PE_DIM):
        super().__init__()
        self.pe_encoder = FourierEmbedder(spec_embed_dim=pe_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=pe_dim,
            nhead=config.MS_TRANSFORMER_HEADS,
            dim_feedforward=config.MS_TRANSFORMER_FFN_DIM,
            dropout=config.MS_DROPOUT,
            batch_first=True,
            activation="relu"
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.MS_TRANSFORMER_LAYERS
        )
        self.esa = ESAModule(feature_dim=pe_dim, out_dim=pe_dim)
        self.proj = nn.Linear(pe_dim, output_dim)

    def forward(self, mz, intensity):
        padding_mask = (mz == 0.0)
        pe_feat = self.pe_encoder(mz, intensity)
        out = self.transformer(pe_feat, src_key_padding_mask=padding_mask)
        spectrum_global_feat = self.esa(out, padding_mask=padding_mask)
        return self.proj(spectrum_global_feat)


class EnergyGate(nn.Module):
    def __init__(self, num_ce_bins=64, embed_dim=32, output_dim=config.MS_PE_DIM,
                 gate_type='sigmoid'):
        super().__init__()
        self.num_bins = num_ce_bins
        self.ce_embedding = nn.Embedding(num_ce_bins, embed_dim)
        self.gate_network = nn.Linear(embed_dim, output_dim)

        if gate_type == 'sigmoid':
            self.out_activation = nn.Sigmoid()
        elif gate_type == 'softmax':
            self.out_activation = nn.Softmax(dim=-1)
        elif gate_type == 'tanh':
            self.out_activation = nn.Tanh()
        else:
            self.out_activation = nn.Identity()

    def energy_to_bins(self, energy):
        energy = torch.clamp(energy, 0.0, 150.0)
        normalized_energy = energy / 150.0
        bin_index = torch.floor(normalized_energy * self.num_bins).long()
        bin_index = torch.where(
            bin_index >= self.num_bins,
            torch.tensor(self.num_bins - 1, device=energy.device),
            bin_index
        )
        return bin_index.squeeze(-1)

    def forward(self, energy):
        ce_bin_indices = self.energy_to_bins(energy)
        ce_emb = self.ce_embedding(ce_bin_indices)
        gate = self.out_activation(self.gate_network(ce_emb))
        return gate


class AdductClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.ms_encoder = CSU_MSMSEncoder()
        self.ce_gate = EnergyGate(
            num_ce_bins=getattr(config, 'CE_NUM_BINS', 64),
            embed_dim=getattr(config, 'CE_EMBED_DIM', 32),
            output_dim=config.MS_PE_DIM,
            gate_type=getattr(config, 'CE_GATE_TYPE', 'sigmoid'),
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.MS_PE_DIM, 128),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(128, config.NUM_ADDUCT_CLASSES),
        )

    def forward(self, mz, intensity, ce=None):
        feat = self.ms_encoder(mz, intensity)
        if ce is not None:
            gate = self.ce_gate(ce)
            feat = feat * gate
        return self.classifier(feat)
