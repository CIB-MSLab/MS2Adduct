# Model self-checks

Standalone scripts that verify the model implementation without training.

| Script | Checks |
|---|---|
| `verify_fourier.py` | Fourier delta encoder: frequency construction, shapes, determinism |
| `verify_polarity_input.py` | polarity-as-input and inference-time masking behaviour |
| `verify_pairdiff.py` | pairwise neutral-loss difference branch (disabled by default) |
| `verify_dreaMS_fidelity.py` | DreaMSHT2 backbone against the original dreaMS checkpoint configuration |

Run from the repository root:

```bash
python analysis/verify_fourier.py
python analysis/verify_polarity_input.py
python analysis/verify_pairdiff.py
python analysis/verify_dreaMS_fidelity.py
```

`verify_dreaMS_fidelity.py` compares the constructed backbone with recorded
values from `ssl_model.ckpt` and therefore needs
`dreaMS/ssl_model.ckpt` to be present.
