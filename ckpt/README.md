# Released checkpoints

The three final checkpoints are **not** stored in git (about 400 MB each).
They are distributed through the Zenodo deposit and can be installed with:

```bash
ZENODO_RECORD=<record-id> bash scripts/download_weights.sh
```

## Files

| Zenodo file | Installed as | Seed | Test Acc (free) | Test Macro F1 (free) |
|---|---|---:|---:|---:|
| `tx_no_ion_aux_s1.pth` | `ckpt/MS2ADDUCT_ABL_tx_no_ion_aux_s1_000_Acc_0.854_F1_0.773/best_model.pth` | 1 | 0.8545 | 0.7733 |
| `tx_no_ion_aux_s2.pth` | `ckpt/MS2ADDUCT_ABL_tx_no_ion_aux_s2_000_Acc_0.858_F1_0.773/best_model.pth` | 2 | 0.8583 | 0.7729 |
| `tx_no_ion_aux_s3.pth` | `ckpt/MS2ADDUCT_ABL_tx_no_ion_aux_s3_000_Acc_0.859_F1_0.781/best_model.pth` | 3 | 0.8588 | 0.7815 |

The same weights give the deployment protocol numbers when the acquisition ion
mode is used to mask opposite-polarity classes at inference time (0.8844 /
0.7909 averaged over the three seeds).

## SHA-256

```
3e6e6e61cebe6d7456b829f2328876fbbb0164cba168cbf9d3f6f86f53ce5f08  tx_no_ion_aux_s1.pth
8b26baf4fb80bcdece336236d0ace8304a33483510979707d56f10fe1397343e  tx_no_ion_aux_s2.pth
761733638f5b26586b496241c09896d3832feb592deacf70ae556d259b07f93b  tx_no_ion_aux_s3.pth
```

## Run metadata in git

Each run directory already contains:

- `trainlog.txt` — the exact command, configuration header, per-epoch losses and
  the final test evaluation;
- `ablation_meta.json` — the effective config and provenance of this run.

Training writes checkpoints to `ckpt/<prefix>NNN_Acc_*_F1_*/` and also stores a
copy of the code used for that run under `codes/`.

## Loading

```python
from eval.common import load_model
model, run_dir = load_model("ckpt/MS2ADDUCT_ABL_tx_no_ion_aux_s1_000_Acc_0.854_F1_0.773")
```

The loader infers the delta-encoder architecture from the state dict and refuses
to run if `config.py` does not match the released architecture.
