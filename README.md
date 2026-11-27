# MS2ADDUCT

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24587551.svg)](https://doi.org/10.5281/zenodo.24587551)

Adduct-ion type identification from MS2 spectra: given a tandem mass spectrum,
its precursor m/z, collision energy and acquisition ion mode, predict which of
nine adduct ions produced the observed precursor.

The model combines a dreaMS-pretrained spectrum encoder with a per-peak
neutral-loss (delta) branch and a learned fusion of the spectrum, delta and
collision-energy representations.

> This repository is a cleaned release of the code used for the manuscript.
> Ablation experiments and traditional-ML baseline comparisons are intentionally
> not included.

## Data and model weights

The Python code, the three trained checkpoints and the Spectraverse splits are
all in the Zenodo record **[10.5281/zenodo.24587551](https://doi.org/10.5281/zenodo.24587551)**
(record id `24587551`, version 1.0.0, MIT license):

| Artefact | Zenodo file | Installed as | Size |
|---|---|---:|---:|
| Checkpoint, seed 1 | `tx_no_ion_aux_s1.pth` | `ckpt/MS2ADDUCT_ABL_tx_no_ion_aux_s1_000_Acc_0.854_F1_0.773/best_model.pth` | 387 MB |
| Checkpoint, seed 2 | `tx_no_ion_aux_s2.pth` | `ckpt/MS2ADDUCT_ABL_tx_no_ion_aux_s2_000_Acc_0.858_F1_0.773/best_model.pth` | 387 MB |
| Checkpoint, seed 3 | `tx_no_ion_aux_s3.pth` | `ckpt/MS2ADDUCT_ABL_tx_no_ion_aux_s3_000_Acc_0.859_F1_0.781/best_model.pth` | 387 MB |
| Spectraverse splits | `spectraverse_{train,val,test}.pkl` | `data/splits/{train,val,test}.pkl` | 740 MB |
| Source code snapshot | `MS2Adduct_code_v1.0.0.tar.gz` | - (same as this repository) | 62 KB |
| Checksums | `SHA256SUMS_weights`, `SHA256SUMS_data` | used by the download scripts | 1 KB |

Download and install everything with:

```bash
bash scripts/download_weights.sh    # -> ckpt/*/best_model.pth
bash scripts/download_data.sh       # -> data/splits/*.pkl
```

Both scripts default to record `24587551` and verify every file against the
published SHA-256 sums. The dreaMS pretrained backbone (`ssl_model.ckpt`,
1.39 GB) is **not** in the record; fetch it from the official dreaMS release as
described under [Pretrained dreaMS checkpoint](#pretrained-dreams-checkpoint).

## Model

Final architecture (see [config.py](config.py) for every switch):

| Component | Choice |
|---|---|
| Spectrum encoder | dreaMS self-supervised backbone, 8 heads, 100 peaks, finetuned |
| Delta branch | one token per peak, delta = precursor_mz - peak_mz, Fourier features + 2-layer Transformer + CLS |
| Fusion | `tri_xattn`: spectrum features as query, [delta, CE] as key/value cross-attention |
| Collision energy | `cat`: CE descriptor as a third token (`CE_TOKEN_ENABLED=True`) |
| Head | hierarchical (positive / negative polarity heads) |
| Ion auxiliary head | removed (`ION_AUX_ENABLED=False`) |
| Polarity | not an input; used at inference time to suppress opposite-polarity classes (`POLARITY_MASK_OUTPUT=True`, `POLARITY_MASK_INFERENCE_ONLY=True`) |
| Selection metric | macro F1 (`--select f1`) |

Reported on the Spectraverse test split (three independent training seeds):

| Protocol | Accuracy | Macro F1 |
|---|---:|---:|
| `free` (no polarity information) | 0.8572 | 0.7759 |
| `masked` (deployment; acquisition ion mode known) | **0.8844** | **0.7909** |

Macro-F1 seed standard error is about 0.0045, so single-seed differences below
roughly 0.01 should not be interpreted.

### Code map

```
train_dreaMS_harmonictsfm_2.py      training entry point
config.py                           all hyperparameters and switches (single source of truth)
dataloader.py                       pickle -> top-100 peaks, padding, augmentation
model_dreaMS_harmonictsfm_2.py      DreaMSHT2: dreaMS-faithful spectrum encoder
model_dreaMS_harmonictsfm.py        delta Transformer + fusion + hierarchical head + polarity mask
model_dreaMS.py                     dreaMS backbone, Fourier features, weight loading
fusion.py                           FusionBlock / CEAdapter (tri_xattn lives here)
model.py                            EnergyGate
train_utils.py                      training/validation/test loops, focal loss, hierarchical assembly
```

## Installation

Development used Python 3.8.10, PyTorch 2.1.1 + CUDA 12.1, NumPy 1.24.4,
scikit-learn 1.3.0 and tqdm 4.67.1 (see [requirements.txt](requirements.txt)).
A single 24 GB GPU is enough for the default batch size of 26.

```bash
conda create -n ms2adduct python=3.8
conda activate ms2adduct
pip install -r requirements.txt
```

`config.py` uses relative paths, so run all commands from the repository root.

## Data

The processed splits are published in the Zenodo record
[10.5281/zenodo.24587551](https://doi.org/10.5281/zenodo.24587551):

```bash
bash scripts/download_data.sh
```

This fills `data/splits/` (Spectraverse, 392,513 / 50,235 / 46,049 spectra).
The splits are grouped by molecular structure with seed 42 (8:1:1); training
seeds are a separate random source. They can also be rebuilt from the public
source spectra; see [data/README.md](data/README.md).

Each record is a dict with at least:

| Key | Meaning |
|---|---|
| `mz`, `intensity` | peak lists |
| `precursor_mz` | delta-branch anchor |
| `normalized_collision_energy` | collision energy, binned into 64 levels |
| `adduct` | one of the nine classes |
| `ion_mode` / `polarity` | optional; inferred from `adduct` when absent |
| `smiles` | used only for the molecule-level split |

## Pretrained dreaMS checkpoint

Training finetunes the dreaMS self-supervised backbone. Download
`ssl_model.ckpt` (about 1.39 GB) from the official dreaMS release and place it at:

```
dreaMS/ssl_model.ckpt
```

No `msml` package is required: `model_dreaMS.py` loads the checkpoint with a
custom unpickler. Without the file the script warns and trains from scratch,
which is **not** the reported configuration.

## Model weights

The three released checkpoints (one per training seed, about 400 MB each) are
published in the same Zenodo record:

```bash
bash scripts/download_weights.sh
```

They are extracted into `ckpt/<run>/best_model.pth` using the original run
directory names. Training logs and run metadata are already in `ckpt/`.
See [ckpt/README.md](ckpt/README.md) for checksums and file names.

## Training

```bash
bash scripts/run_train.sh              # seeds 1 2 3, sequentially
SEEDS="1" bash scripts/run_train.sh    # one seed
```

Equivalent manual command for one seed:

```bash
python train_dreaMS_harmonictsfm_2.py \
    --seed 1 --select f1 \
    --fusion tri_xattn --ce_fusion cat \
    --prefix MS2ADDUCT_DreaMSHT2best_s1_
```

Checkpoints are written to `ckpt/<prefix>NNN_Acc_*_F1_*/best_model.pth`, and the
test split is evaluated automatically at the end. The released checkpoints were
produced with prefix `MS2ADDUCT_ABL_tx_no_ion_aux_`; the default prefix above
only changes the output directory name, not the model.

**Do not pass `--bases`.** With `DELTA_ENCODER="fourier"` the hand-crafted
harmonic table is unused; passing `--bases` switches the delta encoder back to
the legacy `"bases"` architecture.

## Evaluation and prediction

Evaluate a checkpoint on a labelled split:

```bash
python eval/evaluate.py \
    --ckpt ckpt/MS2ADDUCT_ABL_tx_no_ion_aux_s1_000_Acc_0.854_F1_0.773 \
    --data data/splits/test.pkl --protocol masked
```

Use `--protocol free` for the input-aligned comparison without polarity
information. Both protocols use the same weights.

Predict on unlabelled spectra in the same pickle format:

```bash
python eval/predict.py \
    --ckpt ckpt/MS2ADDUCT_ABL_tx_no_ion_aux_s1_000_Acc_0.854_F1_0.773 \
    --data my_spectra.pkl --out predictions.csv
```

Pass several checkpoints to average their probabilities (and majority-vote
their argmax predictions). See [eval/README.md](eval/README.md).

## Self-checks

The model code ships with standalone consistency checks:

```bash
python analysis/verify_fourier.py           # Fourier delta encoder
python analysis/verify_polarity_input.py    # polarity input / masking
python analysis/verify_pairdiff.py          # pairwise-difference branch (disabled by default)
python analysis/verify_dreaMS_fidelity.py   # dreaMS backbone against the original checkpoint
```

## Notes and caveats

- `config.py` is a module-level singleton. Do not construct a model, change
  `config.*`, and construct another model in the same process.
- `DREAMS2_GRAPHORMER` and `MAX_PEAKS` are coupled: the reported model uses
  100 peaks with graphormer disabled. 60 peaks should be paired with graphormer
  enabled.
- `PAIRDIFF_ENABLED` is `False`. The pairwise-difference branch was a negative
  result.
- `DELTA_TOKEN_MODE` is `"peak"`; `"pair"` and `"peak_top5"` are disproved
  variants.
- Results are noisy at the level of single seeds; use all three seeds for
  comparisons.

## Citation

The manuscript is currently under review. If you use this code before it
appears, please cite the repository and the dreaMS paper for the pretrained
backbone.

## License

Released under the [MIT License](LICENSE). The Spectraverse splits are
redistributed under the terms of the original Spectraverse release; please cite
that dataset alongside this repository.
