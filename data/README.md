# Data

`config.py` reads pickles from `data/splits/` (Spectraverse) and
`data/splits_enveda/` (Enveda-180). Both are grouped by molecular structure
with seed 42 at an 8:1:1 ratio; the training seed is a separate random source.

## Spectraverse (main benchmark)

| Split | Spectra | Classes |
|---|---:|---|
| train | 392,513 | all nine |
| val | 50,235 | all nine |
| test | 46,049 | all nine |

Nine adduct classes, in label order:

```
[M+H]+  [M-H]-  [M+Na]+  [M+NH4]+  [M+HCOOH-H]-  [M+K]+  [M+CH3COOH-H]-  [M+Cl]-  [M]+
```

The processed splits are available from the Zenodo deposit (see the root
README) and can be downloaded with `scripts/download_data.sh`.

### Rebuilding the splits

1. Obtain the Spectraverse spectra (`spectraverse-1.0.1.mgf`) and metadata CSV
   from the original Spectraverse release.
2. Build the filtered training pickle:

   ```bash
   python data/build_training_dataset.py
   ```

   The script defines the adduct parsing and the peak/metadata fields; adjust
   the input path at the top to your local copy of the MGF file.
3. Split by structure:

   ```bash
   python data/split_by_smiles.py
   ```

   The split is by canonical SMILES (seed 42), not by spectrum, so spectra of
   the same molecule never appear in two splits.

## Enveda-180 (secondary dataset)

`data/build_enveda_dataset.py` builds `data/splits_enveda/`. It covers 32 adduct
types in the raw data; records outside the nine-class vocabulary are ignored at
load time. `[M]+` is absent from this dataset. These splits are not part of the
Zenodo deposit; build them locally if you want to rerun the secondary results.

| Split | Records (all adduct types) |
|---|---:|
| train | 940,228 |
| val | 117,586 |
| test | 117,388 |

The counts above include every adduct type present in the raw file. The loader
keeps only the nine benchmark classes, so the effective training set is smaller.

`config.DATA_SETS` also defines `enveda_test` (train on Spectraverse, test on
Enveda), used for cross-dataset evaluation.

## In-house QExactive set (`data/cib-ms/`)

External validation set acquired in-house on a QExactive (3,437 spectra after
deduplication, three adduct classes):

| File | Records | Purpose |
|---|---:|---|
| `cib-ms.pkl` | 5,189 | all usable records |
| `cib-ms-dedup-spectraverse.pkl` | 3,437 | deduplicated against Spectraverse; used for evaluation |
| `cib-ms-dedup-train.pkl` | 3,812 | deduplicated against the Spectraverse training split |

`build_cib_dataset.py` builds these pickles from the raw `.ms` files. The raw
directory is not distributed; only the derived pickles are stored in git.

## Record format

Every pickle is a list of dicts. The training loader only reads:

| Key | Type | Meaning |
|---|---|---|
| `mz` | list[float] | peak m/z values |
| `intensity` | list[float] | raw peak intensities |
| `precursor_mz` | float | precursor m/z (delta anchor) |
| `normalized_collision_energy` | float or None | CE, binned into 64 levels |
| `adduct` | str | one of the nine classes |
| `ion_mode` / `polarity` | str/int, optional | acquisition polarity; inferred from `adduct` when absent |
| `smiles` | str | used only by `split_by_smiles.py` |

`dataloader.py` keeps the top 100 peaks, normalizes to the base peak, zero-pads
to 100, and applies augmentation only during training.

## Notes on provenance

The Spectraverse and Enveda spectra are redistributed under the terms of their
original releases. Please cite the corresponding dataset papers. The in-house
QExactive spectra were acquired by the authors; contact them regarding reuse.
