# Data

`config.py` reads pickles from `data/splits/` (Spectraverse). The splits are
grouped by molecular structure with seed 42 at an 8:1:1 ratio; the training seed
is a separate random source.

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

The processed splits are published in the Zenodo record
[10.5281/zenodo.24587551](https://doi.org/10.5281/zenodo.24587551) and can be
downloaded with `scripts/download_data.sh`.

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

The Spectraverse spectra are redistributed under the terms of the original
release. Please cite the corresponding dataset paper.
