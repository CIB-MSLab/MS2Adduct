# Evaluation and prediction

Both scripts are self-contained and load the released checkpoints through
`eval/common.py`.

## Evaluate

```bash
python eval/evaluate.py \
    --ckpt ckpt/MS2ADDUCT_ABL_tx_no_ion_aux_s1_000_Acc_0.854_F1_0.773 \
    --data data/splits/test.pkl \
    --protocol masked \
    --out results/eval_s1_masked.json
```

- `--protocol masked` (default, deployment): the acquisition ion mode is known
  and used to suppress opposite-polarity adduct classes at inference time.
- `--protocol free`: no polarity information is used.

Reported metrics: accuracy, macro F1 over the nine classes, top-3 accuracy,
per-class precision/recall/F1, and the confusion matrix.

## Predict

```bash
python eval/predict.py \
    --ckpt ckpt/MS2ADDUCT_ABL_tx_no_ion_aux_s*/ \
    --data my_spectra.pkl \
    --out predictions.csv
```

Input is a pickle file containing a list of dicts:

```python
{
    "mz": [...], "intensity": [...],
    "precursor_mz": 342.15,
    "normalized_collision_energy": 30.0,   # optional, default 0
    "ion_mode": "ESI+",                    # optional but recommended
    "adduct": "[M+H]+",                    # optional, only used for scoring
    # extra keys such as id/name/smiles are copied through --keep
}
```

Output columns: predicted adduct, confidence, the nine class probabilities,
per-seed predictions when several checkpoints are given, and the selected input
metadata fields.

With several checkpoints, probabilities are averaged and the argmax is decided
by majority vote (ties fall back to the mean-probability argmax). Use
`--no-ensemble-vote` to report the mean-probability argmax only.

Polarity handling:

- `--polarity auto` (default) reads `ion_mode` / `polarity` from each record;
- records without that metadata are predicted without masking;
- `--polarity positive|negative` overrides missing metadata;
- `--no-mask` disables the inference-time polarity constraint entirely.

Peak preprocessing matches training: top 100 peaks by intensity, base-peak
normalization, zero padding to 100.
