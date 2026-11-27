# Zenodo staging area

This directory is **not** part of the git repository (see `.gitignore`). It is a
local staging area for the files that are too large for GitHub.

## Weights (`weights/`)

Upload to the Zenodo record:

```
tx_no_ion_aux_s1.pth
tx_no_ion_aux_s2.pth
tx_no_ion_aux_s3.pth
```

These are hardlinks to the original `ckpt/.../best_model.pth` files, so no extra
disk space is used locally. Checksums are in `SHA256SUMS_weights`.

## Data (`data/`)

Upload to the same Zenodo record:

```
spectraverse_train.pkl   -> data/splits/train.pkl
spectraverse_val.pkl     -> data/splits/val.pkl
spectraverse_test.pkl    -> data/splits/test.pkl
```

The names on the left are the names to use in the Zenodo record; the arrows show
where `scripts/download_data.sh` installs them. Checksums are in
`SHA256SUMS_data`.

Local leftovers from earlier drafts (`data/enveda_*.pkl`,
`data/cib_ms_raw_msdir.tar.gz`) are not part of the deposit and are not used by
the released code.

## Code (`code/`)

`MS2Adduct_code_<commit>.tar.gz` is created with

```bash
git archive --format=tar.gz --prefix=MS2Adduct-<commit>/ <commit> \
  -o zenodo_staging/code/MS2Adduct_code_<commit>.tar.gz
```

## Upload

`zenodo_upload.py` creates the record, uploads every file listed above and can
publish it. See the module docstring for the exact commands; the access token
is read from `ZENODO_TOKEN`.

## After creating the record

1. Upload `SHA256SUMS_weights` and `SHA256SUMS_data` as separate files.
2. Replace `<record-id>` in the root README, `ckpt/README.md`, and the download
   script usage examples with the real Zenodo record id.

## Expected sizes

| Item | Size |
|---|---:|
| three checkpoints | 3 x 406 MB |
| Spectraverse splits | 740 MB |
| code archive | 3.4 MB |
