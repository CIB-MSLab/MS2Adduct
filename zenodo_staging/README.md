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
enveda_train.pkl         -> data/splits_enveda/train.pkl
enveda_val.pkl           -> data/splits_enveda/val.pkl
enveda_test.pkl          -> data/splits_enveda/test.pkl
cib_ms_raw_msdir.tar.gz  -> data/cib-ms/MSDIR/
```

The names on the left are the names to use in the Zenodo record; the arrows show
where `scripts/download_data.sh` installs them. Checksums are in
`SHA256SUMS_data` (paths relative to the repository root).

## After creating the record

1. Upload `SHA256SUMS_weights` and `SHA256SUMS_data` as separate files.
2. Replace `<record-id>` in the root README, `ckpt/README.md`, and the download
   script usage examples with the real Zenodo record id.

## Expected sizes

| Item | Size |
|---|---:|
| three checkpoints | 3 x 406 MB |
| Spectraverse splits | 740 MB |
| Enveda splits | 532 MB |
| raw cib-ms archive | 1.6 MB |
