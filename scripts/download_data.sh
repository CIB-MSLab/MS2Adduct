#!/usr/bin/env bash
# Download the processed data splits from the Zenodo deposit.
#
#   ZENODO_RECORD=1234567 bash scripts/download_data.sh
#
# Expected files in the record (upload from zenodo_staging/data/):
#   spectraverse_train.pkl, spectraverse_val.pkl, spectraverse_test.pkl
#   enveda_train.pkl,       enveda_val.pkl,       enveda_test.pkl
#   cib_ms_raw_msdir.tar.gz        (optional; raw QExactive .ms files)
#   SHA256SUMS_data
#
# The splits are derived from public datasets by data/build_*.py and
# data/split_by_smiles.py; you may also rebuild them instead of downloading.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RECORD="${ZENODO_RECORD:-}"
if [[ -z "$RECORD" ]]; then
  echo "ZENODO_RECORD is not set."
  echo "Usage: ZENODO_RECORD=<record-id> bash scripts/download_data.sh"
  echo "Alternatively rebuild the splits with data/build_training_dataset.py"
  echo "and data/split_by_smiles.py (see data/README.md)."
  exit 1
fi

BASE="https://zenodo.org/records/${RECORD}/files"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

curl -fL "$BASE/SHA256SUMS_data?download=1" -o "$TMP/SHA256SUMS"

FILES=(
  spectraverse_train.pkl spectraverse_val.pkl spectraverse_test.pkl
  enveda_train.pkl       enveda_val.pkl       enveda_test.pkl
  cib_ms_raw_msdir.tar.gz
)
for file in "${FILES[@]}"; do
  echo "[data] downloading $file"
  curl -fL "$BASE/${file}?download=1" -o "$TMP/$file"
done

( cd "$TMP" && sha256sum -c --ignore-missing SHA256SUMS )

mkdir -p data/splits data/splits_enveda
for split in train val test; do
  mv "$TMP/spectraverse_${split}.pkl" "data/splits/${split}.pkl"
  mv "$TMP/enveda_${split}.pkl"       "data/splits_enveda/${split}.pkl"
done

if [[ -f "$TMP/cib_ms_raw_msdir.tar.gz" ]]; then
  tar -xzf "$TMP/cib_ms_raw_msdir.tar.gz" -C data/cib-ms
fi
echo "[data] done"
