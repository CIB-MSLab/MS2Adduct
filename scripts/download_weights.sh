#!/usr/bin/env bash
# Download the released checkpoints from the Zenodo deposit.
#
# Record: https://doi.org/10.5281/zenodo.24587551 (id 24587551, v1.0.0)
#
#   bash scripts/download_weights.sh
#   ZENODO_RECORD=1234567 bash scripts/download_weights.sh   # another version
#
# Expected files in the record (upload from zenodo_staging/weights/):
#   tx_no_ion_aux_s1.pth, tx_no_ion_aux_s2.pth, tx_no_ion_aux_s3.pth,
#   SHA256SUMS_weights
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RECORD="${ZENODO_RECORD:-24587551}"
if [[ -z "$RECORD" ]]; then
  echo "ZENODO_RECORD is empty."
  echo "Usage: bash scripts/download_weights.sh"
  echo "The checkpoints are not redistributed in git (3 x ~400 MB)."
  exit 1
fi

BASE="https://zenodo.org/records/${RECORD}/files"
RUNS=(
  "tx_no_ion_aux_s1.pth:MS2ADDUCT_ABL_tx_no_ion_aux_s1_000_Acc_0.854_F1_0.773"
  "tx_no_ion_aux_s2.pth:MS2ADDUCT_ABL_tx_no_ion_aux_s2_000_Acc_0.858_F1_0.773"
  "tx_no_ion_aux_s3.pth:MS2ADDUCT_ABL_tx_no_ion_aux_s3_000_Acc_0.859_F1_0.781"
)

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

curl -fL "$BASE/SHA256SUMS_weights?download=1" -o "$TMP/SHA256SUMS"

for entry in "${RUNS[@]}"; do
  file="${entry%%:*}"
  echo "[weights] downloading $file"
  curl -fL "$BASE/$file?download=1" -o "$TMP/$file"
done

( cd "$TMP" && sha256sum -c SHA256SUMS )

for entry in "${RUNS[@]}"; do
  file="${entry%%:*}"
  run="${entry#*:}"
  mkdir -p "ckpt/$run"
  mv "$TMP/$file" "ckpt/$run/best_model.pth"
  echo "[weights] $file -> ckpt/$run/best_model.pth"
done
echo "[weights] done"
