#!/usr/bin/env bash
# Train the final MS2ADDUCT model (DreaMS backbone + Fourier delta + tri_xattn
# + CE token, no ion-mode auxiliary head).
#
#   bash scripts/run_train.sh                 # seeds 1 2 3, sequentially
#   SEEDS="1" bash scripts/run_train.sh       # one seed
#   PY=/path/to/python bash scripts/run_train.sh
#
# Run from anywhere: the script switches to the repository root because
# config.py uses relative data paths.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-python}"
SEEDS="${SEEDS-1 2 3}"
PREFIX="${PREFIX:-MS2ADDUCT_DreaMSHT2best_}"

# Preflight: refuse to train if config.py does not describe the released model.
"$PY" - <<'PY'
import sys
sys.path.insert(0, ".")
import config

WANT = {
    "FUSION": "tri_xattn",
    "CE_FUSION": "cat",
    "CE_TOKEN_ENABLED": True,
    "DELTA_ENCODER": "fourier",
    "DELTA_TOKEN_MODE": "peak",
    "ION_AUX_ENABLED": False,
    "POLARITY_AS_INPUT": False,
    "POLARITY_MASK_OUTPUT": True,
    "POLARITY_MASK_INFERENCE_ONLY": True,
    "PAIRDIFF_ENABLED": False,
}
bad = {k: (getattr(config, k, "<missing>"), v) for k, v in WANT.items()
       if getattr(config, k, None) != v}
if bad:
    raise SystemExit(f"[train] config does not match the final model (actual, expected): {bad}")
print(f"[train] preflight OK: {WANT}")
PY

for s in $SEEDS; do
  echo "[train] seed=$s start $(date '+%F %T')"
  "$PY" train_dreaMS_harmonictsfm_2.py \
      --seed "$s" \
      --select f1 \
      --fusion tri_xattn \
      --ce_fusion cat \
      --prefix "${PREFIX}s${s}_"
  echo "[train] seed=$s done  $(date '+%F %T')"
done
