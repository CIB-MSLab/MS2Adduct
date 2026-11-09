#!/usr/bin/env bash
# 复现当前最优模型: DreaMSHT2 + Fourier δ + tri_xattn + CE token + 无离子辅助头。
# 必须在仓库任意位置调用均可 —— 脚本会自行切到仓库根目录 (训练脚本用相对路径读数据)。
#
#   bash best_model/run_train.sh                  # seeds 1 2 3, 顺序执行
#   SEEDS="1" bash best_model/run_train.sh        # 只跑一个 seed
#   PY=/path/to/python bash best_model/run_train.sh
#
# 注意: 这是训练脚本, 不是消融工具。不要额外传 --bases (会切回手工基频旧架构)。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/home/ssc/anaconda3/envs/torch/bin/python}"
SEEDS="${SEEDS-1 2 3}"   # 空字符串 = 只跑预检, 不训练
PREFIX="${PREFIX:-MS2ADDUCT_DreaMSHT2best_}"

# ── 预检: 确认用的是 best_model/ 的配置, 且就是当前最优定义 ──
"$PY" - <<'PY'
import sys
sys.path.insert(0, "best_model")
import config

WANT = {
    "FUSION": "tri_xattn",
    "CE_FUSION": "cat",
    "CE_TOKEN_ENABLED": True,
    "DELTA_ENCODER": "fourier",
    "DELTA_TOKEN_MODE": "peak",
    "ION_AUX_ENABLED": False,          # 2026-11-04 起的最优定义
    "POLARITY_AS_INPUT": False,
    "POLARITY_MASK_OUTPUT": True,
    "POLARITY_MASK_INFERENCE_ONLY": True,
    "PAIRDIFF_ENABLED": False,
}
bad = {k: (getattr(config, k, "<missing>"), v) for k, v in WANT.items()
       if getattr(config, k, None) != v}
if bad:
    raise SystemExit(f"[best_model] 配置与当前最优模型不符 (实际, 期望): {bad}")
print(f"[best_model] 预检通过: {WANT}")
PY

for s in $SEEDS; do
  echo "[best_model] seed=$s 开始 $(date '+%F %T')"
  "$PY" best_model/train_dreaMS_harmonictsfm_2.py \
      --seed "$s" \
      --select f1 \
      --fusion tri_xattn \
      --ce_fusion cat \
      --prefix "${PREFIX}s${s}_"
  echo "[best_model] seed=$s 结束 $(date '+%F %T')"
done
