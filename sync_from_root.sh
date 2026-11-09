#!/usr/bin/env bash
# 把仓库根目录的最新代码同步进本快照, 并重新应用"当前最优模型"的配置覆盖。
#
#   bash best_model/sync_from_root.sh
#
# 同步后与根目录的**唯一差异**是 config.py 里的 ION_AUX_ENABLED=False
# (2026-11-04 定案: 去掉离子模式辅助头为最优)。根目录保持 True 以复现历史消融臂。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FILES=(
  config.py
  dataloader.py
  fusion.py
  model.py
  model_dreaMS.py
  model_dreaMS_harmonic.py
  model_dreaMS_harmonictsfm.py
  model_dreaMS_harmonictsfm_2.py
  train_dreaMS_harmonic.py
  train_dreaMS_harmonictsfm_2.py
)

for f in "${FILES[@]}"; do
  cp -p "$f" "best_model/$f"
  echo "synced $f"
done

# ── 重新应用最优模型的配置覆盖 ──
PY="${PY:-/home/ssc/anaconda3/envs/torch/bin/python}"
"$PY" - <<'PY'
import pathlib
import re

p = pathlib.Path("best_model/config.py")
src = p.read_text(encoding="utf-8")

BLOCK = '''# ★ 2026-11-04 定案: 最终架构去掉离子模式辅助头 (ION_AUX_ENABLED=False) 为当前最优。
#   Fourier δ + tri_xattn, 盲协议 3 seeds (spectraverse test):
#     free   Acc 0.8572 / Macro F1 0.7759   (带离子头 0.8583 / 0.7717)
#     masked Acc 0.8844 / Macro F1 0.7909   (带离子头 0.8833 / 0.7880)
#   配对 ΔF1 +0.0028 (t=0.76, n.s.), 方向 2/3 —— 点估计略优, 按点估计定为最优。
#   组件/融合排序不变 (见 ablation/results/noion_all_masked.json, anchor=tx_no_ion_aux)。
# 注意: 本快照 (best_model/) 显式设为 False; 仓库根目录 config.py 仍为 True,
#   因为历史消融臂 (tri_xattn / control 等) 依赖该默认值才能复现旧 ckpt。
#   在根目录重跑任何旧臂时, 不要以 best_model/ 的取值为准。
ION_AUX_ENABLED = False'''

if "ION_AUX_ENABLED = True" in src:
    src = re.sub(r"^ION_AUX_ENABLED = True$", BLOCK, src, count=1, flags=re.M)
    p.write_text(src, encoding="utf-8")
    print("applied override: ION_AUX_ENABLED = False")
elif "ION_AUX_ENABLED = False" in src:
    print("override already present: ION_AUX_ENABLED = False")
else:
    raise SystemExit("best_model/config.py 里找不到 ION_AUX_ENABLED 开关, 同步失败")
PY
