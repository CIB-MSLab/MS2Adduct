"""
build_enveda_dataset.py
将 enveda-180-filtered.mgf 转换为 PKL，然后按 SMILES 划分 train/val/test。

处理流程与 spectraverse-1.0.1.mgf 完全一致 (见 build_isotope_dataset.py / build_training_dataset.py):
  流式解析 MGF → 每条记录 {smiles, adduct, precursor_mz, normalized_collision_energy, mz, intensity}
  强度归一化到 0-100 (除以谱内最大强度) → PKL → 按 SMILES 8:1:1 划分 (seed=42)

与 spectraverse 的两处格式差异, 在此适配:
  1. 前体 m/z 字段名不同: spectraverse 用 PRECURSOR_MZ=, enveda 用 PEPMASS=
     (已用 ADDUCT_FORMULA 独立核对: C18H24N5O 单同位素质量 = 326.1982 ≈ PEPMASS 326.1987)
  2. 碰撞能量字段名不同: spectraverse 用 NORMALIZED_COLLISION_ENERGY_1= (0-100),
     enveda 用 COLLISION_ENERGIES= (Bruker timsTOF, 20/40/60)。
     - 负离子模式在原始数据里记为负值 (-40/-60), 取绝对值还原
     - 多档合并谱 (如 "20,40,60") 取各档绝对值的算术平均
     - 无法解析时置 None (与 spectraverse 缺 NCE 时的处理一致)

加合物写法差异 (仅命名, 化学实体相同), 做别名归一化:
  [M+FA-H]-   → [M+HCOOH-H]-       FA  = formic acid  = HCOOH
  [M+Hac-H]-  → [M+CH3COOH-H]-     Hac = acetic acid  = CH3COOH
其余加合物 (二聚体 [2M+...], 源内碎裂 [M-H2O+H]+ / [M-CO+H]+ 等) 不属于 9 类目标加合物,
保留原始字符串, 由 dataloader 按 config.ADDUCT_TO_IDX 过滤 —— 与 spectraverse 的处理方式一致
(spectraverse 原始文件恰好只含 9 类, 无需过滤)。
"""
import json
import os
import pickle
import random
import time
from collections import Counter, defaultdict

# 文件被放在同名目录里, 故路径为 目录/文件
MGF_PATH = "enveda-180-filtered.mgf/enveda-180-filtered.mgf"
OUTPUT_JSON = "enveda-180-filtered.json"
OUTPUT_PKL = "enveda-180-filtered.pkl"
OUTPUT_SPLITS = "splits_enveda"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42

# 同义加合物命名 → 9 类目标命名 (化学实体完全相同)
ADDUCT_ALIAS = {
    "[M+FA-H]-": "[M+HCOOH-H]-",
    "[M+Hac-H]-": "[M+CH3COOH-H]-",
}

# 9 类目标加合物 (与 config.ADDUCT_TO_IDX 一致)
CANONICAL_ADDUCTS = {
    "[M+H]+", "[M-H]-", "[M+Na]+", "[M+NH4]+", "[M+HCOOH-H]-",
    "[M+K]+", "[M+CH3COOH-H]-", "[M+Cl]-", "[M]+",
}


def parse_ce(raw):
    """解析 COLLISION_ENERGIES: 支持单值/逗号分隔多值, 负离子模式为负值。返回平均绝对值。"""
    if not raw:
        return None
    vals = []
    for tok in raw.split(','):
        tok = tok.strip()
        if not tok:
            continue
        try:
            vals.append(abs(float(tok)))
        except ValueError:
            return None
    return sum(vals) / len(vals) if vals else None


def parse_mgf(mgf_path):
    print(f"[1/3] 解析 MGF: {mgf_path}")
    t0 = time.time()
    records = []
    current = {}
    peaks = []

    n_empty_smiles = 0
    n_empty_peaks = 0
    n_alias = 0

    with open(mgf_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if line == 'BEGIN IONS':
                current = {}
                peaks = []
            elif line == 'END IONS':
                smi = current.get('SMILES', '')
                adduct = current.get('ADDUCT', '')
                precursor_mz = current.get('PEPMASS', '')
                ce = parse_ce(current.get('COLLISION_ENERGIES', ''))

                if not smi:
                    n_empty_smiles += 1
                elif not peaks:
                    n_empty_peaks += 1
                else:
                    if adduct in ADDUCT_ALIAS:
                        adduct = ADDUCT_ALIAS[adduct]
                        n_alias += 1

                    # 归一化强度到 0-100
                    max_int = max(p[1] for p in peaks)
                    mz_list = [p[0] for p in peaks]
                    intensity_list = [p[1] / max_int * 100 for p in peaks]

                    records.append({
                        'smiles': smi,
                        'adduct': adduct,
                        'precursor_mz': float(precursor_mz) if precursor_mz else 0.0,
                        'normalized_collision_energy': ce,
                        'mz': mz_list,
                        'intensity': intensity_list,
                    })

                if len(records) % 200000 == 0 and records:
                    print(f"    已解析 {len(records)} 个谱图...")
            elif line.startswith('SMILES='):
                current['SMILES'] = line[7:]
            elif line.startswith('ADDUCT='):
                current['ADDUCT'] = line[7:]
            elif line.startswith('PEPMASS='):
                value = line[8:]
                # PEPMASS 可能是 "326.198729" 或多峰谱的 "326.19 327.19"
                current['PEPMASS'] = value.split()[0] if value.split() else ''
            elif line.startswith('COLLISION_ENERGIES='):
                current['COLLISION_ENERGIES'] = line[len('COLLISION_ENERGIES='):]
            else:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        peaks.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        pass

    print(f"    → 提取 {len(records)} 个谱图, 耗时 {time.time() - t0:.1f}s")
    print(f"    跳过: 无 SMILES {n_empty_smiles} 条, 无峰 {n_empty_peaks} 条")
    print(f"    加合物别名归一化: {n_alias} 条")
    return records


def save_outputs(records, json_path, pkl_path, write_json=False):
    print(f"\n[2/3] 保存输出...")
    if write_json:
        print(f"  JSON → {json_path} ...")
        t0 = time.time()
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        size_mb = os.path.getsize(json_path) / (1024 * 1024)
        print(f"    {size_mb:.0f} MB, 耗时 {time.time() - t0:.1f}s")

    print(f"  PKL → {pkl_path} ...")
    t0 = time.time()
    with open(pkl_path, 'wb') as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(pkl_path) / (1024 * 1024)
    print(f"    {size_mb:.0f} MB, 耗时 {time.time() - t0:.1f}s")


def split_and_save(records, output_dir):
    print(f"\n[3/3] 按 SMILES 划分 → {output_dir}/")

    groups = defaultdict(list)
    for r in records:
        groups[r['smiles']].append(r)

    smiles_list = list(groups.keys())
    random.seed(SEED)
    random.shuffle(smiles_list)

    n = len(smiles_list)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    splits = {
        'train': set(smiles_list[:n_train]),
        'val': set(smiles_list[n_train:n_train + n_val]),
        'test': set(smiles_list[n_train + n_val:]),
    }

    os.makedirs(output_dir, exist_ok=True)
    for name, smi_set in splits.items():
        split_records = [r for s in smi_set for r in groups[s]]
        pkl_path = os.path.join(output_dir, f'{name}.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(split_records, f, protocol=pickle.HIGHEST_PROTOCOL)
        mb = os.path.getsize(pkl_path) / (1024 * 1024)
        print(f"  {name}: {len(split_records)} 谱图 ({len(smi_set)} SMILES) → {mb:.0f} MB")


def report(records):
    unique_smiles = len(set(r['smiles'] for r in records))
    total_peaks = sum(len(r['mz']) for r in records)
    avg_peaks = total_peaks / len(records) if records else 0

    adduct_counts = Counter(r['adduct'] for r in records)
    n_canonical = sum(v for k, v in adduct_counts.items() if k in CANONICAL_ADDUCTS)

    n_ce_missing = sum(1 for r in records if r['normalized_collision_energy'] is None)

    print("\n" + "=" * 60)
    print("统计:")
    print(f"  总谱图数:  {len(records)}")
    print(f"  唯一 SMILES: {unique_smiles}")
    print(f"  唯一 ADDUCT: {len(adduct_counts)}")
    print(f"  总峰数:    {total_peaks}")
    print(f"  平均峰数:  {avg_peaks:.1f}")
    print(f"  无 CE 记录: {n_ce_missing} ({n_ce_missing / len(records) * 100:.1f}%)")
    print("=" * 60)

    print(f"\n9 类目标加合物覆盖: {n_canonical} / {len(records)} "
          f"({n_canonical / len(records) * 100:.1f}%)")
    print(f"\n{'加合物':<20}{'谱图数':>10}  {'9类内':>6}")
    print("-" * 40)
    for k, v in adduct_counts.most_common():
        print(f"{k:<20}{v:>10}  {'  ✓' if k in CANONICAL_ADDUCTS else '  -'}")
    print("=" * 60)


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 60)
    print("构建 enveda-180-filtered 训练数据集")
    print("=" * 60)

    records = parse_mgf(MGF_PATH)
    save_outputs(records, OUTPUT_JSON, OUTPUT_PKL)
    split_and_save(records, OUTPUT_SPLITS)
    report(records)


if __name__ == '__main__':
    main()
