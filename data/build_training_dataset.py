"""
从 spectraverse-1.0.1.mgf 提取训练数据集。
每条记录: SMILES, ADDUCT, PRECURSOR_MZ, mz, intensity
输出: JSON + PKL
"""
import json
import os
import pickle
import time

MGF_PATH = "spectraverse-1.0.1.mgf"
OUTPUT_JSON = "training_dataset.json"
OUTPUT_PKL = "training_dataset.pkl"


def parse_mgf(mgf_path):
    """流式解析 MGF，返回所有谱图记录列表"""
    records = []
    current = {}
    peaks = []

    with open(mgf_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if line == 'BEGIN IONS':
                current = {}
                peaks = []
            elif line == 'END IONS':
                smi = current.get('SMILES', '')
                adduct = current.get('ADDUCT', '')
                precursor_mz = current.get('PRECURSOR_MZ', '')

                if smi and peaks:
                    # 归一化强度到 0-100
                    max_int = max(p[1] for p in peaks)
                    mz_list = [p[0] for p in peaks]
                    intensity_list = [p[1] / max_int * 100 for p in peaks]

                    records.append({
                        'smiles': smi,
                        'adduct': adduct,
                        'precursor_mz': float(precursor_mz) if precursor_mz else 0.0,
                        'mz': mz_list,
                        'intensity': intensity_list,
                    })

                if len(records) % 100000 == 0 and records:
                    print(f"   已解析 {len(records)} 个谱图...")
            elif line.startswith('SMILES='):
                current['SMILES'] = line[7:]
            elif line.startswith('ADDUCT='):
                current['ADDUCT'] = line[7:]
            elif line.startswith('PRECURSOR_MZ='):
                current['PRECURSOR_MZ'] = line[13:]
            else:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        peaks.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        pass

    return records


def save_outputs(records, json_path, pkl_path):
    print(f"\n保存 JSON → {json_path} ...")
    t0 = time.time()
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    size_mb = os.path.getsize(json_path) / (1024 * 1024)
    print(f"  JSON: {size_mb:.0f} MB, 耗时 {time.time() - t0:.1f}s")

    print(f"保存 PKL → {pkl_path} ...")
    t0 = time.time()
    with open(pkl_path, 'wb') as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(pkl_path) / (1024 * 1024)
    print(f"  PKL: {size_mb:.0f} MB, 耗时 {time.time() - t0:.1f}s")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 60)
    print("构建训练数据集")
    print("=" * 60)

    print(f"\n[1/2] 解析 MGF: {MGF_PATH}")
    print("    (大文件，预计需要数分钟)...")
    t0 = time.time()
    records = parse_mgf(MGF_PATH)
    print(f"    → 提取 {len(records)} 个谱图, 耗时 {time.time() - t0:.1f}s")

    print(f"\n[2/2] 保存输出...")
    save_outputs(records, OUTPUT_JSON, OUTPUT_PKL)

    # 统计
    unique_smiles = len(set(r['smiles'] for r in records))
    unique_adducts = len(set(r['adduct'] for r in records))
    total_peaks = sum(len(r['mz']) for r in records)

    print("\n" + "=" * 60)
    print("数据集统计:")
    print(f"  总谱图数: {len(records)}")
    print(f"  唯一 SMILES: {unique_smiles}")
    print(f"  唯一 ADDUCT: {unique_adducts}")
    print(f"  总峰数: {total_peaks}")
    print(f"  平均峰数/谱图: {total_peaks / len(records):.1f}")
    print("=" * 60)
