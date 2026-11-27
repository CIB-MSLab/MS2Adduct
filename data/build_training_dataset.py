"""
Extract the training dataset from spectraverse-1.0.1.mgf.
Fields per record: SMILES, ADDUCT, PRECURSOR_MZ, mz, intensity.
Output: JSON + PKL.
"""
import json
import os
import pickle
import time

MGF_PATH = "spectraverse-1.0.1.mgf"
OUTPUT_JSON = "training_dataset.json"
OUTPUT_PKL = "training_dataset.pkl"


def parse_mgf(mgf_path):
    """Stream-parse an MGF file and return all spectrum records."""
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
                    print(f"   parsed {len(records)} spectra...")
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
    print(f"\nSaving JSON -> {json_path} ...")
    t0 = time.time()
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    size_mb = os.path.getsize(json_path) / (1024 * 1024)
    print(f"  JSON: {size_mb:.0f} MB, elapsed {time.time() - t0:.1f}s")

    print(f"Saving PKL -> {pkl_path} ...")
    t0 = time.time()
    with open(pkl_path, 'wb') as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(pkl_path) / (1024 * 1024)
    print(f"  PKL: {size_mb:.0f} MB, elapsed {time.time() - t0:.1f}s")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 60)
    print("Building the training dataset")
    print("=" * 60)

    print(f"\n[1/2] Parsing MGF: {MGF_PATH}")
    print("    (large file, expect a few minutes)...")
    t0 = time.time()
    records = parse_mgf(MGF_PATH)
    print(f"    -> extracted {len(records)} spectra in {time.time() - t0:.1f}s")

    print(f"\n[2/2] Saving outputs...")
    save_outputs(records, OUTPUT_JSON, OUTPUT_PKL)

    unique_smiles = len(set(r['smiles'] for r in records))
    unique_adducts = len(set(r['adduct'] for r in records))
    total_peaks = sum(len(r['mz']) for r in records)

    print("\n" + "=" * 60)
    print("Dataset statistics:")
    print(f"  total spectra: {len(records)}")
    print(f"  unique SMILES: {unique_smiles}")
    print(f"  unique ADDUCT: {unique_adducts}")
    print(f"  total peaks: {total_peaks}")
    print(f"  mean peaks/spectrum: {total_peaks / len(records):.1f}")
    print("=" * 60)
