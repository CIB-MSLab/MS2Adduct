"""
Split a training dataset into train / val / test by SMILES.
All spectra sharing a SMILES must land in the same split to prevent leakage.
Random seed 42.

Each dataset is split independently (each shuffles its own SMILES list), so the
splits of one dataset never affect the others:
  training_dataset.pkl    -> splits/             spectraverse-1.0.1
  MassSpecGym_adduct.pkl  -> splits_MassSpecGym/
"""
import json
import pickle
import os
import random
from collections import defaultdict

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42

DATASETS = [
    {"input": "training_dataset.pkl", "output_dir": "splits"},
    {"input": "MassSpecGym_adduct.pkl", "output_dir": "splits_MassSpecGym"},
]


def load_dataset(path):
    print(f"Loading dataset: {path}")
    if path.endswith('.pkl'):
        with open(path, 'rb') as f:
            return pickle.load(f)
    elif path.endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {path}")


def split_by_smiles(records, train_ratio, val_ratio, test_ratio, seed):
    """Group by SMILES before splitting so no SMILES spans two splits."""
    groups = defaultdict(list)
    for r in records:
        groups[r['smiles']].append(r)

    smiles_list = list(groups.keys())
    print(f"Total spectra: {len(records)}")
    print(f"Unique SMILES: {len(smiles_list)}")

    random.seed(seed)
    random.shuffle(smiles_list)

    n = len(smiles_list)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_smiles = set(smiles_list[:n_train])
    val_smiles = set(smiles_list[n_train:n_train + n_val])
    test_smiles = set(smiles_list[n_train + n_val:])

    train_records = [r for s in train_smiles for r in groups[s]]
    val_records = [r for s in val_smiles for r in groups[s]]
    test_records = [r for s in test_smiles for r in groups[s]]

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_records)} spectra ({len(train_smiles)} SMILES)")
    print(f"  Val:   {len(val_records)} spectra ({len(val_smiles)} SMILES)")
    print(f"  Test:  {len(test_records)} spectra ({len(test_smiles)} SMILES)")

    return train_records, val_records, test_records


def save_splits(train, val, test, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for name, records in [('train', train), ('val', val), ('test', test)]:
        pkl_path = os.path.join(output_dir, f'{name}.pkl')
        print(f"\nSaving {name} ({len(records)} records)...")
        with open(pkl_path, 'wb') as f:
            pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  {pkl_path} ({os.path.getsize(pkl_path) / 1024 / 1024:.0f} MB)")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    for ds in DATASETS:
        print("=" * 60)
        print(f"Splitting by SMILES: {ds['input']} (seed={SEED})")
        print("=" * 60)

        records = load_dataset(ds['input'])
        train, val, test = split_by_smiles(records, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED)
        save_splits(train, val, test, ds['output_dir'])

        print()

    print("=" * 60)
    print("All done")
    print("=" * 60)
