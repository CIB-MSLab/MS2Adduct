"""
按 SMILES 划分训练数据集为 train / val / test。
同一 SMILES 的所有谱图必须分到同一集合，防止数据泄漏。
随机种子 42。

各数据集独立划分（各自用自己的 SMILES 列表洗牌），互不影响:
  training_dataset.pkl    → splits/             spectraverse-1.0.1
  MassSpecGym_adduct.pkl  → splits_MassSpecGym/
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
    print(f"加载数据集: {path}")
    if path.endswith('.pkl'):
        with open(path, 'rb') as f:
            return pickle.load(f)
    elif path.endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise ValueError(f"不支持的文件格式: {path}")


def split_by_smiles(records, train_ratio, val_ratio, test_ratio, seed):
    """按 SMILES 分组后划分，保证同一 SMILES 不出现在多个集合中"""
    # 按 SMILES 分组
    groups = defaultdict(list)
    for r in records:
        groups[r['smiles']].append(r)

    smiles_list = list(groups.keys())
    print(f"总谱图数: {len(records)}")
    print(f"唯一 SMILES: {len(smiles_list)}")

    # 随机打乱 SMILES
    random.seed(seed)
    random.shuffle(smiles_list)

    n = len(smiles_list)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_smiles = set(smiles_list[:n_train])
    val_smiles = set(smiles_list[n_train:n_train + n_val])
    test_smiles = set(smiles_list[n_train + n_val:])

    # 展开记录
    train_records = [r for s in train_smiles for r in groups[s]]
    val_records = [r for s in val_smiles for r in groups[s]]
    test_records = [r for s in test_smiles for r in groups[s]]

    print(f"\n划分结果:")
    print(f"  Train: {len(train_records)} 谱图 ({len(train_smiles)} SMILES)")
    print(f"  Val:   {len(val_records)} 谱图 ({len(val_smiles)} SMILES)")
    print(f"  Test:  {len(test_records)} 谱图 ({len(test_smiles)} SMILES)")

    return train_records, val_records, test_records


def save_splits(train, val, test, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for name, records in [('train', train), ('val', val), ('test', test)]:
        pkl_path = os.path.join(output_dir, f'{name}.pkl')
        print(f"\n保存 {name} ({len(records)} 条)...")
        with open(pkl_path, 'wb') as f:
            pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  {pkl_path} ({os.path.getsize(pkl_path) / 1024 / 1024:.0f} MB)")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    for ds in DATASETS:
        print("=" * 60)
        print(f"按 SMILES 划分: {ds['input']} (seed={SEED})")
        print("=" * 60)

        records = load_dataset(ds['input'])
        train, val, test = split_by_smiles(records, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED)
        save_splits(train, val, test, ds['output_dir'])

        print()

    print("=" * 60)
    print("全部完成")
    print("=" * 60)
