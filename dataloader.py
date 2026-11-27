import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import config


def ion_mode_to_idx(v):
    """Map an ion-mode literal to 0 (positive) / 1 (negative).

    Accepts the spellings used by the different datasets: 'ESI+'/'ESI-'/
    'positive'/'negative'/'+'/'-' as well as 0/1. Unrecognised values fall back
    to positive, matching the default class [M+H]+ in ADDUCT_TO_IDX.
    """
    s = str(v).strip().lower()
    if s in ('1', 'negative', 'neg', 'esi-', '-', 'n'):
        return 1
    return 0


class AdductDataset(Dataset):
    def __init__(self, pkl_path, augment=False, fraction=None):
        if fraction is None:
            fraction = float(getattr(config, 'DATA_FRACTION', 1.0))
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        self.samples = []
        for item in data:
            adduct = item.get('adduct', '')
            if adduct in config.ADDUCT_TO_IDX:
                self.samples.append(item)

        if fraction < 1.0:
            n = max(1, int(len(self.samples) * fraction))
            rng = np.random.RandomState(42)
            idx = rng.choice(len(self.samples), n, replace=False)
            self.samples = [self.samples[i] for i in idx]

        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        mz_arr = np.array(item.get('mz', [100.0]), dtype=np.float32)
        int_arr = np.array(item.get('intensity', [1e-5]), dtype=np.float32)

        adduct = item.get('adduct', '')
        label = config.ADDUCT_TO_IDX.get(adduct, 0)

        ce = item.get('normalized_collision_energy')
        if ce is None:
            ce = 0.0
        ce = float(ce)

        ion = item.get('ion_mode', item.get('polarity'))
        if ion is None or (isinstance(ion, str) and not ion.strip()):
            polarity = config.POLARITY_OF_ADDUCT.get(adduct, 0)
        else:
            polarity = ion_mode_to_idx(ion)

        if self.augment and len(mz_arr) > 0:
            low_idx = np.where(int_arr < config.MS_AUG_REMOVE_THRESHOLD)[0]
            if len(low_idx) > 0:
                n_remove = int(len(low_idx) * config.MS_AUG_REMOVE_RATIO)
                if n_remove > 0:
                    remove_idx = np.random.choice(
                        low_idx, n_remove, replace=False
                    )
                    keep = np.ones(len(mz_arr), dtype=bool)
                    keep[remove_idx] = False
                    mz_arr, int_arr = mz_arr[keep], int_arr[keep]

            if len(mz_arr) > 0:
                jitter = np.random.uniform(
                    -config.MS_AUG_JITTER_RANGE,
                    config.MS_AUG_JITTER_RANGE,
                    size=int_arr.shape,
                )
                int_arr = np.clip(int_arr * (1.0 + jitter), 0.0, None)

                n_noise = np.random.randint(
                    0, config.MS_AUG_MAX_NOISE_PEAKS + 1
                )
                if n_noise > 0:
                    noise_mz = np.random.uniform(
                        10.0, 1500.0, size=n_noise
                    ).astype(np.float32)
                    noise_int = np.random.uniform(
                        0.0001, 0.01, size=n_noise
                    ).astype(np.float32)
                    mz_arr = np.concatenate((mz_arr, noise_mz))
                    int_arr = np.concatenate((int_arr, noise_int))

        if len(mz_arr) == 0:
            mz_arr = np.array([100.0], dtype=np.float32)
            int_arr = np.array([1e-5], dtype=np.float32)

        sort_idx = np.argsort(int_arr)[::-1][:config.MAX_PEAKS]
        mz_arr, int_arr = mz_arr[sort_idx], int_arr[sort_idx]

        if int_arr.max() > 0:
            int_arr = int_arr / int_arr.max()

        pad_len = config.MAX_PEAKS - len(mz_arr)
        if pad_len > 0:
            mz_arr = np.pad(mz_arr, (0, pad_len), constant_values=0.0)
            int_arr = np.pad(int_arr, (0, pad_len), constant_values=0.0)

        return {
            'mz': torch.tensor(mz_arr, dtype=torch.float32),
            'intensity': torch.tensor(int_arr, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long),
            'ce': torch.tensor([ce], dtype=torch.float32),
            'polarity': torch.tensor([polarity], dtype=torch.long),
            'precursor_mz': torch.tensor(
                [float(item.get('precursor_mz', 0.0))], dtype=torch.float32
            ),
        }


def get_dataloader(pkl_path, batch_size, augment=False, shuffle=True):
    dataset = AdductDataset(pkl_path, augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
    )
