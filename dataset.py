"""
PHENO TYPE — dataset.py
Loads final_dna_v2.csv and provides MalwareDataset + StratifiedBatchSampler.
"""

import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from collections import defaultdict

FAMILY_TO_IDX = {
    'AgentTesla': 0,
    'Formbook':   1,
    'Lokibot':    2,
    'Redline':    3,
    'njRAT':      4,
}
IDX_TO_FAMILY = {v: k for k, v in FAMILY_TO_IDX.items()}

# Class weights: 1932 / (5 × n_samples)
CLASS_WEIGHTS = {
    0: 0.772,   # AgentTesla  (500)
    1: 1.421,   # Formbook    (272)
    2: 0.886,   # Lokibot     (436)
    3: 0.772,   # Redline     (500)
    4: 1.723,   # njRAT       (224)
}


class MalwareDataset(Dataset):
    """
    Loads the CSV and exposes (tokens: LongTensor[1200], label: int) pairs.
    sha256 is stored as metadata for deduplication checks.
    """

    def __init__(self, csv_path: str, indices=None):
        df = pd.read_csv(csv_path)
        if indices is not None:
            df = df.iloc[indices].reset_index(drop=True)

        self.sha256  = df['sha256'].values
        self.labels  = df['family'].map(FAMILY_TO_IDX).values.astype('int64')
        tok_cols     = [f'tok_{i}' for i in range(1200)]
        self.tokens  = df[tok_cols].values.astype('int64')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return (
            torch.tensor(self.tokens[i], dtype=torch.long),
            torch.tensor(self.labels[i],  dtype=torch.long),
        )


class StratifiedBatchSampler(Sampler):
    """
    Guarantees each batch contains samples from every family.
    With batch_size=256 and 5 families each batch targets ~51 per family.
    """

    def __init__(self, labels, batch_size: int, drop_last: bool = False):
        self.batch_size = batch_size
        self.drop_last  = drop_last
        self.num_classes = len(set(labels))

        # Group indices by class
        self.class_indices = defaultdict(list)
        for idx, lbl in enumerate(labels):
            self.class_indices[int(lbl)].append(idx)

        # Shuffle on init
        for k in self.class_indices:
            random.shuffle(self.class_indices[k])

        n_per_class  = batch_size // self.num_classes
        remainder    = batch_size - n_per_class * self.num_classes

        # How many full batches can we build?
        min_class_len  = min(len(v) for v in self.class_indices.values())
        self.n_batches = min_class_len // n_per_class

        self._n_per_class = n_per_class
        self._remainder   = remainder   # extra samples drawn round-robin

    def __iter__(self):
        # Re-shuffle each epoch
        shuffled = {k: list(v) for k, v in self.class_indices.items()}
        for k in shuffled:
            random.shuffle(shuffled[k])

        class_iters = {k: iter(shuffled[k]) for k in shuffled}

        for _ in range(self.n_batches):
            batch = []
            for k in range(self.num_classes):
                for _ in range(self._n_per_class):
                    try:
                        batch.append(next(class_iters[k]))
                    except StopIteration:
                        break
            # Fill remainder round-robin across classes
            extra_k = 0
            for _ in range(self._remainder):
                try:
                    batch.append(next(class_iters[extra_k % self.num_classes]))
                except StopIteration:
                    pass
                extra_k += 1

            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches


def make_splits(csv_path: str, train_frac=0.70, val_frac=0.15, seed=42):
    """
    Stratified train / val / test split.
    Returns three MalwareDataset objects.
    """
    from sklearn.model_selection import train_test_split

    df     = pd.read_csv(csv_path, usecols=['family'])
    labels = df['family'].map(FAMILY_TO_IDX).values
    indices = np.arange(len(labels))

    # First split off test
    train_val_idx, test_idx = train_test_split(
        indices, test_size=1 - train_frac - val_frac,
        stratify=labels[indices], random_state=seed
    )
    # Then split train vs val
    relative_val = val_frac / (train_frac + val_frac)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=relative_val,
        stratify=labels[train_val_idx], random_state=seed
    )

    return (
        MalwareDataset(csv_path, train_idx),
        MalwareDataset(csv_path, val_idx),
        MalwareDataset(csv_path, test_idx),
    )
