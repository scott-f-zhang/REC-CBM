from typing import Optional, List
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import random
import numpy as np


ESSAY_CONCEPT_COLUMNS: List[str] = [
    "TC", "UE", "OC", "GM", "VA", "SV", "CTD", "FR"
]

# Dataset metadata
final_label = ['score']
final_label_vals = [0, 1, 2, 3, 4, 5]
concepts = ['TC', 'UE', 'OC', 'GM', 'VA', 'SV', 'CTD', 'FR']
concept_vals = [0, 1, 2, 3, 4]


class EssayDataset(Dataset):
    """Dataset wrapper for essay data prepared as CSVs with columns:

    Required columns per row:
      - text: string (full_text from essay data)
      - label: int (0-5, representing rounded score)
      - concept columns: TC, UE, OC, GM, VA, SV, CTD, FR with numeric values
        (0, 1, 2)

    Files: train.csv / dev.csv / test.csv
    """

    def __init__(
        self,
        split: str,
        tokenizer,
        max_len: int,
        max_samples_ratio: Optional[float] = None,
        random_seed: int = 42,
        num_concepts: Optional[int] = None,
        max_samples: Optional[int] = None,
    ):
        assert split in ("train", "val", "test"), f"Unsupported split: {split}"

        self.tokenizer = tokenizer
        self.max_len = max_len

        # Dataset metadata
        self.final_label = ['score']
        self.final_label_vals = [0, 1, 2, 3, 4, 5]
        self.concepts = ['TC', 'UE', 'OC', 'GM', 'VA', 'SV', 'CTD', 'FR']
        if num_concepts is not None:
            self.concepts = self.concepts[:num_concepts]
        self.concept_vals = [0, 1, 2, 3, 4]

        # Resolve repo root and dataset directory to avoid CWD dependence
        SELF_DIR = os.path.dirname(os.path.abspath(__file__))
        MAIN_DIR = os.path.dirname(SELF_DIR)
        ROOT_DIR = os.path.dirname(MAIN_DIR)
        ESSAY_DIR = os.path.join(ROOT_DIR, "dataset", "essay")

        # Map 'val' split to files named with 'dev' prefix
        split_prefix = 'dev' if split == 'val' else split
        csv_path = os.path.join(ESSAY_DIR, f"{split_prefix}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"EssayDataset cannot find file: {csv_path}")

        df = pd.read_csv(csv_path)

        # Mandatory columns
        required_cols = ["text", "label"] + ESSAY_CONCEPT_COLUMNS
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' in {csv_path}")

        # Basic fields
        self.data = df
        self.text = df["text"].astype(str)
        self.labels = df["label"].astype(int)

        # Use concept values directly as they are already numeric
        self.concept_data = {}
        for c in self.concepts:
            self.concept_data[c] = df[c].astype(int)

        # Initialize indices with all rows
        all_indices = list(range(len(self.labels)))
        split_seed = hash(f"{random_seed}_{split}") % (2**32)

        # max_samples (absolute) overrides max_samples_ratio
        if max_samples is not None:
            num_samples = min(max_samples, len(all_indices))
            rng = np.random.default_rng(split_seed)
            self.indices = rng.choice(all_indices, size=num_samples, replace=False).tolist()
        # Apply max_samples_ratio limit if specified
        elif max_samples_ratio is not None and 0.0 < max_samples_ratio <= 1.0:
            num_samples = int(len(all_indices) * max_samples_ratio)
            if num_samples < len(all_indices):
                rng = random.Random(split_seed)
                self.indices = rng.sample(all_indices, num_samples)
            else:
                self.indices = all_indices
        else:
            # If ratio is None or invalid, use all data
            self.indices = all_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int):
        i = self.indices[index]
        text = self.text[i]
        label = int(self.labels[i])

        # Build concept label vector in fixed order
        concept_values = [int(self.concept_data[c][i]) for c in self.concepts]

        enc = self.tokenizer.encode_plus(
            str(text), add_special_tokens=True, max_length=self.max_len,
            truncation=True, padding="max_length", return_attention_mask=True, return_tensors="pt"
        )

        item = {
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
            "concept_labels": torch.tensor(concept_values, dtype=torch.long),
        }

        # Also expose per-concept fields by plain name and with _concept suffix
        for idx, c in enumerate(self.concepts):
            item[c] = torch.tensor(concept_values[idx], dtype=torch.long)
            item[f"{c}_concept"] = item[c]

        return item

