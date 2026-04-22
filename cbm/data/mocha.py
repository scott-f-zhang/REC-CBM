import os
import json
from typing import List, Dict, Any, Optional, Tuple
import torch
from torch.utils.data import Dataset
import numpy as np
import random


class MOCHADataset(Dataset):
    """
    Minimal MOCHA wrapper for the standard pipeline (no concept supervision).
    - Inputs: concatenation of context/question/reference/candidate
    - Labels: discrete 'score' values (1-5) mapped to 0-indexed classes (0-4)
    """

    def __init__(
        self,
        split: str,
        tokenizer,
        max_len: int,
        bins: int = 5,
        base_dir: Optional[str] = None,
        max_samples_ratio: Optional[float] = None,
        random_seed: int = 42,
        num_concepts: Optional[int] = None,
        max_samples: Optional[int] = None,
    ):
        assert split in ("train", "val", "test"), f"Unsupported split: {split}"
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.bins = bins

        # Resolve repo root and expected dataset locations
        SELF_DIR = os.path.dirname(os.path.abspath(__file__))
        MAIN_DIR = os.path.dirname(SELF_DIR)
        ROOT_DIR = os.path.dirname(MAIN_DIR)
        DEFAULT_DIR = os.path.join(ROOT_DIR, "dataset", "mocha")
        self.data_dir = base_dir or DEFAULT_DIR

        # Expected layout: dataset/mocha/{train,dev,test}.json
        train_path = os.path.join(self.data_dir, "train.json")
        dev_path = os.path.join(self.data_dir, "dev.json")
        test_path = os.path.join(self.data_dir, "test.json")
        for p in (train_path, dev_path, test_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"MOCHA file not found: {p}")

        # Load JSON
        with open(train_path, "r") as f:
            train_obj = json.load(f)
        with open(dev_path, "r") as f:
            dev_obj = json.load(f)
        with open(test_path, "r") as f:
            test_obj = json.load(f)

        # Normalize objects to flat lists
        train_items: List[Dict[str, Any]] = self._json_to_items(train_obj)
        dev_items: List[Dict[str, Any]] = self._json_to_items(dev_obj)
        test_items: List[Dict[str, Any]] = self._json_to_items(test_obj)

        # Select items per split
        if split == "train":
            self.items = train_items
        elif split == "val":
            self.items = dev_items
        else:
            self.items = test_items

        # MOCHA score field is already discrete (1-5), so we can map directly
        # If bins != 5, we still use direct mapping but validate the range
        # Note: Original scores are 1-5, we map to 0-indexed (0-4) for bins=5
        self.use_direct_mapping = True
        self._edges = None  # Not needed for direct mapping

        # Dataset metadata
        self.final_label = ['quality']
        self.final_label_vals = list(range(self.bins))  # 0..bins-1
        # Use MOCHA gemini_evaluation concept dimensions
        self.concepts = ['RQ', 'IA', 'TG', 'CR', 'PQ', 'C', 'CC']
        if num_concepts is not None:
            self.concepts = self.concepts[:num_concepts]
        # Map discrete values (1,2,3) to 0-indexed classes (0,1,2)
        self.concept_vals = [0, 1, 2]

        # Initialize indices with all items
        all_indices = list(range(len(self.items)))
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

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        i = self.indices[index]
        ex = self.items[i]
        text = self._compose_text(ex)
        enc = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        label_idx = self._score_to_bin(self._extract_score(ex))
        concept_labels = self._extract_concept_labels(ex)
        item = {
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "label": torch.tensor(label_idx, dtype=torch.long),
            "concept_labels": torch.tensor(concept_labels, dtype=torch.long),
        }
        # Also expose per-concept fields by plain name (e.g. batch['RQ'])
        for idx, c in enumerate(self.concepts):
            item[c] = torch.tensor(concept_labels[idx], dtype=torch.long)
        return item

    def _json_to_items(self, obj: Any) -> List[Dict[str, Any]]:
        """
        Normalize MOCHA json object to a flat list of examples.
        Supported forms:
          - list of examples
          - {"data": [...]}
          - {dataset_name: {id: example_dict, ...}, ...}
        """
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            if "data" in obj and isinstance(obj["data"], list):
                return obj["data"]
            # Flatten nested dataset → id → example shape
            flat: List[Dict[str, Any]] = []
            for dataset_name, group in obj.items():
                if not isinstance(group, dict):
                    continue
                for ex_id, ex in group.items():
                    if isinstance(ex, dict):
                        ex_norm = dict(ex)
                        # keep provenance
                        ex_norm.setdefault("_dataset", dataset_name)
                        ex_norm.setdefault("_id", ex_id)
                        flat.append(ex_norm)
            if flat:
                return flat
        # Fallback: wrap as single-element list if unknown structure
        return [obj]

    def _extract_score(self, ex: Dict[str, Any]) -> float:
        # MOCHA 'score' as continuous label
        val = ex.get("score", 0.0)
        try:
            return float(val)
        except Exception:
            return 0.0

    def _compose_text(self, ex: Dict[str, Any]) -> str:
        context = str(ex.get("context", "")).strip()
        question = str(ex.get("question", "")).strip()
        reference = str(ex.get("reference", "")).strip()
        candidate = str(ex.get("candidate", "")).strip()
        # Concatenate segments with clear separators
        parts = []
        if context:
            parts.append(f"[CTX] {context}")
        if question:
            parts.append(f"[Q] {question}")
        if reference:
            parts.append(f"[REF] {reference}")
        if candidate:
            parts.append(f"[CAND] {candidate}")
        return " [SEP] ".join(parts) if parts else ""

    def _score_to_bin(self, score: float) -> int:
        # MOCHA score is discrete (1-5), map directly to 0-indexed classes
        # Score 1 -> Class 0, Score 2 -> Class 1, ..., Score 5 -> Class 4
        score_int = int(score)
        # Map 1-5 to 0-4
        idx = score_int - 1
        # Clamp into 0..bins-1 to handle edge cases or different bin counts
        if idx < 0:
            idx = 0
        if idx >= self.bins:
            idx = self.bins - 1
        return idx

    def _extract_concept_labels(self, ex: Dict[str, Any]) -> List[int]:
        """
        Extract 7 concept labels from ex['gemini_evaluation'] with mapping to 3 classes:
          - Original values are discrete: 1, 2, 3
          - Map to 0-indexed classes: 1 -> 0, 2 -> 1, 3 -> 2
        Missing/invalid values default to 1 (middle class).
        """
        gev = ex.get("gemini_evaluation", {}) or {}
        labels: List[int] = []
        for k in self.concepts:
            v = gev.get(k, None)
            try:
                iv = int(float(v))
                # Map discrete values 1,2,3 to 0-indexed classes 0,1,2
                if iv == 1:
                    cls = 0
                elif iv == 2:
                    cls = 1
                elif iv == 3:
                    cls = 2
                else:
                    # Fallback for unexpected values: map to middle class
                    cls = 1
            except Exception:
                cls = 1
            labels.append(int(cls))
        return labels
