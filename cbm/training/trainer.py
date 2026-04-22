"""REC-CBM training pipeline for CONCEPT_ENCODER and REC_CBM."""
import os
from typing import Dict, Optional

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from ..data.essay import EssayDataset
from ..data.mocha import MOCHADataset
from ..data.qa import QADataset
from ..models.models import build_concept_encoder, build_rec_cbm
from ..models.pretrained_loaders import load_model_and_tokenizer
from ..utils import normalize_result, safe_model_name, set_seed
from ..utils.hyperparams import get_method_params
from .epoch_loops import evaluate_one_epoch, test_one_epoch, train_one_epoch

VALID_METHODS = {"CONCEPT_ENCODER", "REC_CBM"}

_DATASET_CLASSES = {
    "qa": QADataset,
    "essay": EssayDataset,
    "mocha": MOCHADataset,
}


def _load_datasets(dataset: str, tokenizer, max_len: int, seed: int,
                   num_concepts: Optional[int] = None):
    if dataset not in _DATASET_CLASSES:
        raise ValueError(f"Unsupported dataset: {dataset}. Choose from {sorted(_DATASET_CLASSES)}.")
    cls = _DATASET_CLASSES[dataset]
    train_ds = cls("train", tokenizer, max_len, random_seed=seed, num_concepts=num_concepts)
    val_ds = cls("val", tokenizer, max_len, random_seed=seed, num_concepts=num_concepts)
    test_ds = cls("test", tokenizer, max_len, random_seed=seed, num_concepts=num_concepts)
    # MOCHA concept labels are always in {0, 1, 2}; the other two datasets store them as
    # a learnable set per column.
    num_each_concept_classes = 3 if dataset == "mocha" else len(train_ds.concept_vals)
    return (
        train_ds, val_ds, test_ds,
        len(train_ds.final_label_vals),
        len(train_ds.concepts),
        num_each_concept_classes,
    )


def _build_head(method: str, num_labels: int, num_concept_labels: int,
                num_each_concept_classes: int, hidden_size: int,
                attention_temperature: float):
    common = dict(
        num_concept_classes=num_each_concept_classes,
        num_task_classes=num_labels,
        num_concepts=num_concept_labels,
        hidden_dimension=hidden_size,
        use_independent_channels=True,
        attention_temperature=attention_temperature,
        query_init_method="orthogonal",
    )
    if method == "REC_CBM":
        return build_rec_cbm(**common, lcec_eps=1e-4)
    return build_concept_encoder(
        **common,
        is_bottleneck_mode=True,
        num_concept_classes_per_concept=num_each_concept_classes,
        use_relu=False,
        use_sigmoid=False,
    )


def _build_optimizer(model, head, lr: float, query_lr_multiplier: float,
                     freeze_base_model: bool, use_warmup: bool, verbose: bool):
    weight_decay = 0.01 if use_warmup else 0.0
    if freeze_base_model:
        return AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    concept_queries = getattr(model, "concept_queries", None)
    if concept_queries is not None and query_lr_multiplier != 1.0:
        query_ids = {id(concept_queries)}
        base_params = [p for p in model.parameters() if id(p) not in query_ids]
        opt = AdamW([
            {"params": base_params + list(head.parameters()), "lr": lr, "weight_decay": weight_decay},
            {"params": [concept_queries], "lr": lr * query_lr_multiplier, "weight_decay": weight_decay},
        ])
        if verbose:
            print(f"  -> Query LR: {lr * query_lr_multiplier:.2e} "
                  f"(backbone: {lr:.2e}, mult: {query_lr_multiplier}x)")
        return opt
    return AdamW(list(model.parameters()) + list(head.parameters()),
                 lr=lr, weight_decay=weight_decay)


def train_one(
    dataset: str,
    model_name: str,
    method: str,
    seed: int,
    save_root: str,
    num_epochs: int = 50,
    batch_size: int = 8,
    max_len: int = 512,
    early_stopping_patience: int = 3,
    use_warmup: bool = True,
    warmup_ratio: float = 0.1,
    freeze_base_model: bool = False,
    save_model: bool = True,
    verbose: bool = True,
    lambda_con: float = 1.0,
    lambda_task: float = 1.0,
) -> Dict:
    """Train one model on one dataset end-to-end.

    Method-specific hyperparameters (lr, attention_temperature, λ_rank,
    λ_den, λ_spa, query_lr_multiplier) are read from ``config/hyperparams.json``
    via :func:`get_method_params`; everything else is an explicit kwarg.

    Returns a dict with keys ``task``, ``concept`` (both list of 2-tuples
    ``(acc, macro_f1)``) and ``best_epoch``.
    """
    if method not in VALID_METHODS:
        raise ValueError(f"Unsupported method: {method}. Choose from {sorted(VALID_METHODS)}")

    set_seed(seed)
    hp = get_method_params(dataset, model_name, method)
    lr = float(hp["lr"])
    attention_temperature = float(hp["attention_temperature"])
    lambda_rank = float(hp["lambda_rank"])
    query_lr_multiplier = float(hp.get("query_lr_multiplier", 1.0))
    enable_lcec = method == "REC_CBM"
    lambda_den = float(hp.get("lambda_den", 0.1))
    lambda_spa = float(hp.get("lambda_spa", 0.01))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer, hidden_size = load_model_and_tokenizer(model_name)
    if freeze_base_model:
        for p in model.parameters():
            p.requires_grad = False
        if verbose:
            print("  -> Base model frozen")

    (train_ds, val_ds, test_ds, num_labels,
     num_concept_labels, num_each_concept_classes) = _load_datasets(
        dataset, tokenizer, max_len, seed,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             num_workers=4, pin_memory=True, persistent_workers=True)

    head = _build_head(method, num_labels, num_concept_labels,
                       num_each_concept_classes, hidden_size, attention_temperature)
    model.to(device)
    head.to(device)

    model_path = head_path = None
    if save_model:
        safe = safe_model_name(model_name)
        save_dir = os.path.join(save_root, dataset, safe)
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"{safe}_{method.lower()}.pth")
        head_path = os.path.join(save_dir, f"{safe}_{method.lower()}_head.pth")

    optimizer = _build_optimizer(model, head, lr, query_lr_multiplier,
                                 freeze_base_model, use_warmup, verbose)
    scheduler = None
    total_steps = len(train_loader) * num_epochs
    if use_warmup:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * warmup_ratio),
            num_training_steps=total_steps,
        )

    criterion = torch.nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    patience_counter = 0
    best_epoch = 0
    best_model_state = best_head_state = None

    for epoch in range(num_epochs):
        train_metrics = train_one_epoch(
            model, head, train_loader, device,
            {"optimizer": optimizer, "criterion": criterion, "scheduler": scheduler},
            lambda_con, lambda_task,
            enable_rank=True,
            lambda_rank=lambda_rank,
            use_independent_channels=True,
            freeze_base_model=freeze_base_model,
            verbose=verbose,
            enable_lcec=enable_lcec,
            lambda_den=lambda_den,
            lambda_spa=lambda_spa,
        )
        metrics = evaluate_one_epoch(
            model, head, val_loader, device, criterion,
            lambda_con, lambda_task,
            use_independent_channels=True,
            verbose=verbose,
        )
        val_f1 = metrics["val_macro_f1"]

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_head_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            if save_model:
                torch.save(model, model_path)
                torch.save(head, head_path)
        else:
            patience_counter += 1

        if verbose:
            print(
                f"[Epoch {epoch + 1}/{num_epochs}] "
                f"train_loss={train_metrics['total_loss']:.4f} "
                f"val_loss={metrics['val_loss']:.4f} "
                f"val_task_f1={val_f1:.4f} "
                f"val_concept_f1={metrics['concept_macro_f1']:.4f} "
                f"patience={patience_counter}/{early_stopping_patience}"
            )

        if patience_counter >= early_stopping_patience:
            if verbose:
                print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        head.load_state_dict(best_head_state)
        model.to(device)
        head.to(device)
        if verbose:
            print(f"  -> Using best model from epoch {best_epoch + 1}")

    test_acc, test_macro_f1, concept_acc, concept_macro_f1 = test_one_epoch(
        model, head, test_loader, device,
        use_independent_channels=True,
        verbose=verbose,
    )

    if verbose:
        print(f"Test Task F1 = {test_macro_f1*100:.2f}%, "
              f"Test Concept F1 = {concept_macro_f1*100:.2f}%")

    return {
        "task": [(test_acc, test_macro_f1)],
        "concept": [(concept_acc, concept_macro_f1)],
        "best_epoch": int(best_epoch),
    }


def save_result_row(dataset: str, model_name: str, method: str,
                    result: dict, results_root: str) -> str:
    """Upsert a single training result into ``<root>/<dataset>/result_<dataset>.csv``.

    Rows matching ``(dataset, model, method)`` are replaced; others are kept.
    Returns the written path.
    """
    task_scores, concept_scores = normalize_result(result)
    row = {
        "dataset": dataset,
        "model": model_name,
        "method": method,
        "score": str(task_scores),
        "concept_score": str(concept_scores),
    }
    out_dir = os.path.join(results_root, dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"result_{dataset}.csv")

    new_df = pd.DataFrame([row])
    if os.path.exists(out_file):
        existing = pd.read_csv(out_file)
        key = ["dataset", "model", "method"]
        existing = existing[~existing[key].apply(tuple, axis=1).isin(
            new_df[key].apply(tuple, axis=1))]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(out_file, index=False)
    print(f"Saved result to: {out_file}")
    return out_file
