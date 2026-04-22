"""Stage II (Grade Predictor) training: freeze Stage I encoder, train LCEC head.

Stage I (external): Train the concept encoder via CONCEPT_ENCODER method
(``main.py --mode 2stage`` handles this automatically if no cached checkpoint
is found). See paper Fig 4.1(a)(b).

Stage II (this module): Freeze backbone + concept encoder, attach
``LCECResidualHead``, train only the LCEC correction parameters
(``L_tril``, ``log_sigma2``, ``task_head``). See paper Fig 4.1(c).
"""
import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from ..models.lcec_residual import LCECResidualHead
from ..models.models import RecCBM
from ..models.pretrained_loaders import load_model_and_tokenizer
from ..utils import safe_model_name, set_seed
from .epoch_loops import evaluate_one_epoch, test_one_epoch, train_one_epoch
from .trainer import _load_datasets


# Stage I checkpoints are written here by the main training pipeline.
SAVED_MODEL_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "saved_models",
)
MAIN_SAVE_ROOT = os.path.join(SAVED_MODEL_ROOT, "main")


# ── Stage II (LCEC) hyperparameters ─────────────────────────────────────────
# QA uses a per-backbone hybrid: error_fix grid-search values for bert / gpt2 /
# roberta / bart, but t5-base stays on the main (v1 paper) values because t5
# is sensitive to Stage II LR — the error_fix lr=0.05 drops t5 task_f1 by
# ~0.044 vs the main lr=0.02, while the other 4 backbones gain ~0.003–0.009
# from error_fix. Essay and MOCHA use the same values across both variants.
STAGE2_PARAMS = {
    "qa": {
        "bert-base-uncased":  {"lr": 0.02,  "lambda_den": 0.1, "lambda_spa": 0.0},
        "gpt2":               {"lr": 0.005, "lambda_den": 0.0, "lambda_spa": 0.005},
        "roberta-base":       {"lr": 0.05,  "lambda_den": 0.0, "lambda_spa": 0.05},
        "t5-base":            {"lr": 0.02,  "lambda_den": 0.1, "lambda_spa": 0.0},
        "facebook/bart-base": {"lr": 0.05,  "lambda_den": 0.0, "lambda_spa": 0.0},
    },
    "essay": {
        "bert-base-uncased":  {"lr": 0.02,  "lambda_den": 0.1, "lambda_spa": 0.0},
        "gpt2":               {"lr": 0.05,  "lambda_den": 0.0, "lambda_spa": 0.005},
        "roberta-base":       {"lr": 0.005, "lambda_den": 0.0, "lambda_spa": 0.0},
        "t5-base":            {"lr": 0.02,  "lambda_den": 0.0, "lambda_spa": 0.005},
        "facebook/bart-base": {"lr": 0.005, "lambda_den": 0.2, "lambda_spa": 0.005},
    },
    "mocha": {
        "bert-base-uncased":  {"lr": 0.005, "lambda_den": 0.0, "lambda_spa": 0.005},
        "gpt2":               {"lr": 0.05,  "lambda_den": 0.1, "lambda_spa": 0.005},
        "roberta-base":       {"lr": 0.005, "lambda_den": 0.2, "lambda_spa": 0.0},
        "t5-base":            {"lr": 0.05,  "lambda_den": 0.0, "lambda_spa": 0.0},
        "facebook/bart-base": {"lr": 0.05,  "lambda_den": 0.0, "lambda_spa": 0.005},
    },
}


def _load_checkpoint(dataset: str, model_name: str, device: torch.device):
    """Load the cached CONCEPT_ENCODER backbone + head from ``MAIN_SAVE_ROOT``."""
    safe_name = safe_model_name(model_name)
    save_dir = os.path.join(MAIN_SAVE_ROOT, dataset, safe_name)
    backbone_path = os.path.join(save_dir, f"{safe_name}_concept_encoder.pth")
    head_path = os.path.join(save_dir, f"{safe_name}_concept_encoder_head.pth")
    if not (os.path.exists(backbone_path) and os.path.exists(head_path)):
        raise FileNotFoundError(
            f"CONCEPT_ENCODER checkpoint not found for {dataset}/{model_name}. "
            f"Tried: {save_dir}"
        )
    backbone = torch.load(backbone_path, map_location=device, weights_only=False)
    head = torch.load(head_path, map_location=device, weights_only=False)
    return backbone, head


def run_stage2(
    dataset: str,
    model_name: str,
    seed: int,
    num_epochs: int = 50,
    patience: int = 3,
    lr: float = 1e-3,
    batch_size: int = 8,
    max_len: int = 512,
    lambda_den: float = 0.1,
    lambda_spa: float = 0.01,
    lcec_eps: float = 1e-4,
    save_model: bool = False,
) -> dict:
    """Train a ``LCECResidualHead`` on top of a frozen Stage I encoder."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load pre-trained Stage I checkpoint ──────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Stage 2: {dataset} / {model_name} / seed={seed}")
    print(f"{'=' * 60}")

    backbone, base_head = _load_checkpoint(dataset, model_name, device)
    concept_extraction_model = base_head.concept_extraction_model
    num_concepts = base_head.num_concepts
    num_concept_classes = base_head.num_concept_classes
    print(f"  Loaded CONCEPT_ENCODER checkpoint: K={num_concepts}, C={num_concept_classes}")

    # ── Freeze backbone + concept encoder ────────────────────────────────
    for p in backbone.parameters():
        p.requires_grad = False
    for p in concept_extraction_model.parameters():
        p.requires_grad = False
    backbone.eval()

    # ── Load dataset ─────────────────────────────────────────────────────
    _, tokenizer, _ = load_model_and_tokenizer(model_name)
    train_ds, val_ds, test_ds, num_labels, _, num_each_concept_classes = _load_datasets(
        dataset, tokenizer, max_len, seed=seed, num_concepts=num_concepts,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"  Dataset: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print(f"  Task classes={num_labels}, Concept classes={num_each_concept_classes}")

    # ── Build residual LCEC head on top of frozen Stage I task head ──────
    lcec_head = LCECResidualHead(
        base_task_head=base_head.concept_to_task_model,
        num_concepts=num_concepts,
        num_concept_classes=num_concept_classes,
        num_task_classes=num_labels,
        eps=lcec_eps,
    )
    head = RecCBM(
        concept_extraction_model=concept_extraction_model,
        concept_to_task_model=lcec_head,
        num_concepts=num_concepts,
        num_concept_classes=num_concept_classes,
    )
    head.to(device)

    # Override train() so the frozen concept encoder always stays in eval mode.
    _original_train = head.train

    def _frozen_train(mode: bool = True):
        _original_train(mode)
        concept_extraction_model.eval()
        return head

    head.train = _frozen_train

    trainable = sum(p.numel() for p in lcec_head.parameters() if p.requires_grad)
    print(f"  LCEC residual head: Linear({lcec_head.task_head.in_features} -> {num_labels})")
    print(f"  Trainable params: {trainable} (L_tril, log_sigma2, task_head)")

    # ── Optimizer + scheduler ────────────────────────────────────────────
    optimizer = AdamW(lcec_head.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    num_warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps,
    )
    criterion = torch.nn.CrossEntropyLoss()
    print(f"  LR={lr}, warmup={num_warmup_steps}/{total_steps} steps")
    print(f"  lambda_den={lambda_den}, lambda_spa={lambda_spa}")

    # ── Training loop ────────────────────────────────────────────────────
    best_score = -1.0
    best_epoch = 0
    patience_counter = 0
    best_head_state = None

    for epoch in range(num_epochs):
        train_metrics = train_one_epoch(
            model=backbone,
            head=head,
            data_loader=train_loader,
            device=device,
            loss_fn={"optimizer": optimizer, "criterion": criterion, "scheduler": scheduler},
            lambda_con=0.0,
            lambda_task=1.0,
            enable_rank=False,
            lambda_rank=0.0,
            use_independent_channels=True,
            freeze_base_model=True,
            verbose=True,
            enable_lcec=True,
            lambda_den=lambda_den,
            lambda_spa=lambda_spa,
        )
        metrics = evaluate_one_epoch(
            model=backbone,
            head=head,
            data_loader=val_loader,
            device=device,
            criterion=criterion,
            lambda_con=0.0,
            lambda_task=1.0,
            use_independent_channels=True,
            verbose=True,
        )
        val_f1 = metrics["val_macro_f1"]
        print(
            f"  Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_metrics['total_loss']:.4f} | "
            f"Val Task F1: {val_f1*100:.2f}% | "
            f"Best: {best_score*100:.2f}% | Patience: {patience_counter}/{patience}"
        )

        if val_f1 > best_score:
            best_score = val_f1
            best_epoch = epoch
            patience_counter = 0
            best_head_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # ── Restore best and test ────────────────────────────────────────────
    if best_head_state is not None:
        head.load_state_dict(best_head_state)
        head.to(device)
        print(f"  Restored best model from epoch {best_epoch+1}")

    test_acc, test_f1, concept_acc, concept_f1 = test_one_epoch(
        model=backbone,
        head=head,
        data_loader=test_loader,
        device=device,
        use_independent_channels=True,
        verbose=True,
    )
    print(f"  Test: Task Acc={test_acc*100:.2f}%, Task F1={test_f1*100:.2f}%, "
          f"Concept Acc={concept_acc*100:.2f}%, Concept F1={concept_f1*100:.2f}%")

    if save_model:
        safe_name = safe_model_name(model_name)
        save_dir = os.path.join(MAIN_SAVE_ROOT, dataset, safe_name)
        os.makedirs(save_dir, exist_ok=True)
        head_path = os.path.join(save_dir, f"{safe_name}_rec_cbm_residual_head.pth")
        head.train = _original_train
        torch.save(head, head_path)
        head.train = _frozen_train
        print(f"  Saved head to {head_path}")

    return {
        "task": [(test_acc, test_f1)],
        "concept": [(concept_acc, concept_f1)],
    }
