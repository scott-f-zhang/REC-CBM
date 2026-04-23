from typing import Tuple, Dict, List
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.models import masked_mean_pool


def _macro_f1(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    if not len(true_labels):
        return 0.0
    return float(f1_score(true_labels, pred_labels, average='macro', zero_division=0))


def _forward_head(model, head, input_ids, attention_mask, use_independent_channels: bool):
    """Shared backbone + head forward, branching on independent vs shared channels."""
    model_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    sequence_hidden_states = model_outputs.last_hidden_state
    if use_independent_channels:
        return head(sequence_hidden_states, attention_mask)
    pooled = masked_mean_pool(sequence_hidden_states, attention_mask)
    return head(pooled)


def train_one_epoch(
    model,
    head,
    data_loader: DataLoader,
    device,
    loss_fn,
    lambda_con: float,
    lambda_task: float,
    enable_rank: bool = False,
    lambda_rank: float = 0.0,
    use_independent_channels: bool = True,
    freeze_base_model: bool = True,
    verbose: bool = True,
    enable_lcec: bool = False,
    lambda_den: float = 0.0,
    lambda_spa: float = 0.0,
) -> Dict[str, float]:
    """Train one epoch for REC-CBM.

    Loss terms (paper Fig 4.1):
        L = λ_con · L_con + λ_task · L_task
            + λ_rank · L_rank   (if enable_rank)
            + λ_den · L_den + λ_spa · L_spa   (if enable_lcec)
    """
    if freeze_base_model:
        model.eval()
    else:
        model.train()
    head.train()

    total_losses: List[float] = []
    task_losses: List[float] = []
    con_losses: List[float] = []

    optimizer = loss_fn["optimizer"]
    ce = loss_fn["criterion"]
    scheduler = loss_fn.get("scheduler", None)

    for batch in tqdm(data_loader, desc="Training", unit="batch", disable=not verbose):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label = batch["label"].to(device)
        concept_labels_raw = batch["concept_labels"].to(device)
        concept_labels = torch.t(concept_labels_raw).contiguous().view(-1)

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            head_outputs = _forward_head(model, head, input_ids, attention_mask, use_independent_channels)
            concept_outputs = head_outputs[1:]
            task_outputs = head_outputs[0:1]

            con_logits = torch.cat(concept_outputs, dim=0)
            con_loss = ce(con_logits, concept_labels)
            task_loss = ce(task_outputs[0], label)
            loss = con_loss * lambda_con + task_loss * lambda_task

            if enable_rank and lambda_rank > 0.0:
                rank_losses = []
                for labels_k, logits_k in zip(concept_labels_raw.T, concept_outputs):
                    if labels_k.numel() < 2 or labels_k.max() == labels_k.min():
                        continue
                    class_values = torch.arange(
                        logits_k.size(1), device=logits_k.device, dtype=logits_k.dtype
                    )
                    probs = torch.softmax(logits_k, dim=1)
                    scores = (probs * class_values).sum(dim=1)
                    diff = scores.unsqueeze(1) - scores.unsqueeze(0)
                    mask = labels_k.unsqueeze(1) > labels_k.unsqueeze(0)
                    if mask.any():
                        rank_losses.append((-F.logsigmoid(diff)[mask]).mean())
                if rank_losses:
                    loss = loss + lambda_rank * torch.stack(rank_losses).mean()

            # ── Stage II LCEC losses (residual head) ─────────────────────────
            if enable_lcec:
                from ..models.models import LCECHead
                lcec_model = head.concept_to_task_model
                if isinstance(lcec_model, LCECHead):
                    mu_post = lcec_model.last_mu_post.float()
                    c_gold_norm = concept_labels_raw.float() / lcec_model.C_minus_1
                    if lambda_den > 0.0:
                        loss = loss + lambda_den * F.mse_loss(mu_post, c_gold_norm)
                    if lambda_spa > 0.0:
                        loss = loss + lambda_spa * lcec_model.get_omega_offdiag_l1()

        total_losses.append(loss.item())
        task_losses.append(task_loss.item())
        con_losses.append(con_loss.item())

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    return {
        'total_loss': sum(total_losses) / len(total_losses) if total_losses else 0.0,
        'task_loss': sum(task_losses) / len(task_losses) if task_losses else 0.0,
        'concept_loss': sum(con_losses) / len(con_losses) if con_losses else 0.0,
    }


def evaluate_one_epoch(
    model,
    head,
    data_loader: DataLoader,
    device,
    criterion,
    lambda_con: float,
    lambda_task: float,
    use_independent_channels: bool = True,
    verbose: bool = True,
) -> Dict[str, float]:
    """Evaluate model on validation set and compute metrics including losses."""
    model.eval()
    head.eval()
    predict_labels = np.array([])
    true_labels = np.array([])
    concept_predict_labels = np.array([])
    concept_true_labels = np.array([])

    total_loss_sum = 0.0
    con_loss_sum = 0.0
    task_loss_sum = 0.0
    num_batches = 0

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch in tqdm(data_loader, desc="Val", unit="batch", disable=not verbose):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)
            concept_labels_raw = batch["concept_labels"].to(device)
            concept_labels = torch.t(concept_labels_raw).contiguous().view(-1)

            head_outputs = _forward_head(model, head, input_ids, attention_mask, use_independent_channels)
            concept_outputs = head_outputs[1:]
            task_outputs = head_outputs[0:1]

            predictions = torch.argmax(task_outputs[0], axis=1)
            predict_labels = np.append(predict_labels, predictions.cpu().numpy())
            true_labels = np.append(true_labels, label.cpu().numpy())

            con_logits = torch.cat(concept_outputs, dim=0)
            concept_predictions = torch.argmax(con_logits, axis=1)
            concept_predict_labels = np.append(concept_predict_labels, concept_predictions.cpu().numpy())
            concept_true_labels = np.append(concept_true_labels, concept_labels.cpu().numpy())

            con_loss = criterion(con_logits, concept_labels)
            task_loss = criterion(task_outputs[0], label)
            loss = con_loss * lambda_con + task_loss * lambda_task

            total_loss_sum += loss.item()
            con_loss_sum += con_loss.item()
            task_loss_sum += task_loss.item()
            num_batches += 1

    if num_batches == 0:
        raise ValueError("No batches processed during validation. Cannot compute validation loss.")

    val_acc = float((predict_labels == true_labels).sum()) / float(len(true_labels)) if len(true_labels) else 0.0
    concept_acc = float((concept_predict_labels == concept_true_labels).sum()) / float(len(concept_true_labels)) if len(concept_true_labels) else 0.0

    return {
        "val_acc": val_acc,
        "val_macro_f1": _macro_f1(true_labels, predict_labels),
        "concept_acc": concept_acc,
        "concept_macro_f1": _macro_f1(concept_true_labels, concept_predict_labels),
        "val_loss": total_loss_sum / num_batches,
        "val_con_loss": con_loss_sum / num_batches,
        "val_task_loss": task_loss_sum / num_batches,
    }


def test_one_epoch(
    model,
    head,
    data_loader: DataLoader,
    device,
    use_independent_channels: bool = True,
    verbose: bool = True,
) -> Tuple[float, float, float, float]:
    model.eval()
    head.eval()
    predict_labels = np.array([])
    true_labels = np.array([])
    concept_predict_labels = np.array([])
    concept_true_labels = np.array([])

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch in tqdm(data_loader, desc="Test", unit="batch", disable=not verbose):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)
            concept_labels_raw = batch["concept_labels"].to(device)
            concept_labels = torch.t(concept_labels_raw).contiguous().view(-1)

            head_outputs = _forward_head(model, head, input_ids, attention_mask, use_independent_channels)
            task_outputs = head_outputs[0:1]
            predictions = torch.argmax(task_outputs[0], axis=1)
            predict_labels = np.append(predict_labels, predictions.cpu().numpy())
            true_labels = np.append(true_labels, label.cpu().numpy())

            concept_outputs = head_outputs[1:]
            con_logits = torch.cat(concept_outputs, dim=0)
            concept_predictions = torch.argmax(con_logits, axis=1)
            concept_predict_labels = np.append(concept_predict_labels, concept_predictions.cpu().numpy())
            concept_true_labels = np.append(concept_true_labels, concept_labels.cpu().numpy())

    test_acc = float((predict_labels == true_labels).sum()) / float(len(true_labels)) if len(true_labels) else 0.0
    concept_acc = float((concept_predict_labels == concept_true_labels).sum()) / float(len(concept_true_labels)) if len(concept_true_labels) else 0.0
    return test_acc, _macro_f1(true_labels, predict_labels), concept_acc, _macro_f1(concept_true_labels, concept_predict_labels)
