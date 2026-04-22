"""Residual-correction variant of LCECHead.

Keeps the existing Stage I concept encoder's grade predictor (frozen) as a
baseline, and uses the LCEC μ_post pathway only to produce a zero-initialized
additive correction. At init the correction contributes nothing, so Stage II
starts with task F1 equal to the CONCEPT_ENCODER baseline. Training pushes the
correction non-zero only when the denoising pathway actually helps.
"""
import torch
import torch.nn as nn

from .models import LCECHead


class LCECResidualHead(LCECHead):
    """Residual-correction LCEC head.

    ``forward(concept_logits_list) = base_task_head(concept_logits_list)
                                    + correction_head(concat[raw, μ_post])``

    where ``correction_head`` is zero-initialized so epoch-0 output equals
    ``base_task_head`` exactly.
    """

    def __init__(
        self,
        base_task_head: nn.Module,
        num_concepts: int,
        num_concept_classes: int,
        num_task_classes: int,
        eps: float = 1e-4,
    ):
        # Parent builds task_head as Linear(K*C + K, num_task); we then
        # zero-init it so the correction path contributes 0 at epoch 0.
        super().__init__(
            num_concepts=num_concepts,
            num_concept_classes=num_concept_classes,
            num_task_classes=num_task_classes,
            eps=eps,
        )
        nn.init.zeros_(self.task_head.weight)
        nn.init.zeros_(self.task_head.bias)

        # Store frozen base task head.
        self.base_task_head = base_task_head
        for p in self.base_task_head.parameters():
            p.requires_grad = False
        self.base_task_head.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_task_head.eval()  # always keep baseline head in eval mode
        return self

    def forward(self, concept_logits_list: list) -> torch.Tensor:
        # Baseline path: flatten concept logits to [B, K*C] for the frozen
        # GradePredictorMLP / LinearLayer base head.
        stacked = torch.stack(concept_logits_list, dim=0).transpose(0, 1)  # [B, K, C]
        flat = stacked.reshape(stacked.size(0), self.K * self.C)           # [B, K*C]
        base_logits = self.base_task_head(flat)

        # Denoising path: replicate parent LCECHead up to μ_post.
        probs_list = [torch.softmax(l, dim=-1) for l in concept_logits_list]
        probs = torch.stack(probs_list, dim=1)                 # [B, K, C]
        c_hat = (probs * self.class_values).sum(dim=-1)        # [B, K]
        c_hat_norm = c_hat / self.C_minus_1                    # [B, K]
        A = self._get_A()                                      # [K, K]
        mu_post = c_hat_norm.float() @ A.T                     # [B, K]
        self.last_mu_post = mu_post                            # for L_den

        # Correction input = concat[raw_logits_flat, μ_post].
        raw_flat = torch.cat(concept_logits_list, dim=-1).float()  # [B, K*C]
        correction_input = torch.cat([raw_flat, mu_post], dim=-1)  # [B, K*C + K]
        correction = self.task_head(correction_input)              # [B, num_task]
        return base_logits + correction
