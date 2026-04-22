import torch
import torch.nn as nn
from typing import Optional


def masked_mean_pool(sequence_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Masked mean pooling for token-level hidden states."""
    if sequence_hidden_states.dim() != 3:
        raise ValueError(f"Expected [B, T, D], got {sequence_hidden_states.shape}")
    if attention_mask.dim() != 2:
        raise ValueError(f"Expected [B, T], got {attention_mask.shape}")
    mask = attention_mask.unsqueeze(-1).to(sequence_hidden_states.dtype)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (sequence_hidden_states * mask).sum(dim=1) / denom


def masked_softmax(attn_logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Masked softmax: softmax over attn_logits with padding positions masked to -inf.

    Args:
        attn_logits: [B, K, T] — attention logits for K concepts over T tokens.
        mask: [B, T] — 1 for valid tokens, 0 for padding.
    """
    if attn_logits.dim() != 3:
        raise ValueError(f"Expected [B, K, T], got {attn_logits.shape}")
    mask = mask.to(dtype=attn_logits.dtype).unsqueeze(1)  # [B, 1, T]
    neg_inf = torch.finfo(attn_logits.dtype).min
    attn_logits = attn_logits.masked_fill(mask == 0, neg_inf)
    return torch.softmax(attn_logits, dim=dim)


class LinearLayer(nn.Module):
    """Fully connected linear layer wrapper."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class GradePredictorMLP(nn.Module):
    """Linear mapping from concat(concept logits) to grade logits."""

    def __init__(self, input_dim: int, num_task_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_task_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class LCECHead(nn.Module):
    """Latent Concept Error Correction head (Stage II, paper Fig 4.1(c)).

    Treats observed concept scores as noisy measurements of latent true
    concepts. Uses a Gaussian posterior with a learnable precision matrix Ω
    (concept associations) and per-concept observation noise to denoise
    before the linear task head.

    Math:
        Ω  = L_tril @ L_tril.T + eps·I       # PD precision matrix
        D  = diag(exp(log_sigma2))            # per-concept observation noise
        A  = (Ω + D⁻¹)⁻¹ @ D⁻¹               # denoising matrix [K,K]
        μ_post = ĉ_norm @ A.T                 # posterior mean [B,K]
        task_logits = task_head( concat[raw_logits_flat, μ_post] )
    """

    def __init__(
        self,
        num_concepts: int,
        num_concept_classes: int,
        num_task_classes: int,
        eps: float = 1e-4,
    ):
        super().__init__()
        K = num_concepts
        # Cholesky factor: init to I → Ω ≈ I (concepts start independent)
        self.L_tril = nn.Parameter(torch.eye(K))
        # Log observation variance per concept: init 0 → σ²=1
        self.log_sigma2 = nn.Parameter(torch.zeros(K))
        # Task head consumes concat[raw logits (K*C), μ_post (K)] to avoid
        # cold-start collapse when concept encoder is randomly initialized.
        self.task_head = nn.Linear(K * num_concept_classes + K, num_task_classes)

        self.K = K
        self.C = num_concept_classes
        self.C_minus_1 = float(max(num_concept_classes - 1, 1))
        self.eps = eps
        # Populated during forward; read by training loop for L_den.
        self.last_mu_post: Optional[torch.Tensor] = None
        self.register_buffer(
            "class_values",
            torch.arange(num_concept_classes, dtype=torch.float32),
        )

    def _get_A(self) -> torch.Tensor:
        """Compute denoising matrix A = Σ_post @ D_inv ∈ R^{K×K}.

        Forces float32 because torch.linalg.solve is numerically unstable in bfloat16.
        """
        L = torch.tril(self.L_tril).float()
        Omega = L @ L.T + self.eps * torch.eye(self.K, device=L.device, dtype=torch.float32)
        D_inv = torch.diag(torch.exp(-self.log_sigma2.float()))
        M = Omega + D_inv
        Sigma_post = torch.linalg.solve(
            M, torch.eye(self.K, device=M.device, dtype=torch.float32)
        )
        return Sigma_post @ D_inv

    def get_omega_offdiag_l1(self) -> torch.Tensor:
        """L1 norm of off-diagonal entries of L_tril for Ω sparsity (L_spa)."""
        L = torch.tril(self.L_tril)
        off_diag_mask = ~torch.eye(self.K, dtype=torch.bool, device=L.device)
        return L[off_diag_mask].abs().sum()

    def forward(self, concept_logits_list: list) -> torch.Tensor:
        """Args: concept_logits_list — list of K tensors, each [B, C].

        Side-effect: self.last_mu_post ← μ_post [B, K] (gradient retained).
        """
        # softmax → expected score ĉ in [0, C-1] → normalise to [0, 1]
        probs_list = [torch.softmax(l, dim=-1) for l in concept_logits_list]
        probs = torch.stack(probs_list, dim=1)            # [B, K, C]
        c_hat = (probs * self.class_values).sum(dim=-1)   # [B, K]
        c_hat_norm = c_hat / self.C_minus_1               # [B, K]

        A = self._get_A()                                 # [K, K]
        mu_post = c_hat_norm.float() @ A.T                # [B, K]
        self.last_mu_post = mu_post

        logits_flat = torch.cat(concept_logits_list, dim=-1).float()   # [B, K*C]
        task_input = torch.cat([logits_flat, mu_post], dim=-1)         # [B, K*C + K]
        return self.task_head(task_input)


class RecCBM(nn.Module):
    """REC-CBM: Rubric-Aware Concept Encoder + Grade Predictor (paper Fig 4.1)."""

    def __init__(self, concept_extraction_model, concept_to_task_model,
                 use_relu: bool = False, use_sigmoid: bool = False,
                 num_concepts: int = 0, num_concept_classes: int = 0):
        super().__init__()
        self.concept_extraction_model = concept_extraction_model
        self.concept_to_task_model = concept_to_task_model
        self.use_relu = use_relu
        self.use_sigmoid = use_sigmoid
        self.num_concepts = num_concepts
        self.num_concept_classes = num_concept_classes

    def forward_concept_to_task(self, concept_logits_list):
        """Forward pass for concept-to-task mapping stage."""
        if self.use_relu:
            concept_logits = [torch.relu(l) for l in concept_logits_list]
        elif self.use_sigmoid:
            concept_logits = [torch.sigmoid(l) for l in concept_logits_list]
        else:
            concept_logits = concept_logits_list

        if isinstance(self.concept_to_task_model, LCECHead):
            # LCEC head expects the list; handles its own flattening.
            task_logits = self.concept_to_task_model(concept_logits)
        else:
            # GradePredictorMLP / LinearLayer expect flat tensor [B, K*C].
            stacked = torch.stack(concept_logits, dim=0).transpose(0, 1)  # [B, K, C]
            flat = stacked.reshape(-1, self.num_concepts * self.num_concept_classes)
            task_logits = self.concept_to_task_model(flat)

        return [task_logits] + list(concept_logits_list)

    def forward(self, input_features, attention_mask=None):
        concept_logits_list = self.concept_extraction_model(input_features, attention_mask)
        return self.forward_concept_to_task(concept_logits_list)


class RubricAwareConceptEncoder(nn.Module):
    """Rubric-Aware Concept Encoder (Stage I, paper Fig 4.1(a)).

    Attention-based concept extraction (X → C) with independent per-concept
    queries that attend over token representations from a pretrained LM.
    """

    def __init__(
        self,
        num_concept_classes: int,
        num_concepts: int = 4,
        is_bottleneck_mode: bool = False,
        hidden_dimension: int = 768,
        use_independent_channels: bool = True,
        attention_temperature: float = 1.0,
        query_init_method: str = "random",
    ):
        """
        Args:
            num_concept_classes: Number of classes per concept.
            num_concepts: K — number of concepts to extract.
            is_bottleneck_mode: If True, only predict concepts (no main task head).
            hidden_dimension: Backbone hidden dim (default 768).
            use_independent_channels: If True, per-concept attention queries;
                                      if False, share a pooled representation.
            attention_temperature: Temperature for the attention softmax.
            query_init_method: 'random' or 'orthogonal'.
        """
        super().__init__()
        self.num_concepts = num_concepts
        self.is_bottleneck_mode = is_bottleneck_mode
        self.num_concept_classes = num_concept_classes
        self.use_independent_channels = use_independent_channels
        self.attention_temperature = attention_temperature

        self.concept_classifiers = nn.ModuleList()

        if num_concepts > 0 and use_independent_channels:
            self.concept_queries = nn.Parameter(torch.empty(num_concepts, hidden_dimension))
            if query_init_method == "orthogonal":
                nn.init.orthogonal_(self.concept_queries, gain=1.0)
            elif query_init_method == "random":
                nn.init.normal_(self.concept_queries, std=1.0)
            else:
                raise ValueError(f"Unknown query_init_method: {query_init_method}")
        else:
            self.concept_queries = None

        # Build concept classifiers
        if num_concepts > 0:
            if not is_bottleneck_mode:  # multitasking: main task head + per-concept heads
                self.concept_classifiers.append(LinearLayer(hidden_dimension, num_concept_classes))
            for _ in range(num_concepts):
                self.concept_classifiers.append(LinearLayer(hidden_dimension, num_concept_classes))
        else:
            self.concept_classifiers.append(LinearLayer(hidden_dimension, num_concept_classes))

    def forward(self, input_features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """Forward pass: produce list of concept (and optionally task) logits."""
        # Vector input [B, D] — backward-compat / shared-channel pooled path.
        if input_features.dim() == 2:
            return [cls(input_features) for cls in self.concept_classifiers]

        if input_features.dim() != 3:
            raise ValueError(f"Expected [B, T, D] or [B, D], got {input_features.shape}")

        sequence_representation = input_features

        if not self.use_independent_channels:
            if attention_mask is None:
                raise ValueError("attention_mask is required for pooling in shared channel mode")
            pooled = masked_mean_pool(sequence_representation, attention_mask)
            return [cls(pooled) for cls in self.concept_classifiers]

        if attention_mask is None:
            raise ValueError("attention_mask is required for independent channel extraction")

        # Independent channels: per-concept attention over the token sequence.
        if self.num_concepts == 0:
            pooled = masked_mean_pool(sequence_representation, attention_mask)
            return [cls(pooled) for cls in self.concept_classifiers]

        attn_logits = torch.einsum(
            "btd,kd->bkt", sequence_representation, self.concept_queries
        ) / self.attention_temperature
        attn_weights = masked_softmax(attn_logits, attention_mask, dim=-1)  # [B, K, T]
        concept_representations = torch.einsum(
            "bkt,btd->bkd", attn_weights, sequence_representation
        )  # [B, K, D]

        concept_logits_list = []
        if not self.is_bottleneck_mode:
            # Main task head reads the plain masked-mean pooled representation.
            shared_pooled = masked_mean_pool(sequence_representation, attention_mask)
            concept_logits_list.append(self.concept_classifiers[0](shared_pooled))

        for k in range(self.num_concepts):
            idx = k + (0 if self.is_bottleneck_mode else 1)
            concept_logits_list.append(
                self.concept_classifiers[idx](concept_representations[:, k, :])
            )
        return concept_logits_list


# ============================================================================
# Factory functions
# ============================================================================


def build_concept_encoder(
    num_concept_classes: int,
    num_task_classes: int,
    num_concepts: int,
    is_bottleneck_mode: bool,
    num_concept_classes_per_concept: int,
    use_relu: bool,
    use_sigmoid: bool,
    hidden_dimension: int = 768,
    use_independent_channels: bool = True,
    attention_temperature: float = 1.0,
    query_init_method: str = "random",
) -> RecCBM:
    """Build REC-CBM Stage I: concept encoder + linear grade predictor (used by CONCEPT_ENCODER)."""
    concept_extraction_model = RubricAwareConceptEncoder(
        num_concept_classes=num_concept_classes,
        num_concepts=num_concepts,
        is_bottleneck_mode=is_bottleneck_mode,
        hidden_dimension=hidden_dimension,
        use_independent_channels=use_independent_channels,
        attention_temperature=attention_temperature,
        query_init_method=query_init_method,
    )
    input_dim = num_concepts * num_concept_classes_per_concept if num_concept_classes_per_concept else num_concepts
    concept_to_task_model = GradePredictorMLP(input_dim=input_dim, num_task_classes=num_task_classes)
    return RecCBM(
        concept_extraction_model=concept_extraction_model,
        concept_to_task_model=concept_to_task_model,
        use_relu=use_relu,
        use_sigmoid=use_sigmoid,
        num_concepts=num_concepts,
        num_concept_classes=num_concept_classes_per_concept,
    )


def build_rec_cbm(
    num_concept_classes: int,
    num_task_classes: int,
    num_concepts: int,
    hidden_dimension: int = 768,
    use_independent_channels: bool = True,
    attention_temperature: float = 1.0,
    query_init_method: str = "orthogonal",
    lcec_eps: float = 1e-4,
) -> RecCBM:
    """Build full REC-CBM: Stage I concept encoder + Stage II LCEC head (used by REC_CBM)."""
    concept_extraction_model = RubricAwareConceptEncoder(
        num_concept_classes=num_concept_classes,
        num_concepts=num_concepts,
        is_bottleneck_mode=True,
        hidden_dimension=hidden_dimension,
        use_independent_channels=use_independent_channels,
        attention_temperature=attention_temperature,
        query_init_method=query_init_method,
    )
    lcec_head = LCECHead(
        num_concepts=num_concepts,
        num_concept_classes=num_concept_classes,
        num_task_classes=num_task_classes,
        eps=lcec_eps,
    )
    return RecCBM(
        concept_extraction_model=concept_extraction_model,
        concept_to_task_model=lcec_head,
        num_concepts=num_concepts,
        num_concept_classes=num_concept_classes,
    )
