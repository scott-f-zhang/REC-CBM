"""Level 1 + 2 smoke test for REC-CBM.

Run:
    python tests/smoke.py

Exit 0 on success, 1 on first failure. No pytest dependency.

Level 1 = static (imports, CLI --help).
Level 2 = no-GPU forward smoke (config, hyperparams lookup, LR lookup,
          model factory shapes, residual-head zero-init, dataset schemas).
"""
import os
import subprocess
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def section(name: str) -> None:
    print(f"\n=== {name} ===")


def check(cond: bool, msg: str) -> None:
    status = "ok" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        sys.exit(1)


# ----- Level 1 ---------------------------------------------------------------

def level1_imports() -> None:
    section("Level 1: imports")
    import main  # noqa: F401
    from cbm import save_result_row, train_one  # noqa: F401
    from cbm.training import trainer, lcec, epoch_loops  # noqa: F401
    from cbm.models import models, lcec_residual, pretrained_loaders  # noqa: F401
    from cbm.data import EssayDataset, QADataset, MOCHADataset  # noqa: F401
    from cbm.utils import get_method_params, set_seed, normalize_result  # noqa: F401
    check(True, "all modules importable")

    check(main.DATASETS == ("qa", "essay", "mocha"), f"main.DATASETS={main.DATASETS}")
    check(len(main.MODELS) == 5, f"main.MODELS has 5 entries, got {len(main.MODELS)}")
    check(callable(main.run_2stage), "main.run_2stage is callable")
    check(callable(train_one), "cbm.train_one is callable")
    check(callable(save_result_row), "cbm.save_result_row is callable")


def level1_cli() -> None:
    section("Level 1: CLI --help")
    result = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "main.py"), "--help"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    check(result.returncode == 0, f"main.py --help exits 0 (got {result.returncode})")
    for token in ("--dataset", "--model", "--seed"):
        check(token in result.stdout, f"help text mentions '{token}'")


# ----- Level 2 ---------------------------------------------------------------

def level2_hyperparams_coverage() -> None:
    section("Level 2: hyperparams coverage")
    import inspect

    from cbm.training.lcec import STAGE2_PARAMS
    from cbm.training.trainer import VALID_METHODS, train_one
    from cbm.utils import get_method_params

    assert VALID_METHODS == {"CONCEPT_ENCODER"}, f"VALID_METHODS={VALID_METHODS}"
    check(True, f"VALID_METHODS = {sorted(VALID_METHODS)}")

    sig = inspect.signature(train_one)
    for required in ("dataset", "model_name", "method", "seed", "save_root"):
        assert required in sig.parameters, f"train_one missing required kwarg '{required}'"
    check(True, "train_one signature has required kwargs")

    datasets = ("qa", "essay", "mocha")
    models = (
        "bert-base-uncased", "gpt2", "roberta-base", "t5-base", "facebook/bart-base",
    )

    for dataset in datasets:
        for model in models:
            hp = get_method_params(dataset, model, "CONCEPT_ENCODER")
            assert "lr" in hp and float(hp["lr"]) > 0, (
                f"missing/invalid lr for {dataset}/{model}"
            )
            assert "attention_temperature" in hp, (
                f"missing attention_temperature for {dataset}/{model}"
            )
            assert "lambda_rank" in hp, (
                f"missing lambda_rank for {dataset}/{model}"
            )
    check(True, f"hyperparams.json covers {len(datasets) * len(models)} combos (CONCEPT_ENCODER)")

    for dataset in datasets:
        for model in models:
            s2 = STAGE2_PARAMS[dataset][model]
            for key in ("lr", "lambda_den", "lambda_spa"):
                assert key in s2, f"STAGE2_PARAMS missing {key} for {dataset}/{model}"
            assert float(s2["lr"]) > 0, f"STAGE2_PARAMS lr<=0 for {dataset}/{model}"
    check(True, f"STAGE2_PARAMS covers {len(datasets) * len(models)} (dataset, model) combos")


def level2_models() -> None:
    section("Level 2: model factories + forward pass")
    import torch

    from cbm.models.lcec_residual import LCECResidualHead
    from cbm.models.models import (
        GradePredictorMLP,
        build_concept_encoder,
    )

    K, C, Y, D, B, T = 8, 3, 6, 768, 2, 10

    m = build_concept_encoder(
        num_concept_classes=C, num_task_classes=Y, num_concepts=K,
        is_bottleneck_mode=True, num_concept_classes_per_concept=C,
        use_relu=False, use_sigmoid=False, hidden_dimension=D,
    )
    x = torch.randn(B, T, D)
    mask = torch.ones(B, T, dtype=torch.long)
    out = m(x, mask)
    check(len(out) == K + 1, f"build_concept_encoder returns K+1={K + 1} tensors, got {len(out)}")
    check(out[0].shape == (B, Y), f"task logits shape = {(B, Y)}, got {tuple(out[0].shape)}")
    check(out[1].shape == (B, C), f"concept logits shape = {(B, C)}, got {tuple(out[1].shape)}")

    # Residual head: correction is zero-initialised, so residual output ≡ base output.
    base = GradePredictorMLP(input_dim=K * C, num_task_classes=Y)
    residual = LCECResidualHead(
        base_task_head=base,
        num_concepts=K, num_concept_classes=C, num_task_classes=Y, eps=1e-4,
    )
    concept_logits_list = [torch.randn(B, C) for _ in range(K)]
    with torch.no_grad():
        stacked = torch.stack(concept_logits_list, dim=0).transpose(0, 1).reshape(B, K * C)
        base_out = base(stacked)
        residual_out = residual(concept_logits_list)
    check(
        torch.allclose(base_out, residual_out, atol=1e-5),
        "LCECResidualHead epoch-0 output equals base head (zero-init correction)",
    )


def level2_datasets() -> None:
    section("Level 2: dataset schemas")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    except Exception as e:
        print(f"  [skip] tokenizer unavailable ({type(e).__name__}: {e})")
        return

    from cbm.data import EssayDataset, MOCHADataset, QADataset

    for name, cls in (("qa", QADataset), ("essay", EssayDataset), ("mocha", MOCHADataset)):
        ds = cls("train", tokenizer, max_len=128, max_samples=2)
        assert len(ds) > 0, f"{name}: empty dataset"
        item = ds[0]
        for key in ("input_ids", "attention_mask", "label", "concept_labels"):
            assert key in item, f"{name}: missing key '{key}'"
        K = item["concept_labels"].shape[-1]
        assert K == len(ds.concepts), (
            f"{name}: concept_labels width {K} != len(concepts) {len(ds.concepts)}"
        )
        first_concept = ds.concepts[0]
        assert first_concept in item, f"{name}: per-concept alias '{first_concept}' missing"
        check(True, f"{name}: batch keys + concept count OK (K={K})")


# ----- Entry point -----------------------------------------------------------

def main_runner() -> None:
    for fn in (
        level1_imports,
        level1_cli,
        level2_hyperparams_coverage,
        level2_models,
        level2_datasets,
    ):
        fn()
    print("\n=== ALL SMOKE CHECKS PASSED ===")


if __name__ == "__main__":
    main_runner()
