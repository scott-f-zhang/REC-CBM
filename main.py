#!/usr/bin/env python3
"""REC-CBM training — residual 2-stage pipeline (paper Fig 4.1).

Stage I trains the CONCEPT_ENCODER; Stage II freezes the backbone + concept
encoder and fits an additive LCEC correction on top (``run_stage2``). Stage I
is auto-trained on first invocation; subsequent runs reuse the cached
checkpoint under ``saved_models/``.

Examples:
    python main.py
    python main.py --dataset qa --model gpt2
    python main.py --dataset essay --model roberta-base --seed 42
"""
import argparse
import os

from cbm.training.lcec import MAIN_SAVE_ROOT, STAGE2_PARAMS, run_stage2
from cbm.training.trainer import save_result_row, train_one
from cbm.utils import safe_model_name


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_MODEL_ROOT = os.path.join(ROOT_DIR, "saved_models", "main")

DATASETS = ("qa", "essay", "mocha")
MODELS = (
    "bert-base-uncased",
    "gpt2",
    "roberta-base",
    "t5-base",
    "facebook/bart-base",
)


def _results_root(dataset: str, seed: int, subdir: str) -> str:
    return os.path.join(
        ROOT_DIR, "data", f"cbm_results_{dataset}", f"seed_{seed}", subdir,
    )


def _has_concept_encoder_checkpoint(dataset: str, model: str) -> bool:
    safe = safe_model_name(model)
    save_dir = os.path.join(MAIN_SAVE_ROOT, dataset, safe)
    backbone = os.path.join(save_dir, f"{safe}_concept_encoder.pth")
    head = os.path.join(save_dir, f"{safe}_concept_encoder_head.pth")
    return os.path.exists(backbone) and os.path.exists(head)


def run_2stage(dataset: str, model: str, seed: int) -> None:
    if _has_concept_encoder_checkpoint(dataset, model):
        print(f"[stage1] Reusing cached CONCEPT_ENCODER checkpoint for {dataset}/{model}.")
    else:
        print(f"[stage1] No CONCEPT_ENCODER checkpoint — training Stage I for {dataset}/{model}.")
        stage1 = train_one(
            dataset=dataset, model_name=model, method="CONCEPT_ENCODER",
            seed=seed, save_root=SAVED_MODEL_ROOT,
        )
        save_result_row(dataset, model, "CONCEPT_ENCODER", stage1,
                        _results_root(dataset, seed, "concept_encoder"))

    s2_hp = STAGE2_PARAMS[dataset][model]
    result = run_stage2(
        dataset=dataset,
        model_name=model,
        seed=seed,
        num_epochs=50,
        patience=3,
        lr=float(s2_hp["lr"]),
        lambda_den=float(s2_hp["lambda_den"]),
        lambda_spa=float(s2_hp["lambda_spa"]),
        save_model=True,
    )
    # run_stage2 returns the same dict shape as train_one; reuse the saver.
    save_result_row(dataset, model, "REC_CBM", result,
                    _results_root(dataset, seed, "rec_cbm_2stage"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="REC-CBM training — residual 2-stage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py\n"
            "  python main.py --dataset qa --model gpt2\n"
            "  python main.py --dataset essay --model roberta-base --seed 42\n"
        ),
    )
    parser.add_argument("--dataset", default="qa", choices=DATASETS)
    parser.add_argument("--model", default="gpt2", choices=MODELS)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    run_2stage(args.dataset, args.model, args.seed)


if __name__ == "__main__":
    main()
