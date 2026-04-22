# REC-CBM — Rubric-aware concept encoder with Latent Concept Error Correction

REC-CBM is a two-stage concept-bottleneck framework for rubric-graded text.

- **Stage I — Concept Encoder** (paper Fig 4.1(a)+(b)): a Rubric-Aware Concept
  Encoder (attention over a concept query bank) trained with an ordinal
  pairwise calibration objective (`L_rank`) in addition to per-concept
  classification (`L_con`).
- **Stage II — Grade Predictor** (paper Fig 4.1(c)): a Latent Concept Error
  Correction (LCEC) head treats the observed concept scores as noisy
  measurements of latent true concepts, learning a per-concept observation
  noise and a precision matrix Ω, and producing a denoised posterior mean
  μ_post for the linear grade classifier. Losses: `L_task`, `L_den`, `L_spa`.

Two training strategies are provided:

- **End-to-end** — joint-train backbone + concept encoder + LCEC head from
  scratch (`train_one(method="REC_CBM")` in `cbm/training/trainer.py`).
- **Residual 2-stage** — a pre-trained Stage I checkpoint is frozen, and an
  additive LCEC correction head is trained on top (`run_stage2` in
  `cbm/training/lcec.py`).

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.9+ and PyTorch with CUDA for GPU training.

---

## Usage

```bash
# End-to-end on qa / gpt2
python main.py --dataset qa --model gpt2 --mode e2e

# Residual 2-stage on qa / gpt2 (auto-trains Stage I CONCEPT_ENCODER if absent)
python main.py --dataset qa --model gpt2 --mode 2stage
```

Supported values:

- `--dataset {qa, essay, mocha}`
- `--model {bert-base-uncased, gpt2, roberta-base, t5-base, facebook/bart-base}`
- `--mode {e2e, 2stage, both}`
- `--seed <int>` — default `123`

Results are written to:

```
data/cbm_results_<dataset>/seed_<seed>/
  concept_encoder/<dataset>/result_<dataset>.csv   # Stage I (2stage only)
  rec_cbm_e2e/<dataset>/result_<dataset>.csv       # e2e mode
  rec_cbm_2stage/<dataset>/result_<dataset>.csv    # 2stage mode
```

Per-(dataset, model, method) hyperparameters — learning rate, temperature,
λ's, query LR multiplier — all live in `config/hyperparams.json` and are
loaded automatically at train time via `cbm.utils.hyperparams.get_method_params`.
Methods recognised there: `CONCEPT_ENCODER`, `REC_CBM`.

---

## Datasets

Splits are bundled under `dataset/{qa,essay,mocha}/` and loaded via
`cbm/data/{qa,essay,mocha}.py`.

| Dataset | Concepts (K)                            | Task classes |
|---------|-----------------------------------------|--------------|
| qa      | FC, CC, TU, CP, R, DU, EE, FR (8)       | 6 (0–5)      |
| essay   | TC, UE, OC, GM, VA, SV, CTD, FR (8)     | 6 (0–5)      |
| mocha   | RQ, IA, TG, CR, PQ, C, CC (7)           | 5 (0–4)      |

---

## Repo Layout

```
main.py                         # Single entry point — e2e or residual 2-stage

cbm/
  config/defaults.py            # TrainingConfig
  models/models.py              # RubricAwareConceptEncoder, LCECHead, RecCBM,
                                # build_concept_encoder, build_rec_cbm
  models/lcec_residual.py       # LCECResidualHead (additive correction)
  training/trainer.py           # train_one(), save_result_row()
  training/lcec.py              # run_stage2() — Stage II LCEC trainer
  training/epoch_loops.py       # train / evaluate / test per epoch
  utils/hyperparams.py          # hyperparams.json loader

dataset/{qa,essay,mocha}/       # Train/dev/test splits
cbm/data/{qa,essay,mocha}.py    # Torch datasets
config/hyperparams.json         # Per-(dataset, model, method) hyperparameters
                                # (lr, attention_temperature, λ_rank, λ_den, λ_spa)
data/cbm_results_<dataset>/     # Training result CSVs (gitignored)
```

---

## License

Released under the MIT License — see `LICENSE`.

## Citation

```bibtex
@article{reccbm2026,
  title={REC-CBM: Rubric-aware Concept Encoder with Latent Concept Error Correction for Rubric-Graded Text Classification},
  author={TBD},
  journal={TBD},
  year={2026}
}
```
