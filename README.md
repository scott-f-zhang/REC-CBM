# REC-CBM — Rubric-aware concept encoder with Latent Concept Error Correction

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.9+ and PyTorch with CUDA for GPU training.

---

## Usage

```bash
# Residual 2-stage on qa / gpt2 (auto-trains Stage I CONCEPT_ENCODER if absent)
python main.py --dataset qa --model gpt2
```

Supported values:

- `--dataset {qa, essay, mocha}`
- `--model {bert-base-uncased, gpt2, roberta-base, t5-base, facebook/bart-base}`
- `--seed <int>` — default `123`

Results are written to:

```
data/cbm_results_<dataset>/seed_<seed>/
  concept_encoder/<dataset>/result_<dataset>.csv   # Stage I
  rec_cbm_2stage/<dataset>/result_<dataset>.csv    # Stage II
```

Stage I hyperparameters (lr, attention_temperature, λ_rank, query_lr_multiplier)
live in `config/hyperparams.json` and are loaded via
`cbm.utils.hyperparams.get_method_params`. Stage II hyperparameters
(lr, λ_den, λ_spa) live in `STAGE2_PARAMS` inside `cbm/training/lcec.py`.

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
main.py                         # Single entry point — residual 2-stage

cbm/
  models/models.py              # RubricAwareConceptEncoder, LCECHead, RecCBM,
                                # build_concept_encoder
  models/lcec_residual.py       # LCECResidualHead (additive correction)
  training/trainer.py           # train_one(), save_result_row() — Stage I
  training/lcec.py              # run_stage2() — Stage II LCEC trainer
  training/epoch_loops.py       # train / evaluate / test per epoch
  utils/hyperparams.py          # hyperparams.json loader

dataset/{qa,essay,mocha}/       # Train/dev/test splits
cbm/data/{qa,essay,mocha}.py    # Torch datasets
config/hyperparams.json         # Stage I hyperparameters per (dataset, model)
                                # (lr, attention_temperature, λ_rank)
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
