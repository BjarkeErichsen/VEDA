# VEDA-E

This directory contains the cleaned VEDA-E codebase and the selected model
weights used for the QM9 and GEOM-Drugs experiments. It is intended as a
small, runnable checkpoint package rather than a complete dataset archive.

The repository contains implementations inherited from the original DDPM,
EDM, and consistency-model experiments. Consistency-model code is retained
for reference, but it is not part of the main VEDA-E reproduction path.

## Environment

The experiments were run with the following conda environment:

```bash
conda activate /home/phz24002/anaconda3/envs/edm
cd /home/phz24002/Drug_Discovery/VEDA/VEDA-E
```

Run training and evaluation on a GPU node. VEDA-E supports one GPU at a time;
the commands below use `CUDA_VISIBLE_DEVICES=0` to make that choice explicit.
The QM9 and GEOM training runs use the W&B projects `veda_qm9` and
`veda_geom`, respectively. Add `--no_wandb` if logging is not needed.

The pinned runtime dependencies are listed in `requirements.txt`. The legacy
`requirements_geoldm.txt` file now points to the same dependency set.

## Included checkpoints

Only the two selected experiment directories are kept. Each directory
contains the saved training arguments and two model files:

| Dataset | Directory | Recommended file |
|---|---|---|
| QM9 | `outputs/qm9_veda_e/` | `generative_model_ema.npy` |
| GEOM-Drugs | `outputs/geom_drugs_veda_e/` | `generative_model_ema.npy` |

`generative_model.npy` is the non-EMA model and
`generative_model_ema.npy` is the EMA model. The EMA model is the recommended
one for evaluation. The corresponding `args.pickle` records the model and
training configuration.

Training snapshots, optimizer states, logs, generated samples, and unrelated
checkpoints were intentionally removed from this cleaned copy.

## Dataset requirements

The raw and processed molecular datasets are **not included** in this
directory. The standard GEOM-Drugs split permutation is included at
`data/geom/geom_permutation.npy`; the compatible `geom_drugs_30.npy` file is
still required separately.

The VEDA-E dataset pipeline and dataset conventions largely follow the
[e3_diffusion_for_molecules](https://github.com/ehoogeboom/e3_diffusion_for_molecules/tree/main)
implementation. Use its dataset preparation instructions as the reference for
restoring the raw and processed QM9 and GEOM-Drugs data before running VEDA-E.

### QM9

The processed QM9 data must be restored under the path expected by the code:

```text
qm9/temp/qm9/
```

The default QM9 data root is `qm9/temp`. The current copy keeps the QM9
loader and preprocessing code, but `qm9/temp/qm9` is empty by design.

### GEOM-Drugs

The training and evaluation entry points accept the processed file through
`--data_file`:

```text
/path/to/geom_drugs_30.npy
```

This large external dataset is not copied into VEDA-E. For example, pass the
path when evaluating the GEOM-Drugs checkpoint:

```bash
python eval_analyze.py \
  --model_path outputs/geom_drugs_veda_e \
  --model_file generative_model_ema.npy \
  --data_file /path/to/geom_drugs_30.npy \
  --n_samples 1000 \
  --batch_size_gen 100 \
  --skip_nll True \
  --sampler test \
  --steps 50 \
  --use_heun False
```

## Evaluation

After restoring the relevant dataset, use the EMA checkpoint. For example,
the following runs the Euler branch with the current `test` schedule:

```bash
# QM9
CUDA_VISIBLE_DEVICES=0 python eval_analyze.py \
  --model_path outputs/qm9_veda_e \
  --model_file generative_model_ema.npy \
  --n_samples 5000 \
  --batch_size_gen 100 \
  --skip_nll True \
  --sampler test \
  --steps 50 \
  --use_heun False

# GEOM-Drugs
CUDA_VISIBLE_DEVICES=0 python eval_analyze.py \
  --model_path outputs/geom_drugs_veda_e \
  --model_file generative_model_ema.npy \
  --data_file /path/to/geom_drugs_30.npy \
  --n_samples 1000 \
  --batch_size_gen 100 \
  --skip_nll True \
  --sampler test \
  --steps 50 \
  --use_heun False
```

Evaluation writes generated molecules and analysis files below the selected
model directory. These outputs are not included in the cleaned package.

### Sampler note

In `equivariant_diffusion/en_consistency.py`:

- `sampler=test` uses the VEDA-E test schedule with `rho=2.5`.
- `--use_heun False` selects Euler; `--use_heun True` enables the Heun
  correction.
- `sampler=test_CM` uses the CM-specific branch with `rho=2` and is not the
  main VEDA-E evaluation path.

The sampler choice and its `rho` value are source-code settings, not fields
stored in the checkpoint. If a strict `rho=2 + Euler` evaluation is needed,
make that scheduler change explicitly before running evaluation.

## Training configuration

The retained `args.pickle` files are the authoritative configuration records.
Their main settings are:

| Dataset | Model | Epochs | Batch size | Learning rate | Layers | Features |
|---|---|---:|---:|---:|---:|---:|
| QM9 | elucidating DM, EDM, `x_pred` | 2000 | 64 | 3e-4 | 9 | 256 |
| GEOM-Drugs | elucidating DM, EDM, `x_pred` | 10 | 16 | 1e-3 | 4 | 256 |

Both use 1000 diffusion steps, EMA decay `0.9999`, normalization factors
`[1, 4, 10]`, and a cosine learning-rate scheduler. QM9 includes charges;
GEOM-Drugs does not. The GEOM-Drugs checkpoint was produced from a resumed
run, but the earlier resume directory and training snapshots are not part of
this cleaned copy.

To launch the two training configurations from scratch, run the following
commands from this directory after restoring the corresponding datasets:

```bash
# QM9
CUDA_VISIBLE_DEVICES=0 python main_qm9.py \
  --exp_name elu_dm3e4_x_pred_lrscheduler \
  --dataset qm9 \
  --datadir qm9/temp \
  --probabilistic_model elucidating_dm \
  --predictor x_pred \
  --n_epochs 2000 \
  --batch_size 64 \
  --lr 3e-4 \
  --n_layers 9 \
  --nf 256 \
  --lr_scheduler cosine \
  --warmup_epochs 20 \
  --normalize_factors '[1, 4, 10]' \
  --n_stability_samples 100 \
  --compute_fcd True

# GEOM-Drugs
CUDA_VISIBLE_DEVICES=0 python main_geom_drugs.py \
  --exp_name geom_eludm_base_x_pred_lrscheduler1en3 \
  --data_file /path/to/geom_drugs_30.npy \
  --probabilistic_model elucidating_dm \
  --predictor x_pred \
  --n_epochs 10 \
  --batch_size 16 \
  --lr 1e-3 \
  --n_layers 4 \
  --nf 256 \
  --lr_scheduler cosine \
  --warmup_epochs 0.2 \
  --n_stability_samples 100 \
  --visualize_every_batch 30000
```

The commands above use the main settings recorded in the corresponding
`args.pickle` files. The GEOM-Drugs command starts a new run; the original
checkpoint was produced by a resumed run whose earlier training state is not
included in this cleaned copy. Multi-GPU `DataParallel` is intentionally not
used in this cleaned reproduction path.

## Main files

- `main_qm9.py` — QM9 training entry point.
- `main_geom_drugs.py` — GEOM-Drugs training entry point.
- `eval_analyze.py` — sampling and molecular evaluation.
- `equivariant_diffusion/en_consistency.py` — EDM/VEDA-E and CM sampling
  implementations.
- `qm9/dataset.py` — dataset loading and preprocessing utilities.
