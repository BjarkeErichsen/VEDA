# VEDA

This repository contains the reference implementation of **VEDA: 3D Molecular Generation via Variance-Exploding Diffusion with Annealing** (Accepted to AAAI 2026, Oral). VEDA integrates a variance-exploding (VE) diffusion schedule with simulated-annealing-inspired noise shaping inside an SE(3)-equivariant backbone, closing the gap between flow-level sampling speed and diffusion-level chemical fidelity. For a complete description of the VE schedule, annealing scheme, and arcsin-based SNR allocation, see the preprint:

- [VEDA: 3D Molecular Generation via Variance-Exploding Diffusion with Annealing](https://arxiv.org/abs/2511.09568)

Compared to the MiDi baseline, VEDA adds residual preconditioning to better interface coordinate-predicting networks with diffusion objectives, achieving state-of-the-art valency stability and a median relaxation energy drop of 1.72 kcal/mol on GEOM-DRUGS while keeping sampling to 100 steps, as reported in the paper.

## Models and Datasets

You can download the pre-trained VEDA model weights from:
- [Google Drive: VEDA Model Weights](https://drive.google.com/drive/folders/1-1jYJ1EKOUX6FA5NrcKUAcbSdB-1mhXu?usp=sharing)

Processed datasets are available in the [Semla-Flow Google Drive folder](https://drive.google.com/drive/folders/1zZ2xkHbATYqrjgRTn6JHj_Xf7Gcpb0pW), associated with the Semla-Flow paper.

Or process the datasets yourself:

### QM9

For QM9, download the processed dataset from the [Semla-Flow Google Drive folder](https://drive.google.com/drive/folders/1zZ2xkHbATYqrjgRTn6JHj_Xf7Gcpb0pW). If you are preparing QM9 from the raw dataset, follow the download and split procedure from [MiDi](https://github.com/cvignac/MiDi). The training and evaluation scripts expect the processed files `train.smol`, `val.smol`, and `test.smol` under the dataset directory.

### GEOM-DRUGS

For GEOM-DRUGS we also follow the URLs provided in the MiDi repo. GEOM-DRUGS is preprocessed using the `preprocess.py` script. GEOM-DRUGS URLs from MiDi are as follows:

- Train: https://drive.switch.ch/index.php/s/UauSNgSMUPQdZ9v
- Validation: https://drive.switch.ch/index.php/s/YNW5UriYEeVCDnL
- Test: https://drive.switch.ch/index.php/s/GQW9ok7mPInPcIo

After downloading the raw pickle files, place them in a `raw` folder and run:
```bash
python preprocess.py --data_path path/to/geom-drugs --raw_data_folder raw --save_data_folder smol
```

## Installation

This checkout is intended to run in the existing `equinv` conda environment. The versions verified in that environment are:

- Python 3.11.11
- PyTorch 2.7.1 with CUDA 12.8 support
- RDKit 2024.09.5
- Lightning 2.5.0.post0
- Open Babel 3.1.0

Activate the environment before running any command:

  ```bash
  conda activate equinv
  ```

You can verify the main dependencies with:

  ```bash
  python -c "import torch, lightning; from rdkit import Chem; from openbabel import pybel; print(torch.__version__, torch.version.cuda, lightning.__version__)"
  ```

If a dependency is missing from the local environment, install the repository requirements after activating `equinv`:

  ```bash
  pip install -r requirements.txt
  ```

The repository is run directly from its root directory; no editable package installation is required.

## Running QM9 experiments

For the QM9 setup of VEDA-S used in the paper, we launch training with:

```bash
python train.py \
  --data_path data/qm9/smol/ \
  --dataset qm9 \
  --epochs 300 \
  --warm_up_steps 2000 \
  --bond_loss_weight 0.5 \
  --use_cat_time_based_weight \
  --x_pred_mode constant
```

This command uses the configuration we report for QM9 metrics in [the VEDA paper](https://arxiv.org/abs/2511.09568).

### Evaluation

To evaluate a trained QM9 model:

```bash
python evaluate.py \
  --data_path path/data/qm9/smol/ \
  --ckpt_path qm9/checkpoints/last.ckpt \
  --dataset qm9 \
  --coord_noise_std_dev 0.4 \
  --temperature 0.9 \
  --adaptive_cat_noise_level
```

## Running GEOM-DRUGS experiments

For GEOM-DRUGS:

```bash
python train.py \
  --data_path data/geom-drugs/smol/ \
  --dataset geom-drugs \
  --val_check_epochs 10 \
  --x_pred_mode adaptive
```

This configuration matches the GEOM-DRUGS runs reported in the paper.

### Evaluation

To evaluate a trained GEOM-DRUGS model:

```bash
python evaluate.py \
  --data_path data/geom-drugs/smol/ \
  --ckpt_path model/last.ckpt \
  --dataset geom-drugs \
  --coord_noise_std_dev 0.4 \
  --sampling_scheduler_factor_rho 2.5 \
  --adaptive_cat_noise_level
```

### GEOM-DRUG-revisited Benchmark

For GEOM-DRUG-revisited Benchmark evaluation, see our parallelized implementation: [geom-drugs-3dgen-evaluation](https://github.com/peiningzhang/geom-drugs-3dgen-evaluation).

To run the benchmark:
1. First, generate molecules using `predict.py` with the same parameters as `evaluate.py`:
   ```bash
   python predict.py \
     --data_path data/geom-drugs/smol/ \
     --ckpt_path model/last.ckpt \
     --dataset geom-drugs \
     --coord_noise_std_dev 0.4 \
     --sampling_scheduler_factor_rho 2.5 \
     --adaptive_cat_noise_level
   ```
   Note: `--save_dir` (default: `./outputs`) and `--save_file` (default: `predictions`) specify where to save the generated molecules. The output will be saved as `{save_dir}/{save_file}.sdf`.
2. Then run the energy benchmark under the directory of `geom-drugs-3dgen-evaluation`:
   ```bash
   cd geom-drugs-3dgen-evaluation
   bash -x run_energy_benchmark.sh path/of/sdf
   ```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{zhang2025veda,
  title={VEDA: 3D Molecular Generation via Variance-Exploding Diffusion with Annealing},
  author={Zhang, Peining and Bi, Jinbo and Song, Minghu},
  journal={arXiv preprint arXiv:2511.09568},
  year={2025}
}
```
