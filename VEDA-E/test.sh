#!/bin/bash

MODEL_PATH="outputs/qm9_veda_e"
N_SAMPLES=1000
SKIP_NLL=True
MODEL_FILE="generative_model_ema.npy"
SAMPLER=test
ALPHA=1.0
USE_HEUN=True
FCD=True

for steps in 10 15 20 25 30 50 100; do
    echo "Running with steps=$steps"
    CUDA_VISIBLE_DEVICES=1 python -x -u eval_analyze.py \
        --model_path $MODEL_PATH \
        --n_samples $N_SAMPLES \
        --skip_nll $SKIP_NLL \
        --steps $steps \
        --model_file $MODEL_FILE \
        --sampler $SAMPLER \
	--alpha $ALPHA \
	--use_heun $USE_HEUN \
	--compute_fcd $FCD
done
