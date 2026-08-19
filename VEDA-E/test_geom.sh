#!/bin/bash

MODEL_PATH="outputs/geom_drugs_veda_e/"
DATA_FILE="/path/to/geom_drugs_30.npy"
N_SAMPLES=1000
SKIP_NLL=True
MODEL_FILE="generative_model_ema.npy"
SAMPLER=test
ALPHA=1

for steps in 50; do
    echo "Running with steps=$steps"
    CUDA_VISIBLE_DEVICES=0 python -x -u eval_analyze.py \
        --model_path $MODEL_PATH \
        --n_samples $N_SAMPLES \
        --skip_nll $SKIP_NLL \
        --steps $steps \
        --model_file $MODEL_FILE \
        --sampler $SAMPLER \
	--data_file "$DATA_FILE" \
	--alpha $ALPHA
done
