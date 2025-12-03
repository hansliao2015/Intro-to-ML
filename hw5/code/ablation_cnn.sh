#!/bin/bash

DEVICE=${1:-cpu}

OUTROOT="results_cnn_ablation"
mkdir -p $OUTROOT

TRAIN="train.csv"
TEST="test4students.csv"

for POOL in max avg; do
for BN in 1 0; do
for DO in 1 0; do
for RES in 1 0; do

    EXP="${OUTROOT}/bn${BN}_do${DO}_res${RES}_pool${POOL}"
    mkdir -p $EXP

    echo "Running $EXP with device=$DEVICE ..."

    python main.py \
        --model_type improved_cnn \
        --use_bn $BN \
        --use_dropout $DO \
        --use_residual $RES \
        --pooling $POOL \
        --seed 42 \
        --batch_size 64 \
        --train_path $TRAIN \
        --test_path $TEST \
        --device $DEVICE \
        --output_dir $EXP \
        > $EXP/log.txt

done
done
done
done
