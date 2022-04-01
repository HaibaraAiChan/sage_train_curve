#!/bin/bash

File=pseudo_mini_batch_range_products_sage.py
epoch=30
Aggre=mean
model=sage
seed=1236 
lr=0.01
dropout=0.5
fan_out=25,35,40
layers=3
Data=ogbn-products
batch_size=(98308)
hidden=64
runs=1
for bs in ${batch_size[@]}
do
        python $File \
        --dataset $Data \
        --aggre $Aggre \
        --seed $seed \
        --selection-method range \
        --batch-size $bs \
        --lr $lr \
        --num-runs $runs \
        --num-epochs $epoch \
        --num-layers $layers \
        --num-hidden $hidden \
        --dropout $dropout \
        --fan-out $fan_out \
        --eval &> logs/sage/1_runs/train_eval/${Data}_${Aggre}_${seed}_layers_${layers}_nb_2_run_${run}_epochs_${epoch}.log

done
