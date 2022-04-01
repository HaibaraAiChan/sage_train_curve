#!/bin/bash

File=full_and_pseudo_mini_batch_range_arxiv_sage.py
Data=ogbn-arxiv


model=sage
seed=1236 
setseed=True
GPUmem=True
lr=0.01
dropout=0.5
layers=3
hidden=256
weightDecay=0.005
# pMethodList=(random) 
# AggreList=(mean)

# batch_size=(45471 22736 11368 5684 2842 1421)

# fan_out_list=(25,35,40 25,35,80 25,70,80 50,70,80)
# hiddenList=(32 64 128 256)
# layersList=(3 4 5 6)
# pMethodList=(random range) 
# AggreList=(mean lstm)

batch_size=(45471 22736 11368)
fan_out_list=(25,35,40 )
hiddenList=(256)
layersList=(3)
pMethodList=(range) 
AggreList=(mean)
run=1
epoch=201
lr=0.0005
dropout=0.3



savePath=../../logs/sage/1_runs/train_eval/arxiv
mkdir ${savePath}

for Aggre in ${AggreList[@]}
do      
        mkdir ${savePath}/${Aggre}/
        for pMethod in ${pMethodList[@]}
        do      
                mkdir ${savePath}/${Aggre}/${pMethod}
                for layers in ${layersList[@]}
                do      
                        mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/
                        for hidden in ${hiddenList[@]}
                        do
                                mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/
                                for fan_out in ${fan_out_list[@]}
                                do
                                        nb=1
                                        for bs in ${batch_size[@]}
                                        do
                                                nb=$(($nb*2))
                                                # nb=32
                                                mkdir ${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/nb_${nb}
                                                logPath=${savePath}/${Aggre}/${pMethod}/layers_${layers}/h_${hidden}/nb_${nb}
                                                # mkdir $logPath
                                                echo $logPath
                                                echo 'number of batches'
                                                echo $nb
                                                python $File \
                                                --dataset $Data \
                                                --aggre $Aggre \
                                                --seed $seed \
                                                --setseed $setseed \
                                                --GPUmem $GPUmem \
                                                --selection-method $pMethod \
                                                --load-full-batch True \
                                                --batch-size $bs \
                                                --lr $lr \
                                                --weight-decay $weightDecay \
                                                --num-runs $run \
                                                --num-epochs $epoch \
                                                --num-layers $layers \
                                                --num-hidden $hidden \
                                                --dropout $dropout \
                                                --fan-out $fan_out \
                                                --log-indent 1 \
                                                --eval \
                                                > ${logPath}/${Data}_${Aggre}_${seed}_l_${layers}_fo_${fan_out}_nb_${nb}_r_${run}_ep_${epoch}.log
                                        done
                                done
                        done
                done
        done
done
