#!/bin/bash

File=pseudo_mini_batch_range_arxiv_sage.py
Data=ogbn-arxiv
epoch=2

model=sage
seed=1236 
setseed=True
GPUmem=True
lr=0.01
dropout=0.5
layers=3
hidden=256
run=1

# pMethodList=(random) 
# AggreList=(mean)

batch_size=(45471 22736 11368 5684 2842 1421)

fan_out_list=(25,35,40 25,35,80 25,70,80 50,70,80)
hiddenList=(32 64 128 256)
layersList=(3 4 5 6)

pMethodList=(random range) 
AggreList=(mean lstm)

savePath=../logs/sage/1_runs/pure_train/${Data}
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
                                                --batch-size $bs \
                                                --lr $lr \
                                                --num-runs $run \
                                                --num-epochs $epoch \
                                                --num-layers $layers \
                                                --num-hidden $hidden \
                                                --dropout $dropout \
                                                --fan-out $fan_out \
                                                --log-indent 1 \
                                                &> ${logPath}/${Data}_${Aggre}_${seed}_l_${layers}_fo_${fan_out}_nb_${nb}_r_${run}_ep_${epoch}.log
                                        done
                                done
                        done
                done
        done
done
