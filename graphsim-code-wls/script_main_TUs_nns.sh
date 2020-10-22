#!/bin/bash

############
# WL-type Graph Neural Networks for continuous labels
############

############
# Datasets with node features
############

# AIDS : excluded since all are above 99%
# BZR
# COX2
# DHFR
# ENZYMES
# PROTEINS_full
# Synthie

############
# How to run
############
# sh script_main_TUs_nns.sh ${gpu} ${seed}
# For example,
# sh script_main_TUs_nns.sh 0 1
# sh script_main_TUs_nns.sh 1 10
# sh script_main_TUs_nns.sh 2 100
# sh script_main_TUs_nns.sh 3 1000
# runs the script four times on each gpu with seed 1, 10, 100, 1000 respectively
# results recorded to log_TU_nn.txt

gpu=$1
seed=$2

### BZR
dataset=BZR
for net in GatedGCN GAT GCN GIN GraphSage MLP MoNet WLS
do
    python main_TUs_nns.py --dataset ${dataset} --net ${net} --gpu ${gpu} --batch_size 20 --seed ${seed}
done

### COX2
dataset=COX2
for net in GatedGCN GAT GCN GIN GraphSage MLP MoNet WLS
do
    python main_TUs_nns.py --dataset ${dataset} --net ${net} --gpu ${gpu} --batch_size 20 --seed ${seed}
done

### DHFR
dataset=DHFR
for net in GatedGCN GAT GCN GIN GraphSage MLP MoNet WLS
do
    python main_TUs_nns.py --dataset ${dataset} --net ${net} --gpu ${gpu} --batch_size 20 --seed ${seed}
done

### ENZYMES
dataset=ENZYMES
for net in GatedGCN GAT GCN GIN GraphSage MLP MoNet WLS
do
    python main_TUs_nns.py --dataset ${dataset} --net ${net} --gpu ${gpu} --batch_size 20 --seed ${seed}
done

### PROTEINS_full
dataset=PROTEINS_full
for net in GatedGCN GAT GCN GIN GraphSage MLP MoNet WLS
do
    python main_TUs_nns.py --dataset ${dataset} --net ${net} --gpu ${gpu} --batch_size 20 --seed ${seed}
done

### Synthie
dataset=Synthie
for net in GatedGCN GAT GCN GIN GraphSage MLP MoNet WLS
do
    python main_TUs_nns.py --dataset ${dataset} --net ${net} --gpu ${gpu} --batch_size 20 --seed ${seed}
done

