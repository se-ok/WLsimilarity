#!/bin/bash

############
# Graph Neural Networks for MNIST / CIFAR10 superpixels
############

# GatedGCN
# GAT
# GCN
# GIN
# GraphSage
# MLP
# MoNet
# WLS

############
# Datasets
############

# SBM_PATTERN
# SBM_CLUSTER

############
# How to run
############

# sh script_main_SBMs_nns.sh 0 1
# sh script_main_SBMs_nns.sh 1 10
# sh script_main_SBMs_nns.sh 2 100
# sh script_main_SBMs_nns.sh 3 1000

# sh script_main_SBMs_nns.sh 0 1
# sh script_main_SBMs_nns.sh 1 10
# sh script_main_SBMs_nns.sh 2 100
# sh script_main_SBMs_nns.sh 3 1000

gpu=$1
seed=$2

dataset=SBM_CLUSTER
for net in GatedGCN GAT GCN GIN GraphSage MLP MoNet WLS
do
    python main_SBMs_nns.py --gpu ${gpu} --dataset ${dataset} --net ${net} --seed ${seed}
done

dataset=SBM_PATTERN
for net in GatedGCN GAT GCN GIN GraphSage MLP MoNet WLS
do
    python main_SBMs_nns.py --gpu ${gpu} --dataset ${dataset} --net ${net} --seed ${seed}
done