#!/bin/bash

############
# WL-type Graph Kernels for continuous labels
############

#GraphSage
#GAT
#GCN
#WWL
#WLS

############
# Datasets
############

# ENZYMES
# PROTEINS_full
# Synthie
# BZR
# COX2
# AIDS
# DHFR

############
# How to run
############

# sh script_main_TUs_kernels.sh 0 BZR
# sh script_main_TUs_kernels.sh 1 COX2
# sh script_main_TUs_kernels.sh 2 DHFR
# sh script_main_TUs_kernels.sh 3 ENZYMES
# sh script_main_TUs_kernels.sh 0 PROTEINS_full
# sh script_main_TUs_kernels.sh 1 Synthie

gpu=$1
dataset=$2
niter=5

for net in GraphSage GAT GCN WWL WLS WLSLin
do
    python main_TUs_kernels.py --iter ${niter} --dataset ${dataset} --net ${net} --gpu ${gpu}
    python main_TUs_kernels.py --iter ${niter} --dataset ${dataset} --net ${net} --gpu ${gpu} -mn
done
