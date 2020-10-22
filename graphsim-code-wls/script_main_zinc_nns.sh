############
# Graph Neural Networks for regression on ZINC dataset
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

# ZINC

############
# How to run
############

# sh script_main_zinc_nns.sh 0 1
# sh script_main_zinc_nns.sh 1 10
# sh script_main_zinc_nns.sh 2 100
# sh script_main_zinc_nns.sh 3 1000

gpu=$1
seed=$2

for net in GatedGCN GAT GCN GIN GraphSage MLP MoNet WLS
do
    python main_molecules_nns.py --gpu ${gpu} --dataset ZINC --net ${net} --seed ${seed}
done