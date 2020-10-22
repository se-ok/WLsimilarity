import os
import argparse
import json
import time
import random
from datetime import datetime
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import dgl

from data.data import LoadData

from nets.TUs_graph_classification.kernels import GraphSageKernel, GATKernel, GCNKernel, WWLKernel
from nets.TUs_graph_classification.wls_kernel import WLSKernel
from utils.distances import kernel_distance, euclidean_matrix
from utils.gridsearch import gridsearch, axis_split
from utils.normalizer import NormalizeNormal

log_filename = 'log_TU_kernel.txt'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter', type=int, required=True, help='Number of WL iterations')
    parser.add_argument('--gpu', default='0', help="GPU to use")
    parser.add_argument('--dataset', required=True)
    parser.add_argument('-bs', '--batch_size', default=50, type=int)
    parser.add_argument('--net', required=True)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('-mn', '--mean_node', action='store_true', help="Set the readout function to mean instead of sum")

    return parser.parse_args()

args = get_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

seed = args.seed

dataset = LoadData(args.dataset)
all_loader = DataLoader(dataset.all, batch_size=args.batch_size, collate_fn=dataset.collate)
in_dim = dataset.all.graph_lists[0].ndata['feat'].size(-1)

normalizer_class = NormalizeNormal
normalizer = normalizer_class(axis=0, cuda=True)
normalizer.fit(all_loader, data_fn=lambda x:x[0].ndata['feat'])

config = 'configs/TUs_graph_classification_kernels.json'
with open(config) as inf:
    config = json.load(inf)

net_params = config[args.net]

net_cls = {
    'GraphSage' : GraphSageKernel,
    'GAT' : GATKernel,
    'GCN' : GCNKernel,
    'WWL' : WWLKernel,
    'WLS' : WLSKernel,
    'WLSLin' : WLSKernel
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_embedding(net, loader):
    with torch.no_grad():
        graph_list, label_list = [], []
        for graphs, labels, snorm_n, snorm_e in loader:
            n_feat = graphs.ndata['feat'].cuda()
            e_feat = graphs.edata['feat'].cuda()

            graphs = net(graphs, n_feat, e_feat, snorm_n, snorm_e)

            graph_list.extend(dgl.unbatch(graphs))
            label_list.append(labels)
        
        label_list = torch.cat(label_list)

    return graph_list, label_list

readout = 'mean' if args.mean_node else 'sum'
distance_type = 'euclidean'

# placeholders
num_splits = len(dataset.train)
scoreboard_train, scoreboard_val, scoreboard_test = [], [], []
time_embedding, time_wdist, time_svc = 0, 0, 0

for key, _params in net_params.items():
    params = deepcopy(_params)

    params['in_dim'] = in_dim
    params['n_iter'] = args.iter

    set_seed(seed)
    
    net = net_cls[args.net](params)

    net.set_normalizer(normalizer)

    print(f'Computing representation from {args.net}{key}/{len(net_params)} on {args.dataset}...')
    print(params)
    # time for representation computing
    time_start = time.perf_counter()
    graphs, labels = get_embedding(net, all_loader)
    time_embedding += time.perf_counter() - time_start

    # time for wasserstein distance
    time_start = time.perf_counter()
    kernel_matrices = euclidean_matrix(graphs, net.dims, readout=readout)
    time_wdist += time.perf_counter() - time_start

    # time for k-fold SVC
    time_start = time.perf_counter()
    # Apply laplacian kernel using the distance matrix
    gammas = np.logspace(-6, 3, num=10)
    kernels = []
    for K in kernel_matrices:
        kernels.append([np.clip(np.exp(-g * np.power(K,2)), 0, 1e10) for g in gammas])

    # Finding best hyperparameter for SVC using validation set
    param_grid = [
        {'C' : np.logspace(-6, 3, num=10)}
    ]

    data_indices = (dataset.all_idx['train'], dataset.all_idx['val'], dataset.all_idx['test'])

    score_train, score_val, score_test = gridsearch(kernels, labels, param_grid, data_indices)
    time_svc += time.perf_counter() - time_start

    path_name = 'kernel-result'
    if not os.path.isdir(path_name):
        os.makedirs(path_name)
    scores_name = f'{args.net}{key}_{args.dataset}.pt'
    torch.save({
        'train' : score_train,
        'val' : score_val,
        'test' : score_test
    }, os.path.join(path_name, scores_name))

    score_train = axis_split(score_train)
    score_val = axis_split(score_val)
    score_test = axis_split(score_test)

    scoreboard_train.append(score_train)
    scoreboard_val.append(score_val)
    scoreboard_test.append(score_test)

scoreboard_train = torch.cat(scoreboard_train, 1)
scoreboard_val = torch.cat(scoreboard_val, 1)
scoreboard_test = torch.cat(scoreboard_test, 1)

max_val_idx = scoreboard_val.max(1)[1]
test_scores = scoreboard_test.gather(1, max_val_idx.view(-1, 1)).view(-1)

mean_test, std_test = test_scores.mean().item(), test_scores.std().item()

print(f'*** Result {args.net}')
print(f'\tMax Mean Accuracy {mean_test*100:.3f} +- {std_test*100:.3f}')

if not os.path.isfile(log_filename):
    with open(log_filename, 'w') as of:
        log_title = '\t'.join([
            'Dataset', 'Net', 'Distance', 'Readout', 'Normalizer',
            'Test_acc_mean', 'Test_acc_std',
            'time_embedding', 'time_wdist', 'time_svc', 'time_total'
        ])

        print(log_title, file=of)

# average execution time for single run
time_embedding /= len(net_params)
time_wdist /= len(net_params)
time_svc /= len(net_params)

_normalizer = 'unit' if args.unit else 'normal'

with open(log_filename, 'a') as of:
    log_test = '\t'.join([
        args.dataset, args.net, distance_type, readout, _normalizer,
        f'{mean_test*100:.3f}', f'{std_test*100:.3f}'
    ])
    log_test += '\t' + '\t'.join(map(str, [
        time_embedding, time_wdist, time_svc, time_embedding + time_wdist + time_svc
    ]))

    print(log_test, file=of)