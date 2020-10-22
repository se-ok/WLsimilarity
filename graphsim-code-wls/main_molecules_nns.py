import os
import argparse
import random
import time
import json
from tqdm import tqdm
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import dgl

from data.data import LoadData
from nets.molecules_graph_regression.load_net import gnn_model

log_filename = 'log_zinc_nn.txt'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', help="GPU to use")
    parser.add_argument('--iter', type=int, default=4, help='Number of WL-iterations')
    parser.add_argument('-bs', '--batch_size', type=int, default=128)
    parser.add_argument('--dataset', default='ZINC')
    parser.add_argument('--net', required=True)
    parser.add_argument('--seed', default=1, type=int)

    return parser.parse_args()

args = get_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.backends.cudnn.benchmark = True

dataset = LoadData(args.dataset)

config = 'configs/molecules_graph_regression_nns.json'
with open(config) as inf:
    config = json.load(inf)

net_params = config[args.net]

params = {
    "seed": args.seed,
    "epochs": 1000,
    "init_lr": 1e-3,
    "lr_reduce_factor": 0.5,
    "lr_schedule_patience": 5,
    "min_lr": 1e-5,
    "weight_decay": 0.0,
    "max_time": 48
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def train(loader):
    net.train()
    
    losses = []

    with tqdm(loader, total=len(loader)) as batches:
        for idx_train, (graph, label, snorm_n, snorm_e) in enumerate(batches):
            batches.set_description(f'Iteration #{idx_train+1}/{len(loader)}')
            graph.ndata['feat'] = graph.ndata['feat'].cuda()
            graph.edata['feat'] = graph.edata['feat'].cuda()
            label = label.cuda()
            
            logit = net(graph, graph.ndata['feat'], graph.edata['feat'], snorm_n.cuda(), snorm_e.cuda())

            loss = net.loss(logit, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            batches.set_postfix(loss=f'{np.mean(losses):.3f}')

    return np.mean(losses)

def val(loader):
    net.eval()
    
    losses = []

    for graph, label, snorm_n, snorm_e in loader:
        graph.ndata['feat'] = graph.ndata['feat'].cuda()
        graph.edata['feat'] = graph.edata['feat'].cuda()
        label = label.cuda()
        
        logit = net(graph, graph.ndata['feat'], graph.edata['feat'], snorm_n.cuda(), snorm_e.cuda())
        loss = net.loss(logit, label)

        losses.append(loss.item())

    return np.mean(losses)

train_loader = DataLoader(dataset.train, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=dataset.collate)
val_loader = DataLoader(dataset.val, batch_size=args.batch_size, collate_fn=dataset.collate)
test_loader = DataLoader(dataset.test, batch_size=args.batch_size, collate_fn=dataset.collate)

for key, _net_param in net_params.items():
    print(f'Starting {args.net}{key} on {args.dataset}')
    time_start = time.perf_counter()

    net_param = deepcopy(_net_param)
    net_param['num_atom_type'] = dataset.num_atom_type
    net_param['num_bond_type'] = dataset.num_bond_type
    net_param['device'] = 'cuda:0'

    set_seed(params['seed'])

    net = gnn_model(args.net, net_param)
    net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                        factor=params['lr_reduce_factor'],
                                                        patience=params['lr_schedule_patience'])

    with tqdm(range(params['epochs'])) as epochs:
        for e in epochs:
            epochs.set_description(f'Epoch #{e+1}')
            
            train_loss = train(train_loader)
            val_loss = val(val_loader)

            epochs.set_postfix(lr=optimizer.param_groups[0]['lr'],
                            train_loss=f'{train_loss:.3f}',
                            val_loss=f'{val_loss:.3f} (LR reduce count {scheduler.num_bad_epochs})')

            scheduler.step(val_loss)

            if optimizer.param_groups[0]['lr'] < params['min_lr']:
                # print("\n!! LR EQUAL TO MIN LR SET.")
                break
    
    test_loss = val(test_loader)
    print(f'Test Loss : {test_loss:.3f}, # params : {sum([p.numel() for p in net.parameters()])}')

    time_total = time.perf_counter() - time_start

    if not os.path.isfile(log_filename):
        with open(log_filename, 'w') as of:
            print('\t'.join([
                'Dataset', 'Net', 'Param_key', 'Seed',
                'Train_loss', 'Val_loss', 'Test_loss',
                'Time', '#Params'
            ]), file=of)

    with open(log_filename, 'a') as of:
        print('\t'.join(map(str, [
            args.dataset, args.net, key, args.seed,
            train_loss, val_loss, test_loss,
            time_total, sum([p.numel() for p in net.parameters()])
        ])), file=of)