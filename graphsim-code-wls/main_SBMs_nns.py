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
from nets.SBMs_node_classification.load_net import gnn_model

log_filename = 'log_sbms_nn.txt'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', help="GPU to use")
    parser.add_argument('-bs', '--batch_size', type=int, default=128)
    parser.add_argument('--dataset', required=True, choices=['SBM_CLUSTER', 'SBM_PATTERN'])
    parser.add_argument('--net', required=True)
    parser.add_argument('--seed', type=int, default=1)

    return parser.parse_args()

args = get_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.backends.cudnn.benchmark = True

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self

dataset = LoadData(args.dataset)

config = 'configs/SBM_node_classification_nns.json'
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
    
    total, correct = 0, 0
    losses = []

    with tqdm(loader, total=len(loader)) as batches:
        for idx_train, (graph, label, snorm_n, snorm_e) in enumerate(batches):
            batches.set_description(f'Iteration #{idx_train+1}/{len(loader)}')
            graph.ndata['feat'] = graph.ndata['feat'].cuda()
            graph.edata['feat'] = graph.edata['feat'].cuda()
            snorm_n, snorm_e = snorm_n.cuda(), snorm_e.cuda()
            label = label.cuda()
            
            logit = net(graph, graph.ndata['feat'], graph.edata['feat'], snorm_n, snorm_e)

            loss = net.loss(logit, label)

            total += label.size(0)
            correct += torch.eq(torch.argmax(logit, -1), label).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            batches.set_postfix(loss=f'{np.mean(losses):.3f}', acc=f'{correct / total * 100:.2f}')

    return sum(losses) / len(losses), correct / total

def val(loader):
    net.eval()
    
    total, correct = 0, 0
    losses = []

    with torch.no_grad():
        for graph, label, snorm_n, snorm_e in loader:
            graph.ndata['feat'] = graph.ndata['feat'].cuda()
            graph.edata['feat'] = graph.edata['feat'].cuda()
            snorm_n, snorm_e = snorm_n.cuda(), snorm_e.cuda()
            label = label.cuda()
            
            logit = net(graph, graph.ndata['feat'], graph.edata['feat'], snorm_n, snorm_e)
            loss = net.loss(logit, label)

            total += label.size(0)
            correct += torch.eq(torch.argmax(logit, -1), label).sum().item()

            losses.append(loss.item())

    return sum(losses) / len(losses), correct / total

set_seed(params['seed'])

train_loader = DataLoader(dataset.train, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate, num_workers=5)
val_loader = DataLoader(dataset.val, batch_size=args.batch_size, collate_fn=dataset.collate, num_workers=5)
test_loader = DataLoader(dataset.test, batch_size=args.batch_size, collate_fn=dataset.collate, num_workers=5)

for key, _net_param in net_params.items():
    print(f'Starting {args.net}{key} on {args.dataset}')
    time_start = time.perf_counter()

    net_param = deepcopy(_net_param)
    net_param['in_dim'] = torch.unique(dataset.train[0][0].ndata['feat'],dim=0).size(0)
    net_param['n_classes'] = torch.unique(dataset.train[0][1],dim=0).size(0)
    net_param['device'] = 'cuda:0'

    net = gnn_model(args.net, net_param)
    net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                        factor=params['lr_reduce_factor'],
                                                        patience=params['lr_schedule_patience'])

    with tqdm(range(params['epochs'])) as epochs:
        for e in epochs:
            epochs.set_description(f'Epoch #{e+1}')
            
            train_loss, train_acc = train(train_loader)
            val_loss, val_acc = val(val_loader)

            epochs.set_postfix(lr=optimizer.param_groups[0]['lr'],
                            train_loss=f'{train_loss:.3f}', train_acc=f'{train_acc*100:.2f}',
                            val_loss=f'{val_loss:.3f} (LR reduce count {scheduler.num_bad_epochs})',
                            val_acc=f'{val_acc*100:.2f}')

            scheduler.step(val_loss)

            if optimizer.param_groups[0]['lr'] < params['min_lr']:
                # print("\n!! LR EQUAL TO MIN LR SET.")
                break
    
    test_loss, test_acc = val(test_loader)

    print(f'Test Loss : {test_loss:.3f}, Test acc : {test_acc * 100:.2f}, '
            f'Num params : {sum([p.numel() for p in net.parameters()])}')

    time_total = time.perf_counter() - time_start

    if not os.path.isfile(log_filename):
        with open(log_filename, 'w') as of:
            print('\t'.join([
                'Dataset', 'Net', 'Param_key', 'Seed',
                'Train_loss', 'Val_loss', 'Test_loss',
                'Train_acc', 'Val_acc', 'Test_acc',
                'Time', '#Params'
            ]), file=of)

    with open(log_filename, 'a') as of:
        print('\t'.join(map(str, [
            args.dataset, args.net, key, args.seed,
            train_loss, val_loss, test_loss,
            100 * train_acc, 100 * val_acc, 100 * test_acc,
            time_total, sum([p.numel() for p in net.parameters()])
        ])), file=of)