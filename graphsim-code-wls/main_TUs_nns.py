import os
import argparse
import json
import random
import time
from datetime import datetime
from copy import deepcopy
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import dgl

from data.data import LoadData
from nets.TUs_graph_classification.load_net import gnn_model

log_filename = 'log_TU_nn.txt'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', help="GPU to use")
    parser.add_argument('--dataset', required=True)
    parser.add_argument('-bs', '--batch_size', default=20, type=int)
    parser.add_argument('--net', required=True)
    parser.add_argument('--seed', type=int, default=1)

    return parser.parse_args()

args = get_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.backends.cudnn.benchmark = True

dataset = LoadData(args.dataset)
all_loader = DataLoader(dataset.all, batch_size=args.batch_size, collate_fn=dataset.collate)
in_dim = dataset.all.graph_lists[0].ndata['feat'].size(-1)

config = 'configs/TUs_graph_classification_nns.json'
with open(config) as inf:
    config = json.load(inf)

net_params = config[args.net]

params = {
    "seed": args.seed,
    "epochs": 1000,
    "init_lr": 1e-3,
    "lr_reduce_factor": 0.5,
    "lr_schedule_patience": 25,
    "min_lr": 1e-6,
    "weight_decay": 0.0,
    "max_time": 48
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def train(net, loader, optimizer):
    net.train()
    
    total, correct = 0, 0
    losses = []

    for graph, label, snorm_n, snorm_e in loader:
        nfeat = graph.ndata['feat'].cuda()
        efeat = graph.edata['feat'].cuda()
        label, snorm_n, snorm_e = label.cuda(), snorm_n.cuda(), snorm_e.cuda()
        
        logit = net(graph, nfeat, efeat, snorm_n, snorm_e)

        loss = net.loss(logit, label)

        total += label.size(0)
        correct += torch.eq(torch.argmax(logit, -1), label).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return sum(losses) / len(losses), correct / total

def val(net, loader):
    net.eval()
    
    total, correct = 0, 0
    losses = []

    with torch.no_grad():
        for graph, label, snorm_n, snorm_e in loader:
            nfeat = graph.ndata['feat'].cuda()
            efeat = graph.edata['feat'].cuda()
            label, snorm_n, snorm_e = label.cuda(), snorm_n.cuda(), snorm_e.cuda()
            
            logit = net(graph, nfeat, efeat, snorm_n, snorm_e)

            loss = net.loss(logit, label)

            total += label.size(0)
            correct += torch.eq(torch.argmax(logit, -1), label).sum().item()

            losses.append(loss.item())

    return sum(losses) / len(losses), correct / total

for key, _net_param in net_params.items():
    print(f'Starting {args.net}{key}')
    time_start = time.perf_counter()
    with tqdm(zip(dataset.train, dataset.val, dataset.test), total=len(dataset.train)) as split:
        train_losses, train_accuracy = [], []
        val_losses, val_accuracy = [], []
        test_losses, test_accuracy = [], []
        for idx_split, (d_train, d_val, d_test) in enumerate(split):
            split.set_description(f'Split #{idx_split}')

            set_seed(params['seed'])

            train_loader = DataLoader(d_train, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate)
            val_loader = DataLoader(d_val, batch_size=args.batch_size, collate_fn=dataset.collate)
            test_loader = DataLoader(d_test, batch_size=args.batch_size, collate_fn=dataset.collate)

            net_param = deepcopy(_net_param)
            net_param['in_dim'] = in_dim
            net_param['n_classes'] = dataset.all.num_labels

            net = gnn_model(args.net, net_param)
            net.cuda()

            optimizer = optim.Adam(net.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                factor=params['lr_reduce_factor'],
                                                                patience=params['lr_schedule_patience'])

            with tqdm(range(params['epochs'])) as epochs:
                for e in epochs:
                    epochs.set_description(f'Epoch #{e}')
                    
                    train_loss, train_acc = train(net, train_loader, optimizer)
                    val_loss, val_acc = val(net, val_loader)

                    epochs.set_postfix(lr=optimizer.param_groups[0]['lr'],
                                    train_loss=f'{train_loss:.3f}', train_acc=f'{train_acc*100:.2f}',
                                    val_loss=f'{val_loss:.3f} (LR reduce count {scheduler.num_bad_epochs})',
                                    val_acc=f'{val_acc*100:.2f}')

                    scheduler.step(val_loss)

                    if optimizer.param_groups[0]['lr'] < params['min_lr']:
                        # print("\n!! LR EQUAL TO MIN LR SET.")
                        break

            test_loss, test_acc = val(net, test_loader)

            train_losses.append(train_loss), train_accuracy.append(train_acc)
            val_losses.append(val_loss), val_accuracy.append(val_acc)
            test_losses.append(test_loss), test_accuracy.append(test_acc)

            split.set_postfix_str(f'Test Loss : {np.mean(test_losses):.3f}, '
                f'Test acc : {np.mean(test_accuracy) * 100:.2f} +- {np.std(test_accuracy) * 100:.2f}')
    time_total = time.perf_counter() - time_start

    print(f'Test Loss : {np.mean(test_losses):.3f}, '
            f'Test acc : {np.mean(test_accuracy) * 100:.2f} +- {np.std(test_accuracy) * 100:.2f}, '
            f'Num params : {sum([p.numel() for p in net.parameters()])}')

    if not os.path.isfile(log_filename):
        with open(log_filename, 'w') as of:
            print('\t'.join([
                'Dataset', 'Net', 'Param_key', 'Seed',
                'Mean_train_loss', 'Mean_val_loss', 'Mean_test_loss',
                'Mean_train_acc', 'Std_train_acc',
                'Mean_val_acc', 'Std_val_acc',
                'Mean_test_acc', 'Std_test_acc',
                'Time', '#Params'
            ]), file=of)

    with open(log_filename, 'a') as of:
        print('\t'.join(map(str, [
            args.dataset, args.net, key, args.seed,
            np.mean(train_losses), np.mean(val_losses), np.mean(test_losses),
            100 * np.mean(train_accuracy), 100 * np.std(train_accuracy),
            100 * np.mean(val_accuracy), 100 * np.std(val_accuracy),
            100 * np.mean(test_accuracy), 100 * np.std(test_accuracy),
            time_total, sum([p.numel() for p in net.parameters()])
        ])), file=of)
