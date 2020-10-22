import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

def combine(averages, variances, counts):
    """
    Combine averages and variances to one single average and variance.
    Note that np.var need ddof=1 for sample variance, while torch.var not.

    averages : (N, d) averages of dimension d of N parts
    variances : (N, d) sample variances of dimension d of N parts
    counts : (N,) counts of each part

    Returns (average, variance) where
        average : (d,) average over all parts
        variance : (d,) sample variance over all parts
    """
    N, d = averages.shape
    counts = counts.view(-1, 1)
    weight_counts = counts / counts.sum()
    
    average = torch.sum(averages * weight_counts, 0)

    squares = (counts - 1) * variances + counts * (averages - average).pow(2)
    variance = squares.sum(0) / counts.sum()

    return average, variance

class NormalizeNormal(nn.Module):
    '''Read entire dataset and choose normalizing constant, apply when called
    (data - mean) / std
    '''
    def __init__(self, axis=0, cuda=False):
        super().__init__()

        self.cuda = cuda
        self.axis = axis
        self.mean = 0
        self.std = 1
        
    def fit(self, loader, data_fn=lambda x:x):
        means, variances, counts = [], [], []

        for data in loader:
            data = data_fn(data)
            
            means.append(torch.mean(data, axis=self.axis))
            variances.append(torch.var(data, axis=self.axis))
            counts.append(data.size(0))
        
        means, variances, counts = torch.stack(means), torch.stack(variances), torch.Tensor(counts)
        
        self.mean, var = combine(means, variances, counts)
        self.std = var.sqrt()
        if self.cuda:
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()
    
    def forward(self, data):
        return (data - self.mean) / self.std.clamp(1e-7)
        