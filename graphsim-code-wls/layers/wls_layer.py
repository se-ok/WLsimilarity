import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn

from .kernel_layer import Kernels, RandomProjection, KernelMLP

absolute_max = 1e6

class WLSKernelLayer(nn.Module):
    '''Given graphs and node features, apply the following in sequence.
    - Lift the node features to RKHS
    - Weighted sum using the adjacency matrix to obtain a vector representation
        for weighted set of neighboring features.
    - Reduce dimension via projection to random unit vectors.

    expansion : a class which does lifting the features.
    expansion_params : lifting-specific parameters, for example the order and additive constant in the polynomial kernel.
    scale (float) : multiply to the node features before expansion for computational feasibility.
    binary : if True, the reduction uses binary random unit vectors instead of gaussian.
    '''
    def __init__(self, in_dim, out_dim, kernel_name, kernel_params, cat_scale=0.0, binary=False, cuda=False, **kwargs):
        super().__init__()
        self.do_cuda = cuda
        self.in_dim = in_dim
        self.cat_scale = cat_scale
        kernel = Kernels[kernel_name]
        self.expansion = kernel(cuda=cuda, **kernel_params)
        expansion_dim = self.expansion.dim_repr(in_dim)
        if cat_scale > 0.0:
            expansion_dim += in_dim
        self.reduction = RandomProjection(expansion_dim, out_dim, binary, cuda)

    def forward(self, graph, features):
        features_expanded = self.expansion(features).clamp(-absolute_max, absolute_max)
        
        # aggregation
        graph.ndata['h'] = features_expanded
        graph.update_all(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='h'))
        graph.ndata['h'].clamp_(-absolute_max, absolute_max)

        # self-info. Concate if self.cat_scale > 0, sum if self.cat_scale == 0
        if self.cat_scale > 0:
            features = torch.cat([features * self.cat_scale, graph.ndata['h']], -1)
        else:
            features = graph.ndata['h'] + features_expanded

        features.clamp_(-absolute_max, absolute_max)

        features = self.reduction(features) / math.sqrt(self.in_dim)
        graph.ndata['h'] = features
        
        return graph, graph.ndata['h']

class WLSMLPLayer(nn.Module):
    '''Given graphs and node features, apply the following in sequence.
    - Each node feature goes through MLP to be transformed, corresponding to kernel lifting.
    - The node features are collected from neighbors and summed.
    
    in_dim : input feature dimension
    out_dim : output feature dimension
    n_hidden : number of hidden layers in transformation MLP
    scale_hidden : the hidden dimension of MLP scales up by this number at each layer
    edge_weight : if true, multiply graph.edata['feat'] before summing the neighbor representations.

    The output has dimension in_dim + out_dim since we concat the input to the output before return.
    '''
    def __init__(self, in_dim, out_dim, n_hidden, scale_hidden, dropout=0.0,
                residual=False, aggregation='sum', **kwargs):
        super().__init__()

        self.transform = KernelMLP(in_dim, out_dim // 2, n_hidden, scale_hidden, dropout, residual=residual)

        if aggregation == 'sum':
            self.agg_fn = fn.sum(msg='m', out='h')
        elif aggregation == 'mean':
            self.agg_fn = fn.mean(msg='m', out='h')
        else:
            raise ValueError('Aggregation for WLS MLP shall be either "sum" or "mean"')
    
    def forward(self, graph, features):
        features = self.transform(features)

        graph.ndata['h'] = features
        graph.update_all(fn.copy_src(src='h', out='m'), self.agg_fn)

        features = torch.cat([features, graph.ndata['h']], -1)

        return features


def build_message_fn(src_proj, dst_proj):
    def message_fn(edges, src_proj=src_proj, dst_proj=dst_proj):
        src_feat, dst_feat = src_proj(edges.src['h']), dst_proj(edges.dst['h'])
        weights = torch.sum(src_feat * dst_feat, -1) / math.sqrt(src_feat.size(-1))
        weights = torch.sigmoid(weights.view(-1, 1))
        return {'m' : src_feat * weights}
    return message_fn

class WLSMLPLayerE(nn.Module):
    '''Given graphs and node features, apply the following in sequence.
    - Each node feature goes through MLP to be transformed, corresponding to kernel lifting.
    - The node features are collected from neighbors and summed.
    
    in_dim : input feature dimension
    out_dim : output feature dimension
    n_hidden : number of hidden layers in transformation MLP
    scale_hidden : the hidden dimension of MLP scales up by this number at each layer
    edge_weight : if true, multiply graph.edata['feat'] before summing the neighbor representations.

    The output has dimension in_dim + out_dim since we concat the input to the output before return.
    '''
    def __init__(self, in_dim, out_dim, n_hidden, scale_hidden, dropout=0.0,
                residual=False, **kwargs):
        super().__init__()

        self.transform = KernelMLP(in_dim, out_dim // 2, n_hidden, scale_hidden, dropout, residual=residual)

        proj_dim = out_dim // 2
        self.src_proj = nn.Linear(proj_dim, proj_dim)
        self.dst_proj = nn.Linear(proj_dim, proj_dim)

        self.message_fn = build_message_fn(self.src_proj, self.dst_proj)

    def forward(self, graph, features):
        features = self.transform(features)

        graph.ndata['h'] = features
        graph.update_all(self.message_fn, fn.sum(msg='m', out='h'))

        features = torch.cat([features, graph.ndata['h']], -1)

        return features

def factorial_odd(k):
    '''1 * 3 * ... * k (or k-1)
    '''
    p = 1
    for i in range(3, k+1, 2):
        p *= i
    return p

class WLSEntryLayer(nn.Module):
    def __init__(self, in_dim, order, scale, edge_weight, cuda=False, **kwargs):
        super().__init__()
        self.in_dim = in_dim
        self.order = order
        self.out_dim = in_dim * order
        self.means, self.stds = self.coeffs(order+1)
        self.scale = scale
        self.edge_weight = edge_weight

    def forward(self, graph, features):
        feature_list = [
            (features.pow(i) - self.means[i]) / self.stds[i] * (self.scale ** i) for i in range(1, self.order+1)
        ]

        graph.ndata['h'] = torch.cat(feature_list, -1)
        if self.edge_weight:
            graph.update_all(fn.u_mul_e('h', 'feat', 'm'), fn.sum(msg='m', out='h'))
        else:
            graph.update_all(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='h'))

        return graph, graph.ndata['h']

    @staticmethod
    def coeffs(up_to):
        # E(z^k) for k=0, 1, 2, ..., up_to
        means = [1]
        for i in range(1, up_to+1):
            if i % 2 == 1:
                means.append(0)
            else:
                means.append(factorial_odd(i))

        # Var(z^k) for k=0, 1, 2, ..., up_to
        variances = [1]
        for i in range(1, up_to+1):
            if i % 2 == 1:
                variances.append(factorial_odd(2*i))
            else:
                variances.append(factorial_odd(2*i) - factorial_odd(i)**2)

        return means, [math.sqrt(v) for v in variances]

    def dim_repr(self, in_dim):
        return in_dim * self.order
