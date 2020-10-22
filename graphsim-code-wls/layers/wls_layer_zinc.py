import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn

from .kernel_layer import Kernels, RandomProjection, KernelMLP

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
    def __init__(self, in_dim, out_dim, kernel_name, kernel_params, scale=1.0, cat_scale=0.0, binary=False, cuda=False, **kwargs):
        super().__init__()
        self.do_cuda = cuda
        self.scale = scale
        self.cat_scale = cat_scale
        kernel = Kernels[kernel_name]
        self.expansion = kernel(cuda=cuda, **kernel_params)
        expansion_dim = self.expansion.dim_repr(in_dim)
        if cat_scale > 0.0:
            expansion_dim += in_dim
        self.reduction = RandomProjection(expansion_dim, out_dim, binary, cuda)
        

    def forward(self, graph, features):
        features = features * self.scale
        features_expanded = self.expansion(features)
        
        # aggregation
        graph.ndata['h'] = features_expanded
        graph.update_all(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='h'))

        # self-info. Concate if self.cat_scale > 0, sum if self.cat_scale == 0, nothing if self.cat_scale < 0
        if self.cat_scale > 0:
            features = torch.cat([features * self.cat_scale, graph.ndata['h']], -1)
        elif self.cat_scale < 0:
            features = graph.ndata['h']
        else:
            features = graph.ndata['h'] + features_expanded
        
        features = self.reduction(features)
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


def build_message_fn(embedding, transform):
    def message_fn(edges, embedding=embedding, transform=transform):
        e_feat = embedding(edges.data['feat'])
        features = torch.cat([e_feat, edges.src['h']], -1)
        return {'m' : transform(features)}
    return message_fn
        

class WLSMLPLayerEdge(nn.Module):
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

        self.transform_self = KernelMLP(in_dim, out_dim // 2, n_hidden, scale_hidden, dropout, residual=residual)
        self.transform = KernelMLP(in_dim * 2, out_dim // 2, n_hidden, scale_hidden, dropout, residual=residual)
        if 'num_edge_types' not in kwargs:
            raise KeyError('WLS Layer requires "num_edge_types"')
        self.e_embedding = nn.Embedding(kwargs['num_edge_types'], in_dim)
        self.message_fn = build_message_fn(self.e_embedding, self.transform)

        if aggregation == 'sum':
            self.agg_fn = fn.sum(msg='m', out='h')
        elif aggregation == 'mean':
            self.agg_fn = fn.mean(msg='m', out='h')
        else:
            raise ValueError('Aggregation for WLS MLP shall be either "sum" or "mean"')
    
    def forward(self, graph, features):
        graph.ndata['h'] = features
        graph.update_all(self.message_fn, self.agg_fn)

        features = self.transform_self(features)
        features = torch.cat([features, graph.ndata['h']], -1)

        return features
