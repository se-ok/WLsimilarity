import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

from layers.graph_kernel_layer import WWLKernelLayer, GraphSageKernelLayer, GCNKernelLayer, GATKernelLayer

class KernelBase(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.in_dim = net_params['in_dim']
        self.n_iter = net_params['n_iter']
        self.residual = net_params.pop('residual', False)
        
        self.layers = nn.ModuleList([])
        self.dims = [self.in_dim] * (self.n_iter + 1)
        self.normalizer = None
        
    def forward(self, graph, node_feat, edge_feat, snorm_n, snorm_e):
        if self.normalizer:
            node_feat = self.normalizer(node_feat)

        feat_list = [node_feat]
        for layer in self.layers:
            graph, node_feat = layer(graph, node_feat)
            feat_list.append(node_feat)

        node_feat = torch.cat(feat_list, -1)

        graph.ndata['h'] = node_feat

        return graph
        
    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

class WWLKernel(KernelBase):
    def __init__(self, net_params):
        super().__init__(net_params)
        
        self.layers = nn.ModuleList([WWLKernelLayer() for _ in range(self.n_iter)])

class GraphSageKernel(KernelBase):
    def __init__(self, net_params):
        super().__init__(net_params)
        
        self.layers = nn.ModuleList([GraphSageKernelLayer(self.residual) for _ in range(self.n_iter)])

class GCNKernel(KernelBase):
    def __init__(self, net_params):
        super().__init__(net_params)

        self.layers = nn.ModuleList([GCNKernelLayer(self.residual) for _ in range(self.n_iter)])

class GATKernel(KernelBase):
    def __init__(self, net_params):
        super().__init__(net_params)
        temperature = net_params['temperature']

        self.layers = nn.ModuleList([GATKernelLayer(self.residual, temperature) for _ in range(self.n_iter)])