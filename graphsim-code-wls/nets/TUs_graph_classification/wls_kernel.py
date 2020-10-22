import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

from .kernels import KernelBase
from layers.wls_layer import WLSKernelLayer, WLSEntryLayer

class WLSKernel(KernelBase):
    def __init__(self, net_params):
        super().__init__(net_params)

        in_dim, hidden_dim, out_dim = net_params.pop('in_dim'), net_params.pop('hidden_dim'), net_params.pop('out_dim')

        # feature dimension of input and outputs of each layer
        layers = []
        
        _layer = WLSKernelLayer(in_dim, hidden_dim, **net_params)
        layers.append(_layer)
        
        for _ in range(self.n_iter - 2):
            _layer = WLSKernelLayer(hidden_dim, hidden_dim, **net_params)
            layers.append(_layer)

        _layer = WLSKernelLayer(hidden_dim, out_dim, **net_params)
        layers.append(_layer)

        self.layers = nn.ModuleList(layers)
        self.dims = [self.in_dim] + [hidden_dim] * (self.n_iter - 1) + [out_dim]
        self.normalizer = None

class WLSEntryKernel(KernelBase):
    def __init__(self, net_params):
        super().__init__(net_params)

        in_dim, order = net_params.pop('in_dim'), net_params.pop('order')
        scale = net_params.pop('scale')

        layers = []
        
        for idx in range(self.n_iter):
            _layer = WLSEntryLayer(in_dim * (order ** idx), order, scale, **net_params)
            layers.append(_layer)

        self.layers = nn.ModuleList(layers)
        self.dims = [self.in_dim * (order ** idx) for idx in range(self.n_iter + 1)]
        self.normalizer = None