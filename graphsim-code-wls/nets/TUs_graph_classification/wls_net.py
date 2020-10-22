import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn

from layers.wls_layer import WLSMLPLayer, WLSMLPLayerE
from layers.mlp_readout_layer import MLPReadout

class WLSMLPNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()

        n_iter = net_params.pop('n_iter')
        in_dim, hidden_dim, out_dim = net_params.pop('in_dim'), net_params.pop('hidden_dim'), net_params.pop('out_dim')
        n_mlp_layer, scale_mlp, dropout = net_params.pop('n_mlp_layer'), net_params.pop('scale_mlp'), net_params.pop('dropout')
        residual = net_params.pop('residual')

        hidden_dim = hidden_dim + (hidden_dim % 2)
        out_dim = out_dim + (out_dim % 2)

        layers = []

        _layer = WLSMLPLayer(in_dim, hidden_dim, n_mlp_layer, scale_mlp, dropout, residual)
        layers.append(_layer)
        layers.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(n_iter - 2):
            _layer = WLSMLPLayer(hidden_dim, hidden_dim, n_mlp_layer, scale_mlp, dropout, residual)
            layers.append(_layer)
            layers.append(nn.BatchNorm1d(hidden_dim))

        _layer = WLSMLPLayer(hidden_dim, out_dim, n_mlp_layer, scale_mlp, dropout, residual)
        layers.append(_layer)
        layers.append(nn.BatchNorm1d(out_dim))

        self.layers = nn.ModuleList(layers)
        self.dims = [in_dim] + [hidden_dim] * (n_iter - 1) + [out_dim]
        self.num_classes = net_params['n_classes']
        self.classifier = MLPReadout(out_dim, self.num_classes)

    def forward(self, graph, node_feat, edge_feat, snorm_n, snorm_e, reduce='mean'):
        for layer in self.layers:
            if not isinstance(layer, nn.BatchNorm1d):
                node_feat = layer(graph, node_feat)
            else:
                node_feat = layer(node_feat)

        graph.ndata['h'] = node_feat

        if reduce is None:
            return graph

        graph_repr = dgl.mean_nodes(graph, 'h')
        logit = self.classifier(graph_repr)

        return logit

    def loss(self, logit, label):
        return F.cross_entropy(logit, label)


class WLSMLPNetE(nn.Module):
    '''Anisotropic WLSMLP layer
    '''
    def __init__(self, net_params):
        super().__init__()

        n_iter = net_params.pop('n_iter')
        in_dim, hidden_dim, out_dim = net_params.pop('in_dim'), net_params.pop('hidden_dim'), net_params.pop('out_dim')
        n_mlp_layer, scale_mlp, dropout = net_params.pop('n_mlp_layer'), net_params.pop('scale_mlp'), net_params.pop('dropout')
        residual, aggregation = net_params.pop('residual'), net_params.pop('aggregation')

        hidden_dim = hidden_dim + (hidden_dim % 2)
        out_dim = out_dim + (out_dim % 2)

        layers = []

        _layer = WLSMLPLayerE(in_dim, hidden_dim, n_mlp_layer, scale_mlp, dropout, residual)
        layers.append(_layer)
        layers.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(n_iter - 2):
            _layer = WLSMLPLayerE(hidden_dim, hidden_dim, n_mlp_layer, scale_mlp, dropout, residual)
            layers.append(_layer)
            layers.append(nn.BatchNorm1d(hidden_dim))

        _layer = WLSMLPLayerE(hidden_dim, out_dim, n_mlp_layer, scale_mlp, dropout, residual)
        layers.append(_layer)
        layers.append(nn.BatchNorm1d(out_dim))

        self.layers = nn.ModuleList(layers)
        self.dims = [in_dim] + [hidden_dim] * (n_iter - 1) + [out_dim]
        self.num_classes = net_params['n_classes']
        self.classifier = MLPReadout(out_dim, self.num_classes)

    def forward(self, graph, node_feat, edge_feat, snorm_n, snorm_e, reduce='mean'):
        for layer in self.layers:
            if not isinstance(layer, nn.BatchNorm1d):
                node_feat = layer(graph, node_feat)
            else:
                node_feat = layer(node_feat)

        graph.ndata['h'] = node_feat

        if reduce is None:
            return graph

        graph_repr = dgl.mean_nodes(graph, 'h')
        logit = self.classifier(graph_repr)

        return logit

    def loss(self, logit, label):
        return F.cross_entropy(logit, label)