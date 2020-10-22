import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn

from layers.wls_layer import WLSMLPLayer
from layers.mlp_readout_layer import MLPReadout

class WLSMLPNetSBM(nn.Module):
    def __init__(self, net_params):
        super().__init__()

        n_iter = net_params.pop('n_iter')
        num_node_types = net_params.pop('in_dim')
        embed_dim, hidden_dim, out_dim = net_params.pop('embed_dim'), net_params.pop('hidden_dim'), net_params.pop('out_dim')
        n_mlp_layer, scale_mlp, dropout = net_params.pop('n_mlp_layer'), net_params.pop('scale_mlp'), net_params.pop('dropout')
        self.n_classes = net_params.pop('n_classes')

        hidden_dim = hidden_dim + (hidden_dim % 2)
        out_dim = out_dim + (out_dim % 2)

        layers = []

        _layer = WLSMLPLayer(embed_dim, hidden_dim, n_mlp_layer, scale_mlp, dropout, **net_params)
        layers.append(_layer)
        layers.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(n_iter - 2):
            _layer = WLSMLPLayer(hidden_dim, hidden_dim, n_mlp_layer, scale_mlp, dropout, **net_params)
            layers.append(_layer)
            layers.append(nn.BatchNorm1d(hidden_dim))

        _layer = WLSMLPLayer(hidden_dim, out_dim, n_mlp_layer, scale_mlp, dropout, **net_params)
        layers.append(_layer)
        layers.append(nn.BatchNorm1d(out_dim))
        
        self.n_embedding = nn.Embedding(num_node_types, embed_dim)
        self.layers = nn.ModuleList(layers)
        self.classifier = MLPReadout(out_dim, self.n_classes)

    def forward(self, graph, node_feat, edge_feat, snorm_n, snorm_e):
        node_feat = self.n_embedding(node_feat)

        for layer in self.layers:
            if not isinstance(layer, nn.BatchNorm1d):
                node_feat = layer(graph, node_feat)
            else:
                node_feat = layer(node_feat)

        predict = self.classifier(node_feat)

        return predict

    def loss(self, predict, label):
        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().cuda()
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes>0).float()
        
        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(predict, label)

        return loss


        
