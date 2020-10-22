import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers import GraphConvolution, GraphAggregation, WLSMLPLayer, WLSLinearLayer


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dims, z_dim, vertexes, edges, nodes, dropout):
        super(Generator, self).__init__()

        self.vertexes = vertexes
        self.edges = edges
        self.nodes = nodes

        layers = []
        for c0, c1 in zip([z_dim]+conv_dims[:-1], conv_dims):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(p=dropout, inplace=True))
        self.layers = nn.Sequential(*layers)

        self.edges_layer = nn.Linear(conv_dims[-1], edges * vertexes * vertexes)
        self.nodes_layer = nn.Linear(conv_dims[-1], vertexes * nodes)
        self.dropoout = nn.Dropout(p=dropout)

    def forward(self, x):
        output = self.layers(x)
        edges_logits = self.edges_layer(output)\
                       .view(-1,self.edges,self.vertexes,self.vertexes)
        edges_logits = (edges_logits + edges_logits.permute(0,1,3,2))/2
        edges_logits = self.dropoout(edges_logits.permute(0,2,3,1))

        nodes_logits = self.nodes_layer(output)
        nodes_logits = self.dropoout(nodes_logits.view(-1,self.vertexes,self.nodes))

        return edges_logits, nodes_logits


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, conv_dim, m_dim, b_dim, dropout):
        super(Discriminator, self).__init__()

        graph_conv_dim, aux_dim, linear_dim = conv_dim
        # discriminator
        self.gcn_layer = GraphConvolution(m_dim, graph_conv_dim, b_dim, dropout)
        self.agg_layer = GraphAggregation(graph_conv_dim[-1], aux_dim, b_dim, dropout)

        # multi dense layer
        layers = []
        for c0, c1 in zip([aux_dim]+linear_dim[:-1], linear_dim):
            layers.append(nn.Linear(c0,c1))
            layers.append(nn.Dropout(dropout))
        self.linear_layer = nn.Sequential(*layers)

        self.output_layer = nn.Linear(linear_dim[-1], 1)

    def forward(self, adj, hidden, node, activatation=None):
        adj = adj[:,:,:,1:].permute(0,3,1,2)
        annotations = torch.cat((hidden, node), -1) if hidden is not None else node
        h = self.gcn_layer(annotations, adj)
        annotations = torch.cat((h, hidden, node) if hidden is not None\
                                 else (h, node), -1)
        h = self.agg_layer(annotations, torch.tanh)
        h = self.linear_layer(h)

        # Need to implemente batch discriminator #
        ##########################################

        output = self.output_layer(h)
        output = activatation(output) if activatation is not None else output

        return output, h


class WLSDiscriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, n_atoms, n_bonds, n_iter, in_dim, hidden_dim, out_dim,
                n_mlp_layer, scale_mlp, dropout, residual):
        super().__init__()

        hidden_dim = hidden_dim + (hidden_dim % 2)
        out_dim = out_dim + (out_dim % 2)

        layers = []

        _layer = WLSMLPLayer(in_dim, hidden_dim, n_mlp_layer, scale_mlp, dropout, residual, n_bonds - 1)
        layers.append(_layer)
        layers.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(n_iter - 2):
            _layer = WLSMLPLayer(hidden_dim, hidden_dim, n_mlp_layer, scale_mlp, dropout, residual, n_bonds - 1)
            layers.append(_layer)
            layers.append(nn.BatchNorm1d(hidden_dim))

        _layer = WLSMLPLayer(hidden_dim, out_dim, n_mlp_layer, scale_mlp, dropout, residual, n_bonds - 1)
        layers.append(_layer)
        layers.append(nn.BatchNorm1d(out_dim))

        self.n_embedding = nn.Linear(n_atoms, in_dim)
        self.layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(out_dim, 1)

    def forward(self, adj, hidden, node, activation=None):
        adj = adj[:,:,:,1:].permute(0,3,1,2)
        node = self.n_embedding(node)

        for layer in self.layers:
            if not isinstance(layer, nn.BatchNorm1d):
                node = layer(node, adj)
            else:
                B, N, d = node.shape
                node = layer(node.view(-1, d)).view(B, N, -1)
        
        graph_repr = torch.mean(node, 1)
        logit = self.output_layer(graph_repr)
        output = logit if activation is None else activation(logit)

        return output, graph_repr

class WLSDiscriminatorLinear(nn.Module):
    def __init__(self, n_atoms, n_bonds, n_iter, dim):
        super().__init__()

        self.n_embedding = nn.Linear(n_atoms, dim)
        layers = [
            WLSLinearLayer(dim) for _ in range(n_iter)
        ]
        self.layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(dim, 1)

    def forward(self, adj, hidden, node, activation=None):
        adj = adj[:,:,:,1:].permute(0,3,1,2)
        node = self.n_embedding(node)

        for layer in self.layers:
            node = layer(node, adj)
        
        graph_repr = torch.mean(node, 1)
        logit = self.output_layer(graph_repr)
        output = logit if activation is None else activation(logit)

        return output, graph_repr