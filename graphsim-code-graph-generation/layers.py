import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):

    def __init__(self, in_features, out_feature_list, b_dim, dropout):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_feature_list = out_feature_list

        self.linear1 = nn.Linear(in_features, out_feature_list[0])
        self.linear2 = nn.Linear(out_feature_list[0], out_feature_list[1])

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, adj, activation=None):
        # input : 16x9x9
        # adj : 16x4x9x9

        hidden = torch.stack([self.linear1(input) for _ in range(adj.size(1))], 1)
        hidden = torch.einsum('bijk,bikl->bijl', (adj, hidden))
        hidden = torch.sum(hidden, 1) + self.linear1(input)
        hidden = activation(hidden) if activation is not None else hidden
        hidden = self.dropout(hidden)

        output = torch.stack([self.linear2(hidden) for _ in range(adj.size(1))], 1)
        output = torch.einsum('bijk,bikl->bijl', (adj, output))
        output = torch.sum(output, 1) + self.linear2(hidden)
        output = activation(output) if activation is not None else output
        output = self.dropout(output)

        return output


class GraphAggregation(Module):

    def __init__(self, in_features, out_features, b_dim, dropout):
        super(GraphAggregation, self).__init__()
        self.sigmoid_linear = nn.Sequential(nn.Linear(in_features+b_dim, out_features),
                                            nn.Sigmoid())
        self.tanh_linear = nn.Sequential(nn.Linear(in_features+b_dim, out_features),
                                         nn.Tanh())
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, activation):
        i = self.sigmoid_linear(input)
        j = self.tanh_linear(input)
        output = torch.sum(torch.mul(i,j), 1)
        output = activation(output) if activation is not None\
                 else output
        output = self.dropout(output)

        return output

class MLPLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.linear(x)
        shape = x.shape
        x = self.bn(x.view(-1, x.size(-1)))
        x = x.view(*shape)
        x = self.relu(x)
        x = self.dropout(x)

        return x

class KernelMLP(nn.Module):
    '''Apply MLP to given node features individually.
    '''
    def __init__(self, in_dim, out_dim, n_layers, scale, dropout=0.0, residual=False):
        super().__init__()
        
        hidden_dim = int(scale * max(in_dim, out_dim))
        layers = [MLPLayer(in_dim, hidden_dim, dropout)]
        
        for _ in range(n_layers - 2):
            layers.append(MLPLayer(hidden_dim, hidden_dim, dropout))
        
        layers.append(MLPLayer(hidden_dim, out_dim, dropout))
        
        self.layers = nn.ModuleList(layers)
        self.do_residual = residual

        if in_dim == out_dim:
            self.residual_layer = nn.Sequential()
        else:
            self.residual_layer = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        _x = x
        for layer in self.layers:
            x = layer(x)
        
        if self.do_residual:
            return x + self.residual_layer(_x)
        else:
            return x

class WLSMLPLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_hidden, scale_hidden, dropout=0.0, residual=False,
                num_bond_types=4):
        super().__init__()

        self.transform = KernelMLP(in_dim * 2, out_dim // 2, n_hidden, scale_hidden, dropout, residual)
        self.transform_self = KernelMLP(in_dim, out_dim // 2, n_hidden, scale_hidden, dropout, residual)
        self.embedding = nn.Parameter(torch.Tensor(num_bond_types, in_dim))
        nn.init.xavier_uniform_(self.embedding)

    def forward(self, node_feat, adj):
        # node_feat (B, max_nodes, nodes_dim)
        # adj (B, edge_types, max_nodes, max_nodes)

        residual = self.transform_self(node_feat)

        B, N, Dn = node_feat.shape
        M, De = self.embedding.shape
        
        # concat edge embedding with node features
        node_feat = node_feat.unsqueeze(1).repeat(1, adj.size(1), 1, 1)
        edge_feat = self.embedding.view(1, M, 1, De).repeat(B, 1, N, 1)
        
        concat = torch.cat([edge_feat, node_feat], -1) # (B, M, N, De+Dn)
        
        # transform the features
        features = concat.view(-1, De + Dn)
        features = self.transform(features)
        features = features.view(B, M, N, -1)

        # aggregate over neighbors for each bond type
        features = torch.matmul(adj, features)

        # sum over bond types
        features = features.sum(1)

        # concat with original node features transformed by self.transform_self
        features = torch.cat([residual, features], -1)

        return features

class WLSLinearLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()

    def forward(self, node_feat, adj):
        nfeat_aggregated = torch.matmul(adj, node_feat.unsqueeze(1))   # B, M, N, Dn
        nfeat_aggregated = torch.mean(nfeat_aggregated, 1) # B, N, Dn
        return node_feat + nfeat_aggregated