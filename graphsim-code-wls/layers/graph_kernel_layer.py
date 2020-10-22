import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn

class GraphSageKernelLayer(nn.Module):
    def __init__(self, residual=False):
        super().__init__()
        self.residual = residual
    
    def forward(self, graph, features):
        graph.ndata['h'] = features
        graph.update_all(fn.copy_src(src='h', out='m'), fn.mean(msg='m', out='h'))

        if self.residual:
            graph.ndata['h'] += features
        
        return graph, graph.ndata['h']

class WWLKernelLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, graph, features):
        graph.ndata['h'] = features
        graph.update_all(fn.copy_src(src='h', out='m'), fn.mean(msg='m', out='h'))

        # self-addition
        graph.ndata['h'] += features
        graph.ndata['h'] *= 0.5
        
        return graph, graph.ndata['h']

class GCNKernelLayer(nn.Module):
    def __init__(self, residual=False):
        super().__init__()
        self.residual = residual

    def forward(self, graph, features):
        if self.residual:
            degree = (1 + graph.in_degrees().float()).view(-1, 1)
        else:
            degree = graph.in_degrees().float().view(-1, 1)
        degree = degree.to(features.device)
        degree_inv = torch.where(degree > 0, 1 / degree, torch.tensor(0.0).to(degree.device))
        
        graph.ndata['h'] = features * degree_inv.sqrt()
        graph.update_all(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='h'))
        graph.ndata['h'] *= degree_inv.sqrt()
        
        if self.residual:
            graph.ndata['h'] += features * degree_inv

        return graph, graph.ndata['h']

def gat_msg(edges):
    score = torch.sum(edges.src['h'] * edges.dst['h'], 1)
    return {'m' : edges.src['h'], 's' : score}

def build_gat_reduce(temperature):
    def reduce_fn(nodes, temperature=temperature):
        features = nodes.mailbox['m']

        scores = F.softmax(nodes.mailbox['s'] / temperature, -1).unsqueeze(-1)

        return {'h' : torch.sum(scores * features, 1)}
    return reduce_fn

class GATKernelLayer(nn.Module):
    def __init__(self, residual=False, temperature=1.0):
        super().__init__()
        self.residual = residual
        self.reduce_fn = build_gat_reduce(temperature)

    def forward(self, graph, features):
        graph.ndata['h'] = features
        
        graph.update_all(gat_msg, self.reduce_fn)
        
        if self.residual:
            graph.ndata['h'] += features

        return graph, graph.ndata['h']