from tqdm import tqdm

import numpy as np
import torch
import ot
import dgl

def degrees(graph, normalize=False):
    d = graph.in_degrees()
    d = np.asarray(d, dtype=np.float64)
    if normalize:
        d = d / max(d.sum(), 1e-7)
    return d
    
def gram_matrix(graphs, dims, dist_fn, readout='sum'):
    '''Wrapper function to compute the gram matrix on graphs in batch, returns list of gram matrices as numpy.array
    graphs : list of dgl graphs
    dims : graph features are concatenation of features obtained from all iterations, and this variable has
        the individual feature dimensions for the iterations.
    '''
    graphs = dgl.batch(graphs)
    if readout == 'sum':
        graph_reprs = dgl.sum_nodes(graphs, 'h')
    elif readout == 'mean':
        graph_reprs = dgl.mean_nodes(graphs, 'h')
    else:
        raise ValueError('Readout for gram_matrix shall be either "mean" or "sum"')
    
    distances = []

    dims = np.cumsum([0] + dims)

    with torch.no_grad():
        for dim_start, dim_end in zip(dims, dims[1:]):
            features = graph_reprs[:, dim_start:dim_end]

            gram_matrix = dist_fn(features, features)
            distances.append(gram_matrix.cpu().numpy())

    return distances

def euclidean_matrix(graphs, dims, readout='sum'):
    '''Returns the pairwise euclidean distance between readout feature from all graphs.
    graphs : list of dgl graphs
    dims : graph features are concatenation of features obtained from all iterations, and this variable has
        the individual feature dimensions for the iterations.
    '''
    graphs = dgl.batch(graphs)
    if readout == 'sum':
        graph_reprs = dgl.sum_nodes(graphs, 'h')
    elif readout == 'mean':
        graph_reprs = dgl.mean_nodes(graphs, 'h')
    else:
        raise ValueError('Readout for gram_matrix shall be either "mean" or "sum"')
    
    distances = []

    dims = np.cumsum([0] + dims)

    with torch.no_grad():
        for dim_start, dim_end in zip(dims, dims[1:]):
            features = graph_reprs[:, dim_start:dim_end]

            matrix = dist_matrix(features, features)
            distances.append(matrix.cpu().numpy())

    return distances



def kernel_distance(graphs, dims, kernel_name, stack_feat=False):
    '''Compute pair-wise kernel distance between graphs using kernel_fn.
    Kernel distance : \sum_p,p' v(p)K(p,p')v(p') + \sum_q,q' w(q)K(q,q')w(q') - 2\sum_p,q v(p)K(p,q)w(q)

    weight_degree : if True, the weights are degrees divided by sum. Otherwise 1/n
    stack_feat : if True, the i-th iteration embedding contains all the embeddings from previous iterations too.
    '''
    N = len(graphs)
    assert isinstance(dims, list), "dims shall be given as a list to wasserstein_distance"
    dims = list(np.cumsum([0] + dims))

    distances = []

    with torch.no_grad():
        for idx_iter in range(len(dims) - 1):
            feat_start, feat_end = dims[idx_iter], dims[idx_iter + 1]
            if stack_feat:
                feat_start = 0
            M = np.zeros((N,N))

            # Kernel distance for self
            kernel_self = [kernel_pair(g, g, feat_start, feat_end, kernel_name) for g in graphs]

            # Kernel distance between pairs
            count = tqdm(range(N*(N-1)//2), desc=f'K-dist Iter #{idx_iter}')
            for i, graph_i in enumerate(graphs[:-1]):
                for j, graph_j in enumerate(graphs[i:]):
                    if j == 0:
                        continue
                    k_between = kernel_pair(graph_i, graph_j, feat_start, feat_end, kernel_name)

                    M[i, i+j] = kernel_self[i] + kernel_self[j] - 2 * k_between
                
                    count.update()
            count.close()

            M = M + M.T
            distances.append(M)
        
            print(f'Kernel distance for WL iteration #{idx_iter} done.')                    

    return distances

def kernel_pair(graph_x, graph_y, feat_start, feat_end, kernel_name):
    '''Compute kernel distance between the node features of graph_x and graph_y as sets.
    Kernel distance : \sum_p,q v(p) K(p,q) w(q)
    If weight_degree is True then v, w are normalized degrees.
    If False, v, w are uniform which sums to 1.
    '''
    if kernel_name == 'linear':
        kernel_fn = kernel_linear
    if kernel_name == 'poly':
        kernel_fn = kernel_poly
    if kernel_name == 'rbf':
        kernel_fn = kernel_rbf

    feat_x = graph_x.ndata['h'][:, feat_start:feat_end]
    feat_y = graph_y.ndata['h'][:, feat_start:feat_end]    

    weight_x = torch.ones(feat_x.size(0)) / feat_x.size(0)
    weight_y = torch.ones(feat_y.size(0)) / feat_y.size(0)

    kernel_mat = kernel_fn(feat_x, feat_y)
    weight_x, weight_y = weight_x.to(kernel_mat.device), weight_y.to(kernel_mat.device)
    return torch.mm(torch.mm(weight_x.view(1, -1), kernel_mat), weight_y.view(-1, 1))


def kernel_linear(x, y):
    return torch.matmul(x, y.t())

def kernel_poly(x, y, order=3, c=1):
    inner_prod = torch.matmul(x, y.t())
    k_mat = (c + inner_prod).pow(order)
    return k_mat

def kernel_rbf(x, y, sigma=1):
    dist = dist_matrix(x, y).pow(2)
    return torch.exp(-dist / (2 * sigma))

def dist_matrix(x, y, p=2):
    '''Compute Lp-distance matrices between two tensors (N_x, dim) and (N_y, dim)
    Returns a tensor of shape (N_x, N_y)
    '''
    params_limit = 5e8

    batch_size = int(max(params_limit // x.numel(), 1))

    matrix = []
    for cursor in range(0, len(y), batch_size):
        y_part = y[cursor : cursor + batch_size]
        submatrix = (x.unsqueeze(1) - y_part.unsqueeze(0)).norm(p, dim=-1)
        matrix.append(submatrix)

    return torch.cat(matrix, 1)

