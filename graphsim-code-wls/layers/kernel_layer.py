import torch
import torch.nn as nn
import torch.nn.functional as F

class KernelLinear(nn.Module):
    '''Given a set of features (N, dim),
    apply the identity map corresponding lifting of the linear kernel to each vector of dimension <dim>
    and returns the resulting tensor of shape (N, dim), N representations of dimension dim in RKHS.
    '''
    def __init__(self, cuda=False):
        super(KernelLinear, self).__init__()
        self.cuda = cuda

    def forward(self, features):
        return features

    def dim_repr(self, in_dim):
        return in_dim

class KernelPoly(nn.Module):
    '''Given a set of features (N, dim),
    apply a lifting map for d-th order polynomial kernel to each vector of dimension <dim>
    and returns the resulting tensor of shape (N, D), N representations of dimension D in RKHS.
    '''
    def __init__(self, order, c=1, s=1, cuda=False):
        super(KernelPoly, self).__init__()
        self.order = order  # degree
        self.c = c  # additive constant
        self.s = s  # (c + s x.y)^order
        self.do_cuda = cuda

    def forward(self, features):
        if self.do_cuda:
            features = features.cuda()
        N, dim = features.shape

        feature_list = []
        # summands of order 0
        coeff = self.c ** (self.order / 2)
        zeroth = torch.ones(N, 1) * coeff
        if self.do_cuda:
            zeroth = zeroth.cuda()
        feature_list.append(zeroth)
        
        # summands of order 1
        coeff = (self.c ** ((self.order - 1) / 2)) * (self.order ** 0.5) * (self.s ** 0.5)
        feature_list.append(features * coeff)

        # temporary variable
        f = features

        for i in range(2, self.order + 1):
            factor = (self.order - i + 1) / (self.c * i)
            coeff = coeff * (factor ** 0.5) * (self.s ** 0.5)

            _features = features.unsqueeze(-1)
            _f = f.unsqueeze(-2)
            
            f = (_features * _f).view(N, -1)
            feature_list.append(f * coeff)

        features = torch.cat(feature_list, -1)
        
        return features
    
    def dim_repr(self, in_dim):
        '''Returns the output dimension when input dimension is <in_dim>
        '''
        return sum([in_dim ** i for i in range(self.order + 1)])
            
class KernelRBF(nn.Module):
    '''Given a set of features (N, dim),
    apply a lifting map for d-th order Taylor approximation of RBF kernel to each vector of dimension <dim>
    and returns the resulting tensor of shape (N, D), N representations of dimension D in RKHS.
    '''
    def __init__(self, order, sigma=1, cuda=False):
        super(KernelRBF, self).__init__()
        self.order = order
        self.sigma = sigma
        self.do_cuda = cuda

    def forward(self, features):
        if self.do_cuda:
            features = features.cuda()
        N, dim = features.shape
        norms = features.norm(2, dim=1, keepdim=True)
        
        feature_list = []
        # summands of order 0
        zeroth = torch.zeros(N, 1)
        if self.do_cuda:
            zeroth = zeroth.cuda()
        feature_list.append(zeroth)

        # summands of order 1
        feature_list.append(features)

        # temporary variable
        f = features
        coeff = 1.0
        
        for i in range(2, self.order + 1):
            factor = 1.0 / i
            coeff = coeff * (factor ** 0.5) / self.sigma

            _features = features.unsqueeze(-1)
            _f = f.unsqueeze(-2)

            f = (_features * _f).view(N, -1)
            feature_list.append(f * coeff)

        features = torch.cat(feature_list, -1)

        return features * torch.exp(- norms.pow(2) / (2 * self.sigma * self.sigma))

    def dim_repr(self, in_dim):
        '''Returns the output dimension when input dimension is <in_dim>
        '''
        return sum([in_dim ** i for i in range(self.order + 1)])

class MLPLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()

        layer = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.layer = layer

    def forward(self, x):
        return self.layer(x)

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
        # Reshape for nn.BatchNorm1d
        shape = x.shape
        x = x.view(-1, shape[-1])

        _x = x
        for layer in self.layers:
            x = layer(x)
        
        if self.do_residual:
            x += self.residual_layer(_x)

        return x.view(*shape[:-1], x.size(-1))

class RandomProjection(nn.Module):
    '''Fix a random projection from high dimensional space via unit vectors.
    Applied to 3D tensor of shape (N, D)
    '''
    def __init__(self, in_dim, out_dim, binary=False, cuda=False):
        super(RandomProjection, self).__init__()
        if out_dim < 50:
            print(f'Random projection onto less than 50 dimensional space might not preserve the norm well.')
        matrix = torch.randn(in_dim, out_dim)
        if binary:
            matrix = torch.where(matrix > 0, torch.tensor(1.0), torch.tensor(-1.0))
        if cuda:
            matrix = matrix.cuda()

        self.reduction = F.normalize(matrix, p=2, dim=1)
        
    def forward(self, features):
        return torch.matmul(features, self.reduction)
        
Kernels = {
    'rbf' : KernelRBF,
    'linear' : KernelLinear,
    'poly' : KernelPoly
}

