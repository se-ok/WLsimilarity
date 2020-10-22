from tqdm import tqdm

import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC
import torch

def gridsearch(kernel_matrices, labels, param_grid, data_indices):
    '''For each kernel matrix and SVC parameters in kernel_matrices and param_grid,
    apply K-fold test using data_indices and report the mean, std accuracy corresponding to
    maximum mean accuracy w.r.t. the validation set.

    kernel_matrics : different kernel matrices for the same data, list of 2d numpy arrays
    labels : numpy array of the same length as kernel_matrices[0]
    param_grid : sklearn.model_selectic.ParameterGrid input, e.g. [{'C': np.logspace(-3,3,num=7)}]
    data_indices : a triple (train, val, test) when zipped outputs the indices for the corresponding split.
    '''
    param_grid = list(ParameterGrid(param_grid))

    scoreboard_train = torch.zeros(len(kernel_matrices), len(kernel_matrices[0]), len(param_grid), len(data_indices[0]))
    scoreboard_val = torch.zeros(len(kernel_matrices), len(kernel_matrices[0]), len(param_grid), len(data_indices[0]))
    scoreboard_test = torch.zeros(len(kernel_matrices), len(kernel_matrices[0]), len(param_grid), len(data_indices[0]))

    labels = np.asarray(labels)

    pbar = tqdm(range(len(kernel_matrices) * len(kernel_matrices[0]) * len(param_grid)), desc='GridSearch')
    # kernel matrices has the form (# iter, gamma, params)
    for idx_iter, kernel_mat in enumerate(kernel_matrices):
        for idx_g, K in enumerate(kernel_mat):
            for idx_params, params in enumerate(param_grid):
                for idx_fold, indices in enumerate(zip(*data_indices)):
                    svc = SVC(kernel='precomputed', **params)

                    train_index, val_index, test_index = indices

                    K_train, y_train = K[train_index][:, train_index], labels[train_index]
                    K_val, y_val = K[val_index][:, train_index], labels[val_index]
                    K_test, y_test = K[test_index][:, train_index], labels[test_index]

                    svc.fit(K_train, y_train)

                    idx = (idx_iter, idx_g, idx_params, idx_fold)
                    scoreboard_train[idx] = svc.score(K_train, y_train)
                    scoreboard_val[idx] = svc.score(K_val, y_val)
                    scoreboard_test[idx] = svc.score(K_test, y_test)
                pbar.update()
    pbar.close()

    return scoreboard_train, scoreboard_val, scoreboard_test

def axis_split(scoreboard):
    '''scoreboard is of shape (# iter, # gammas, # params, # splits)
    Transpose and flatten to shape (# splits, -1) so that we can extract the best hyperparameter for each split
    '''
    scoreboard = scoreboard.permute(3, 0, 1, 2)
    scoreboard = scoreboard.view(scoreboard.size(0), -1)
    return scoreboard