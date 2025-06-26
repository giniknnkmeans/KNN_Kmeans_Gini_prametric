import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import confusion_matrix
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from numpy.linalg import inv, pinv
import scipy
import scipy.stats as ss

import torch
import scipy.stats as ss

class GiniDistanceTorch:
    """
    Compute Gini distance between two matrices (Torch version, GPU support)
    """
    def __init__(self, X, gini_param=2, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.X = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.gini_param = gini_param

    def _rank(self, X):
        # Ranks along each column (descending rank: highest value has rank 1)
        # Equivalent to scipy.stats.rankdata(..., method='average', axis=0)
        X = X.clone()
        ranks = torch.zeros_like(X)
        for col in range(X.shape[1]):
            order = torch.argsort(X[:, col], descending=True)
            ranks[order, col] = torch.arange(1, X.shape[0]+1, device=X.device, dtype=X.dtype)
        return ranks

    def compute_gini_ranks(self, X):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        X_cat = torch.cat((self.X, X), dim=0)
        ranks = self._rank(X_cat)
        n_train = self.X.shape[0]
        n_cat = X_cat.shape[0]
        ranks = (ranks / n_cat * n_train) ** (self.gini_param - 1)
        return ranks[:n_train], ranks[n_train:]

    def gini_distance(self, x, Y, decum_rank_x, decum_ranks_Y):
        # x: [features], Y: [n_train, features]
        # decum_rank_x: [features], decum_ranks_Y: [n_train, features]
        distance = -torch.sum((x - Y) * (decum_rank_x - decum_ranks_Y), dim=1)
        return distance

    def compute_distances(self, X):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        ranks_train, ranks_test = self.compute_gini_ranks(X)
        n_test = X.shape[0]
        n_train = self.X.shape[0]
        distances = torch.empty((n_test, n_train), device=self.device)

        for i, x in enumerate(X):
            distances[i, :] = self.gini_distance(
                x, self.X, ranks_test[i], ranks_train
            )
        return distances.cpu().numpy()  

    def compute_minkowski_distances(self, X, p=2):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        n_test = X.shape[0]
        n_train = self.X.shape[0]
        distances = torch.empty((n_test, n_train), device=self.device)
        for i, x in enumerate(X):
            diff = torch.abs(self.X - x)
            dists = torch.sum(diff ** p, dim=1) ** (1. / p)
            distances[i, :] = dists
        return distances.cpu().numpy()
