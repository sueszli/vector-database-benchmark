"""LUNAR: Unifying Local Outlier Detection Methods via Graph Neural Networks
"""
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted
from .base import BaseDetector

def generate_negative_samples(x, sample_type, proportion, epsilon):
    if False:
        for i in range(10):
            print('nop')
    n_samples = int(proportion * len(x))
    n_dim = x.shape[-1]
    rand_unif = x.min() + (x.max() - x.min()) * np.random.rand(n_samples, n_dim).astype('float32')
    x_temp = x[np.random.choice(np.arange(len(x)), size=n_samples)]
    randmat = np.random.rand(n_samples, n_dim) < 0.3
    rand_sub = x_temp + randmat * (epsilon * np.random.randn(n_samples, n_dim)).astype('float32')
    if sample_type == 'UNIFORM':
        neg_x = rand_unif
    if sample_type == 'SUBSPACE':
        neg_x = rand_sub
    if sample_type == 'MIXED':
        neg_x = np.concatenate((rand_unif, rand_sub), 0)
        neg_x = neg_x[np.random.choice(np.arange(len(neg_x)), size=n_samples)]
    neg_y = np.ones(len(neg_x))
    return (neg_x.astype('float32'), neg_y.astype('float32'))

class SCORE_MODEL(nn.Module):

    def __init__(self, k):
        if False:
            i = 10
            return i + 15
        super(SCORE_MODEL, self).__init__()
        self.hidden_size = 256
        self.network = nn.Sequential(nn.Linear(k, self.hidden_size), nn.Tanh(), nn.Linear(self.hidden_size, self.hidden_size), nn.Tanh(), nn.Linear(self.hidden_size, self.hidden_size), nn.Tanh(), nn.Linear(self.hidden_size, 1), nn.Sigmoid())

    def forward(self, x):
        if False:
            return 10
        out = self.network(x)
        out = torch.squeeze(out, 1)
        return out

class WEIGHT_MODEL(nn.Module):

    def __init__(self, k):
        if False:
            return 10
        super(WEIGHT_MODEL, self).__init__()
        self.hidden_size = 256
        self.network = nn.Sequential(nn.Linear(k, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(), nn.LayerNorm(self.hidden_size), nn.Linear(self.hidden_size, k))
        self.final_norm = nn.BatchNorm1d(1)

    def forward(self, x):
        if False:
            return 10
        alpha = self.network(x)
        alpha = F.softmax(alpha, dim=1)
        out = torch.sum(alpha * x, dim=1, keepdim=True)
        out = self.final_norm(out)
        out = torch.squeeze(out, 1)
        return out

class LUNAR(BaseDetector):
    """
    LUNAR class for outlier detection. See https://www.aaai.org/AAAI22Papers/AAAI-51.GoodgeA.pdf for details.
    For an observation, its ordered list of distances to its k nearest neighbours is input to a neural network, 
    with one of the following outputs:

        1) SCORE_MODEL: network directly outputs the anomaly score.
        2) WEIGHT_MODEL: network outputs a set of weights for the k distances, the anomaly score is then the
            sum of weighted distances.

    See :cite:`goodge2022lunar` for details.

    Parameters
    ----------
    model_type: str in ['WEIGHT', 'SCORE'], optional (default = 'WEIGHT')
        Whether to use WEIGHT_MODEL or SCORE_MODEL for anomaly scoring.

    n_neighbors: int, optional (default = 5)
        Number of neighbors to use by default for k neighbors queries.

    negative_sampling: str in ['UNIFORM', 'SUBSPACE', MIXED'], optional (default = 'MIXED)
        Type of negative samples to use between:

        - 'UNIFORM': uniformly distributed samples
        - 'SUBSPACE': subspace perturbation (additive random noise in a subset of features)
        - 'MIXED': a combination of both types of samples

    val_size: float in [0,1], optional (default = 0.1)
        Proportion of samples to be used for model validation

    scaler: object in {StandardScaler(), MinMaxScaler(), optional (default = MinMaxScaler())
        Method of data normalization

    epsilon: float, optional (default = 0.1)
        Hyper-parameter for the generation of negative samples. 
        A smaller epsilon results in negative samples more similar to normal samples.

    proportion: float, optional (default = 1.0)
        Hyper-parameter for the proprotion of negative samples to use relative to the 
        number of normal training samples.

    n_epochs: int, optional (default = 200)
        Number of epochs to train neural network.

    lr: float, optional (default = 0.001)
        Learning rate.

    wd: float, optional (default = 0.1)
        Weight decay.
    
    verbose: int in {0,1}, optional (default = 0):
        To view or hide training progress

    Attributes
    ----------
    """

    def __init__(self, model_type='WEIGHT', n_neighbours=5, negative_sampling='MIXED', val_size=0.1, scaler=MinMaxScaler(), epsilon=0.1, proportion=1.0, n_epochs=200, lr=0.001, wd=0.1, verbose=0):
        if False:
            i = 10
            return i + 15
        super(LUNAR, self).__init__()
        self.model_type = model_type
        self.n_neighbours = n_neighbours
        self.negative_sampling = negative_sampling
        self.epsilon = epsilon
        self.proportion = proportion
        self.n_epochs = n_epochs
        self.scaler = scaler
        self.lr = lr
        self.wd = wd
        self.val_size = val_size
        self.verbose = verbose
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if model_type == 'SCORE':
            self.network = SCORE_MODEL(n_neighbours).to(self.device)
        elif model_type == 'WEIGHT':
            self.network = WEIGHT_MODEL(n_neighbours).to(self.device)

    def fit(self, X, y=None):
        if False:
            print('Hello World!')
        'Fit detector. y is assumed to be 0 for all training samples.\n        Parameters\n        ----------\n        X : numpy array of shape (n_samples, n_features)\n            The input samples.\n        y : Ignored\n            Overwritten with 0 for all training samples (assumed to be normal).\n        Returns\n        -------\n        self : object\n            Fitted estimator.\n        '
        self._set_n_classes(y)
        X = X.astype('float32')
        y = np.zeros(len(X))
        (train_x, val_x, train_y, val_y) = train_test_split(X, y, test_size=self.val_size)
        if self.scaler == None:
            pass
        else:
            self.scaler.fit(train_x)
        if self.scaler == None:
            pass
        else:
            train_x = self.scaler.transform(train_x)
            val_x = self.scaler.transform(val_x)
        (neg_train_x, neg_train_y) = generate_negative_samples(train_x, self.negative_sampling, self.proportion, self.epsilon)
        (neg_val_x, neg_val_y) = generate_negative_samples(val_x, self.negative_sampling, self.proportion, self.epsilon)
        train_x = np.vstack((train_x, neg_train_x))
        train_y = np.hstack((train_y, neg_train_y))
        val_x = np.vstack((val_x, neg_val_x))
        val_y = np.hstack((val_y, neg_val_y))
        self.neigh = NearestNeighbors(n_neighbors=self.n_neighbours + 1)
        self.neigh.fit(train_x)
        (train_dist, _) = self.neigh.kneighbors(train_x[train_y == 0], n_neighbors=self.n_neighbours + 1)
        (neg_train_dist, _) = self.neigh.kneighbors(train_x[train_y == 1], n_neighbors=self.n_neighbours)
        train_dist = np.vstack((train_dist[:, 1:], neg_train_dist))
        (val_dist, _) = self.neigh.kneighbors(val_x, n_neighbors=self.n_neighbours)
        train_dist = torch.tensor(train_dist, dtype=torch.float32).to(self.device)
        train_y = torch.tensor(train_y, dtype=torch.float32).to(self.device)
        val_dist = torch.tensor(val_dist, dtype=torch.float32).to(self.device)
        val_y = torch.tensor(val_y, dtype=torch.float32).to(self.device)
        criterion = nn.MSELoss(reduction='none')
        optimizer = optim.Adam(self.network.parameters(), lr=self.lr, weight_decay=self.wd)
        best_val_score = 0
        for epoch in range(self.n_epochs):
            with torch.no_grad():
                self.network.eval()
                out = self.network(train_dist)
                train_score = roc_auc_score(train_y.cpu(), out.cpu())
                out = self.network(val_dist)
                val_score = roc_auc_score(val_y.cpu(), out.cpu())
                if val_score >= best_val_score:
                    best_dict = {'epoch': epoch, 'model_state_dict': deepcopy(self.network.state_dict()), 'optimizer_state_dict': deepcopy(optimizer.state_dict()), 'train_score': train_score, 'val_score': val_score}
                    best_val_score = val_score
                if self.verbose == 1:
                    print(f'Epoch {epoch} \t Train Score {np.round(train_score, 6)} \t Val Score {np.round(val_score, 6)}')
            self.network.train()
            optimizer.zero_grad()
            out = self.network(train_dist)
            loss = criterion(out, train_y).sum()
            loss.backward()
            optimizer.step()
        if self.verbose == 1:
            print(f"Finished training...\nBest Model: Epoch {best_dict['epoch']} \t Train Score {best_dict['train_score']} \t Val Score {best_dict['val_score']}")
        self.network.load_state_dict(best_dict['model_state_dict'])
        if self.scaler == None:
            X_norm = np.copy(X)
        else:
            X_norm = self.scaler.transform(X)
        (dist, _) = self.neigh.kneighbors(X_norm, self.n_neighbours)
        dist = torch.tensor(dist, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            self.network.eval()
            anomaly_scores = self.network(dist)
        self.decision_scores_ = anomaly_scores.cpu().detach().numpy().ravel()
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        if False:
            while True:
                i = 10
        'Predict raw anomaly score of X using the fitted detector.\n        For consistency, outliers are assigned with larger anomaly scores.\n        Parameters\n        ----------\n        X : numpy array of shape (n_samples, n_features)\n            The training input samples.\n        Returns\n        -------\n        anomaly_scores : numpy array of shape (n_samples,)\n            The anomaly score of the input samples.\n        '
        check_is_fitted(self, ['decision_scores_'])
        X = X.astype('float32')
        if self.scaler == None:
            pass
        else:
            X = self.scaler.transform(X)
        (dist, _) = self.neigh.kneighbors(X, self.n_neighbours)
        dist = torch.tensor(dist, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            self.network.eval()
            anomaly_scores = self.network(dist)
        scores = anomaly_scores.cpu().detach().numpy().ravel()
        return scores