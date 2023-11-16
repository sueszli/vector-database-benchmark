import logging
import os
import random
from collections import Counter
import torch

class EM:
    """
    EM algorithm used to quantize the columns of W to minimize

                         ||W - W_hat||^2

    Args:
        - W: weight matrix of size (in_features x out_features)
        - n_iter: number of k-means iterations
        - n_centroids: number of centroids (size of codebook)
        - eps: for cluster reassignment when an empty cluster is found
        - max_tentatives for cluster reassignment when an empty cluster is found
        - verbose: print error after each iteration

    Remarks:
        - If one cluster is empty, the most populated cluster is split into
          two clusters
        - All the relevant dimensions are specified in the code
    """

    def __init__(self, W, n_centroids=256, n_iter=20, eps=1e-06, max_tentatives=30, verbose=True):
        if False:
            while True:
                i = 10
        self.W = W
        self.n_centroids = n_centroids
        self.n_iter = n_iter
        self.eps = eps
        self.max_tentatives = max_tentatives
        self.verbose = verbose
        self.centroids = torch.Tensor()
        self.assignments = torch.Tensor()
        self.objective = []

    def initialize_centroids(self):
        if False:
            return 10
        '\n        Initializes the centroids by sampling random columns from W.\n        '
        (in_features, out_features) = self.W.size()
        indices = torch.randint(low=0, high=out_features, size=(self.n_centroids,)).long()
        self.centroids = self.W[:, indices].t()

    def step(self, i):
        if False:
            i = 10
            return i + 15
        '\n        There are two standard steps for each iteration: expectation (E) and\n        minimization (M). The E-step (assignment) is performed with an exhaustive\n        search and the M-step (centroid computation) is performed with\n        the exact solution.\n\n        Args:\n            - i: step number\n\n        Remarks:\n            - The E-step heavily uses PyTorch broadcasting to speed up computations\n              and reduce the memory overhead\n        '
        distances = self.compute_distances()
        self.assignments = torch.argmin(distances, dim=0)
        n_empty_clusters = self.resolve_empty_clusters()
        for k in range(self.n_centroids):
            W_k = self.W[:, self.assignments == k]
            self.centroids[k] = W_k.mean(dim=1)
        obj = (self.centroids[self.assignments].t() - self.W).norm(p=2).item()
        self.objective.append(obj)
        if self.verbose:
            logging.info(f'Iteration: {i},\tobjective: {obj:.6f},\tresolved empty clusters: {n_empty_clusters}')

    def resolve_empty_clusters(self):
        if False:
            i = 10
            return i + 15
        '\n        If one cluster is empty, the most populated cluster is split into\n        two clusters by shifting the respective centroids. This is done\n        iteratively for a fixed number of tentatives.\n        '
        counts = Counter(map(lambda x: x.item(), self.assignments))
        empty_clusters = set(range(self.n_centroids)) - set(counts.keys())
        n_empty_clusters = len(empty_clusters)
        tentatives = 0
        while len(empty_clusters) > 0:
            k = random.choice(list(empty_clusters))
            m = counts.most_common(1)[0][0]
            e = torch.randn_like(self.centroids[m]) * self.eps
            self.centroids[k] = self.centroids[m].clone()
            self.centroids[k] += e
            self.centroids[m] -= e
            distances = self.compute_distances()
            self.assignments = torch.argmin(distances, dim=0)
            counts = Counter(map(lambda x: x.item(), self.assignments))
            empty_clusters = set(range(self.n_centroids)) - set(counts.keys())
            if tentatives == self.max_tentatives:
                logging.info(f'Could not resolve all empty clusters, {len(empty_clusters)} remaining')
                raise EmptyClusterResolveError
            tentatives += 1
        return n_empty_clusters

    def compute_distances(self):
        if False:
            while True:
                i = 10
        "\n        For every centroid m, computes\n\n                          ||M - m[None, :]||_2\n\n        Remarks:\n            - We rely on PyTorch's broadcasting to speed up computations\n              and reduce the memory overhead\n            - Without chunking, the sizes in the broadcasting are modified as:\n              (n_centroids x n_samples x out_features) -> (n_centroids x out_features)\n            - The broadcasting computation is automatically chunked so that\n              the tensors fit into the memory of the GPU\n        "
        nb_centroids_chunks = 1
        while True:
            try:
                return torch.cat([(self.W[None, :, :] - centroids_c[:, :, None]).norm(p=2, dim=1) for centroids_c in self.centroids.chunk(nb_centroids_chunks, dim=0)], dim=0)
            except RuntimeError:
                nb_centroids_chunks *= 2

    def assign(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Assigns each column of W to its closest centroid, thus essentially\n        performing the E-step in train().\n\n        Remarks:\n            - The function must be called after train() or after loading\n              centroids using self.load(), otherwise it will return empty tensors\n        '
        distances = self.compute_distances()
        self.assignments = torch.argmin(distances, dim=0)

    def save(self, path, layer):
        if False:
            for i in range(10):
                print('nop')
        '\n        Saves centroids and assignments.\n\n        Args:\n            - path: folder used to save centroids and assignments\n        '
        torch.save(self.centroids, os.path.join(path, '{}_centroids.pth'.format(layer)))
        torch.save(self.assignments, os.path.join(path, '{}_assignments.pth'.format(layer)))
        torch.save(self.objective, os.path.join(path, '{}_objective.pth'.format(layer)))

    def load(self, path, layer):
        if False:
            i = 10
            return i + 15
        '\n        Loads centroids and assignments from a given path\n\n        Args:\n            - path: folder use to load centroids and assignments\n        '
        self.centroids = torch.load(os.path.join(path, '{}_centroids.pth'.format(layer)))
        self.assignments = torch.load(os.path.join(path, '{}_assignments.pth'.format(layer)))
        self.objective = torch.load(os.path.join(path, '{}_objective.pth'.format(layer)))

class EmptyClusterResolveError(Exception):
    pass