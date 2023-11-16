from typing import List, Tuple, Union
from abc import ABC, abstractmethod
from collections import Counter
from itertools import combinations, product
from random import sample
from sys import maxsize
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from catalyst.contrib.data._misc import convert_labels2list, find_value_ids
TTriplets = Tuple[Tensor, Tensor, Tensor]
TTripletsIds = Tuple[List[int], List[int], List[int]]
TLabels = Union[List[int], Tensor]

class IInbatchTripletSampler(ABC):
    """An abstraction of inbatch triplet sampler."""

    @abstractmethod
    def _check_input_labels(self, labels: List[int]) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Check if the batch labels list is valid for the sampler.\n\n        We expect you to implement this method to guarantee correct\n        performance of sampling method. You can pass it\n        but we strongly do not recommend you to do it.\n\n        Args:\n            labels: labels of the samples in the batch;\n                list or Tensor of shape (batch_size;)\n        '
        raise NotImplementedError()

    @abstractmethod
    def sample(self, features: Tensor, labels: TLabels) -> TTriplets:
        if False:
            i = 10
            return i + 15
        '\n        This method includes the logic of sampling/selecting triplets.\n\n        Args:\n            features: tensor of features\n            labels: labels of the samples in the batch, list or Tensor\n                of shape (batch_size;)\n\n        Returns: the batch of triplets\n\n        Raises:\n            NotImplementedError: you should implement it\n        '
        raise NotImplementedError()

class InBatchTripletsSampler(IInbatchTripletSampler):
    """
    Base class for a triplets samplers.
    We expect that the child instances of this class
    will be used to forming triplets inside the batches.
    (Note. It is assumed that set of output features is a
    subset of samples features inside the batch.)
    The batches must contain at least 2 samples for
    each class and at least 2 different classes,
    such behaviour can be garantee via using
    catalyst.data.sampler.BatchBalanceClassSampler

    But you are not limited to using it in any other way.
    """

    def _check_input_labels(self, labels: List[int]) -> None:
        if False:
            while True:
                i = 10
        '\n        The input must satisfy the conditions described in\n        the class documentation.\n\n        Args:\n            labels: labels of the samples in the batch\n        '
        labels_counter = Counter(labels)
        assert all((n > 1 for n in labels_counter.values()))
        assert len(labels_counter) > 1

    @abstractmethod
    def _sample(self, features: Tensor, labels: List[int]) -> TTripletsIds:
        if False:
            return 10
        '\n        This method includes the logic of sampling/selecting triplets\n        inside the batch. It can be based on information about\n        the distance between the features, or the\n        choice can be made randomly.\n\n        Args:\n            features: has the shape of [batch_size, feature_size]\n            labels: labels of the samples in the batch\n\n        Returns: indices of the batch samples to forming triplets.\n        '
        raise NotImplementedError

    def sample(self, features: Tensor, labels: TLabels) -> TTriplets:
        if False:
            return 10
        '\n        Args:\n            features: has the shape of [batch_size, feature_size]\n            labels: labels of the samples in the batch\n\n        Returns:\n            the batch of the triplets in the order below:\n            (anchor, positive, negative)\n        '
        labels = convert_labels2list(labels)
        self._check_input_labels(labels=labels)
        (ids_anchor, ids_pos, ids_neg) = self._sample(features, labels=labels)
        return (features[ids_anchor], features[ids_pos], features[ids_neg])

class AllTripletsSampler(InBatchTripletsSampler):
    """
    This sampler selects all the possible triplets for the given labels
    """

    def __init__(self, max_output_triplets: int=maxsize):
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            max_output_triplets: with the strategy of choosing all\n                the triplets, their number in the batch can be very large,\n                because of it we can sample only random part of them,\n                determined by this parameter.\n        '
        self._max_out_triplets = max_output_triplets

    def _sample(self, *_: Tensor, labels: List[int]) -> TTripletsIds:
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            labels: labels of the samples in the batch\n            *_: note, that we ignore features argument\n\n        Returns:\n            indices of triplets\n        '
        num_labels = len(labels)
        triplets = []
        for label in set(labels):
            ids_pos_cur = set(find_value_ids(labels, label))
            ids_neg_cur = set(range(num_labels)) - ids_pos_cur
            pos_pairs = list(combinations(ids_pos_cur, r=2))
            tri = [(a, p, n) for ((a, p), n) in product(pos_pairs, ids_neg_cur)]
            triplets.extend(tri)
        triplets = sample(triplets, min(len(triplets), self._max_out_triplets))
        (ids_anchor, ids_pos, ids_neg) = zip(*triplets)
        return (list(ids_anchor), list(ids_pos), list(ids_neg))

class HardTripletsSampler(InBatchTripletsSampler):
    """
    This sampler selects hardest triplets based on distances between features:
    the hardest positive sample has the maximal distance to the anchor sample,
    the hardest negative sample has the minimal distance to the anchor sample.

    Note that a typical triplet loss chart is as follows:
    1. Falling: loss decreases to a value equal to the margin.
    2. Long plato: the loss oscillates near the margin.
    3. Falling: loss decreases to zero.

    """

    def __init__(self, norm_required: bool=False):
        if False:
            return 10
        '\n        Args:\n            norm_required: set True if features normalisation is needed\n        '
        self._norm_required = norm_required

    def _sample(self, features: Tensor, labels: List[int]) -> TTripletsIds:
        if False:
            while True:
                i = 10
        '\n        This method samples the hardest triplets inside the batch.\n\n        Args:\n            features: has the shape of [batch_size, feature_size]\n            labels: labels of the samples in the batch\n\n        Returns:\n            the batch of the triplets in the order below:\n            (anchor, positive, negative)\n        '
        assert features.shape[0] == len(labels)
        if self._norm_required:
            features = F.normalize(features.detach(), p=2, dim=1)
        dist_mat = torch.cdist(x1=features, x2=features, p=2)
        (ids_anchor, ids_pos, ids_neg) = self._sample_from_distmat(distmat=dist_mat, labels=labels)
        return (ids_anchor, ids_pos, ids_neg)

    @staticmethod
    def _sample_from_distmat(distmat: Tensor, labels: List[int]) -> TTripletsIds:
        if False:
            while True:
                i = 10
        '\n        This method samples the hardest triplets based on the given\n        distances matrix. It chooses each sample in the batch as an\n        anchor and then finds the harderst positive and negative pair.\n\n        Args:\n            distmat: matrix of distances between the features\n            labels: labels of the samples in the batch\n\n        Returns:\n            the batch of triplets in the order below:\n            (anchor, positive, negative)\n        '
        ids_all = set(range(len(labels)))
        (ids_anchor, ids_pos, ids_neg) = ([], [], [])
        for (i_anch, label) in enumerate(labels):
            ids_label = set(find_value_ids(it=labels, value=label))
            ids_pos_cur = np.array(list(ids_label - {i_anch}), int)
            ids_neg_cur = np.array(list(ids_all - ids_label), int)
            i_pos = ids_pos_cur[distmat[i_anch, ids_pos_cur].argmax()]
            i_neg = ids_neg_cur[distmat[i_anch, ids_neg_cur].argmin()]
            ids_anchor.append(i_anch)
            ids_pos.append(i_pos)
            ids_neg.append(i_neg)
        return (ids_anchor, ids_pos, ids_neg)

class HardClusterSampler(IInbatchTripletSampler):
    """
    This sampler selects hardest triplets based on distance to mean vectors:
    anchor is a mean vector of features of i-th class in the batch,
    the hardest positive sample is the most distant from anchor sample of
    anchor's class, the hardest negative sample is the closest mean vector
    of another classes.

    The batch must contain k samples for p classes in it (k > 1, p > 1).
    """

    def _check_input_labels(self, labels: List[int]) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Check if the labels list is valid: contains k occurrences\n        for each of p classes.\n\n        Args:\n            labels: labels in the batch\n\n        Raises:\n            ValueError: if batch is invalid (contains different samples\n                for classes, contains only one class or only one sample for\n                each class)\n        '
        labels_counter = Counter(labels)
        k = labels_counter[labels[0]]
        if not all((n == k for n in labels_counter.values())):
            raise ValueError('Expected equal number of samples for each class')
        if len(labels_counter) <= 1:
            raise ValueError('Expected at least 2 classes in the batch')
        if k == 1:
            raise ValueError('Expected more than one sample for each class')

    @staticmethod
    def _get_labels_mask(labels: List[int]) -> Tensor:
        if False:
            print('Hello World!')
        '\n        Generate matrix of bool of shape (n_unique_labels, batch_size),\n        where n_unique_labels is a number of unique labels\n        in the batch; matrix[i, j] is True if j-th element of\n        the batch relates to i-th class and False otherwise.\n\n        Args:\n            labels: labels of the batch, shape (batch_size)\n\n        Returns:\n            matrix of indices of classes in batch\n        '
        unique_labels = sorted(np.unique(labels))
        labels_number = len(unique_labels)
        labels_mask = torch.zeros(size=(labels_number, len(labels)))
        for (label_idx, label) in enumerate(unique_labels):
            label_indices = find_value_ids(labels, label)
            labels_mask[label_idx][label_indices] = 1
        return labels_mask.type(torch.bool)

    @staticmethod
    def _count_intra_class_distances(embeddings: Tensor, mean_vectors: Tensor) -> Tensor:
        if False:
            return 10
        "\n        Count matrix of distances from mean vector of each class to it's\n        samples embeddings.\n\n        Args:\n            embeddings: tensor of shape (p, k, embed_dim) where p is a number\n                of classes in the batch, k is a number of samples for each\n                class\n            mean_vectors: tensor of shape (p, embed_dim) -- mean vectors\n                of each class in the batch\n\n        Returns:\n            tensor of shape (p, k) -- matrix of distances from mean vectors to\n                related samples in the batch\n        "
        (p, k, embed_dim) = embeddings.shape
        mean_vectors = mean_vectors.unsqueeze(1).repeat((1, k, 1))
        distances = torch.pow(embeddings - mean_vectors, 2).sum(2)
        return distances

    @staticmethod
    def _count_inter_class_distances(mean_vectors: Tensor) -> Tensor:
        if False:
            while True:
                i = 10
        '\n        Count matrix of distances from mean vectors of classes to each other\n\n        Args:\n            mean_vectors: tensor of shape (p, embed_dim) -- mean vectors\n                of classes\n\n        Returns:\n            tensor of shape (p, p) -- matrix of distances between mean vectors\n        '
        distance = torch.cdist(x1=mean_vectors, x2=mean_vectors, p=2)
        return distance

    @staticmethod
    def _fill_diagonal(matrix: Tensor, value: float) -> Tensor:
        if False:
            for i in range(10):
                print('nop')
        '\n        Set diagonal elements with the value.\n\n        Args:\n            matrix: tensor of shape (p, p)\n            value: value that diagonal should be filled with\n\n        Returns:\n            modified matrix with inf on diagonal\n        '
        (p, _) = matrix.shape
        indices = torch.diag(torch.ones(p)).type(torch.bool)
        matrix[indices] = value
        return matrix

    def sample(self, features: Tensor, labels: TLabels) -> TTriplets:
        if False:
            return 10
        '\n        This method samples the hardest triplets in the batch.\n\n        Args:\n            features: tensor of shape (batch_size; embed_dim) that contains\n                k samples for each of p classes\n            labels: labels of the batch, list or tensor of size (batch_size)\n\n        Returns:\n            p triplets of (mean_vector, positive, negative_mean_vector)\n        '
        labels = convert_labels2list(labels)
        self._check_input_labels(labels)
        labels_mask = self._get_labels_mask(labels)
        p = labels_mask.shape[0]
        embed_dim = features.shape[-1]
        features = features.repeat((p, 1, 1))
        features = features[labels_mask].view((p, -1, embed_dim))
        mean_vectors = features.mean(1)
        d_intra = self._count_intra_class_distances(features, mean_vectors)
        pos_indices = d_intra.max(1).indices
        d_inter = self._count_inter_class_distances(mean_vectors)
        d_inter = self._fill_diagonal(d_inter, float('inf'))
        neg_indices = d_inter.min(1).indices
        positives = torch.stack([features[idx][pos_idx] for (idx, pos_idx) in enumerate(pos_indices)])
        return (mean_vectors, positives, mean_vectors[neg_indices])
__all__ = ['IInbatchTripletSampler', 'InBatchTripletsSampler', 'AllTripletsSampler', 'HardTripletsSampler', 'HardClusterSampler']