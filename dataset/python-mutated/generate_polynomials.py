"""Add polynomial features to the features set"""
import numpy as np
from .normalize import normalize

def generate_polynomials(dataset, polynomial_degree, normalize_data=False):
    if False:
        i = 10
        return i + 15
    'Extends data set with polynomial features of certain degree.\n\n    Returns a new feature array with more features, comprising of\n    x1, x2, x1^2, x2^2, x1*x2, x1*x2^2, etc.\n\n    :param dataset: dataset that we want to generate polynomials for.\n    :param polynomial_degree: the max power of new features.\n    :param normalize_data: flag that indicates whether polynomials need to normalized or not.\n    '
    features_split = np.array_split(dataset, 2, axis=1)
    dataset_1 = features_split[0]
    dataset_2 = features_split[1]
    (num_examples_1, num_features_1) = dataset_1.shape
    (num_examples_2, num_features_2) = dataset_2.shape
    if num_examples_1 != num_examples_2:
        raise ValueError('Can not generate polynomials for two sets with different number of rows')
    if num_features_1 == 0 and num_features_2 == 0:
        raise ValueError('Can not generate polynomials for two sets with no columns')
    if num_features_1 == 0:
        dataset_1 = dataset_2
    elif num_features_2 == 0:
        dataset_2 = dataset_1
    num_features = num_features_1 if num_features_1 < num_examples_2 else num_features_2
    dataset_1 = dataset_1[:, :num_features]
    dataset_2 = dataset_2[:, :num_features]
    polynomials = np.empty((num_examples_1, 0))
    for i in range(1, polynomial_degree + 1):
        for j in range(i + 1):
            polynomial_feature = dataset_1 ** (i - j) * dataset_2 ** j
            polynomials = np.concatenate((polynomials, polynomial_feature), axis=1)
    if normalize_data:
        polynomials = normalize(polynomials)[0]
    return polynomials