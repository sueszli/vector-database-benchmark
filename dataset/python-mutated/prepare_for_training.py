"""Prepares the dataset for training"""
import numpy as np
from .normalize import normalize
from .generate_sinusoids import generate_sinusoids
from .generate_polynomials import generate_polynomials

def prepare_for_training(data, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
    if False:
        i = 10
        return i + 15
    'Prepares data set for training on prediction'
    num_examples = data.shape[0]
    data_processed = np.copy(data)
    features_mean = 0
    features_deviation = 0
    data_normalized = data_processed
    if normalize_data:
        (data_normalized, features_mean, features_deviation) = normalize(data_processed)
        data_processed = data_normalized
    if sinusoid_degree > 0:
        sinusoids = generate_sinusoids(data_normalized, sinusoid_degree)
        data_processed = np.concatenate((data_processed, sinusoids), axis=1)
    if polynomial_degree > 0:
        polynomials = generate_polynomials(data_normalized, polynomial_degree, normalize_data)
        data_processed = np.concatenate((data_processed, polynomials), axis=1)
    data_processed = np.hstack((np.ones((num_examples, 1)), data_processed))
    return (data_processed, features_mean, features_deviation)