"""Add sinusoid features to the features set"""
import numpy as np

def generate_sinusoids(dataset, sinusoid_degree):
    if False:
        i = 10
        return i + 15
    'Extends data set with sinusoid features.\n\n    Returns a new feature array with more features, comprising of\n    sin(x).\n\n    :param dataset: data set.\n    :param sinusoid_degree: multiplier for sinusoid parameter multiplications\n    '
    num_examples = dataset.shape[0]
    sinusoids = np.empty((num_examples, 0))
    for degree in range(1, sinusoid_degree + 1):
        sinusoid_features = np.sin(degree * dataset)
        sinusoids = np.concatenate((sinusoids, sinusoid_features), axis=1)
    return sinusoids