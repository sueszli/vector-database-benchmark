from typing import Sequence
import numpy as np
from math import floor

def split(ds, values: Sequence[float]=[0.7, 0.2, 0.1]):
    if False:
        for i in range(10):
            print('nop')
    'Splits a Dataset into multiple datasets with the provided ratio of entries.\n    Returns a list of datasets with length equal to the number of givens.\n    For small datasets or many partitions, some returns may be empty.\n\n    Args:\n        ds: The Dataset object from which to construct the splits.\n            If already indexed, the splits will be based off that index.\n        values: The proportions for the split. Should each sum to one.\n            Defaults to [0.7, 0.2, 0.1] for a 70% 20% 10% split\n\n    Returns:\n        List of Datasets, one for each float in the given values.\n\n    Raises:\n        ValueError: The values must sum to 1.\n    '
    if not np.isclose(sum(values), 1.0):
        raise ValueError('Given proportions must sum to 1.')
    count = 0
    length = len(ds)
    partitions = []
    for value in values[:-1]:
        amount = floor(length * value)
        partitions.append(ds[count:count + amount])
        count += amount
    partitions.append(ds[count:])
    return partitions