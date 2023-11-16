"""Utils module with methods for fast calculations."""
import typing as t
import numpy as np
import pandas as pd
from deepchecks.core.errors import DeepchecksValueError

def fast_sum_by_row(matrix: np.ndarray) -> np.array:
    if False:
        print('Hello World!')
    'Faster alternative to np.sum(matrix, axis=1).'
    return np.matmul(matrix, np.ones(matrix.shape[1]))

def sequence_to_numpy(sequence: t.Sequence):
    if False:
        while True:
            i = 10
    'Convert a sequence into a numpy array.'
    if isinstance(sequence, np.ndarray):
        return sequence.flatten()
    elif isinstance(sequence, t.List):
        return np.asarray(sequence).flatten()
    elif isinstance(sequence, pd.Series):
        return sequence.to_numpy().flatten()
    else:
        raise DeepchecksValueError('Trying to convert a non sequence into a flat list.')