"""Statistics utility functions of NCF."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np

def random_int32():
    if False:
        for i in range(10):
            print('nop')
    return np.random.randint(low=0, high=np.iinfo(np.int32).max, dtype=np.int32)

def permutation(args):
    if False:
        print('Hello World!')
    'Fork safe permutation function.\n\n  This function can be called within a multiprocessing worker and give\n  appropriately random results.\n\n  Args:\n    args: A size two tuple that will unpacked into the size of the permutation\n      and the random seed. This form is used because starmap is not universally\n      available.\n\n  returns:\n    A NumPy array containing a random permutation.\n  '
    (x, seed) = args
    state = np.random.RandomState(seed=seed)
    output = np.arange(x, dtype=np.int32)
    state.shuffle(output)
    return output

def very_slightly_biased_randint(max_val_vector):
    if False:
        while True:
            i = 10
    sample_dtype = np.uint64
    out_dtype = max_val_vector.dtype
    samples = np.random.randint(low=0, high=np.iinfo(sample_dtype).max, size=max_val_vector.shape, dtype=sample_dtype)
    return np.mod(samples, max_val_vector.astype(sample_dtype)).astype(out_dtype)

def mask_duplicates(x, axis=1):
    if False:
        for i in range(10):
            print('nop')
    'Identify duplicates from sampling with replacement.\n\n  Args:\n    x: A 2D NumPy array of samples\n    axis: The axis along which to de-dupe.\n\n  Returns:\n    A NumPy array with the same shape as x with one if an element appeared\n    previously along axis 1, else zero.\n  '
    if axis != 1:
        raise NotImplementedError
    x_sort_ind = np.argsort(x, axis=1, kind='mergesort')
    sorted_x = x[np.arange(x.shape[0])[:, np.newaxis], x_sort_ind]
    inv_x_sort_ind = np.argsort(x_sort_ind, axis=1, kind='mergesort')
    diffs = sorted_x[:, :-1] - sorted_x[:, 1:]
    diffs = np.concatenate([np.ones((diffs.shape[0], 1), dtype=diffs.dtype), diffs], axis=1)
    return np.where(diffs[np.arange(x.shape[0])[:, np.newaxis], inv_x_sort_ind], 0, 1)