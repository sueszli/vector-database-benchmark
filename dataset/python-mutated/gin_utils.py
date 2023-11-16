"""Gin configurable utility functions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import gin.tf

@gin.configurable
def gin_sparse_array(size, values, indices, fill_value=0):
    if False:
        print('Hello World!')
    arr = np.zeros(size)
    arr.fill(fill_value)
    arr[indices] = values
    return arr

@gin.configurable
def gin_sum(values):
    if False:
        for i in range(10):
            print('nop')
    result = values[0]
    for value in values[1:]:
        result += value
    return result

@gin.configurable
def gin_range(n):
    if False:
        while True:
            i = 10
    return range(n)