"""Implementation of HyperLogLog

This implements the HyperLogLog algorithm for cardinality estimation, found
in

    Philippe Flajolet, Éric Fusy, Olivier Gandouet and Frédéric Meunier.
        "HyperLogLog: the analysis of a near-optimal cardinality estimation
        algorithm". 2007 Conference on Analysis of Algorithms. Nice, France
        (2007)

"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object

def compute_first_bit(a):
    if False:
        for i in range(10):
            print('nop')
    'Compute the position of the first nonzero bit for each int in an array.'
    bits = np.bitwise_and.outer(a, 1 << np.arange(32))
    bits = bits.cumsum(axis=1).astype(bool)
    return 33 - bits.sum(axis=1)

def compute_hll_array(obj, b):
    if False:
        for i in range(10):
            print('nop')
    if not 8 <= b <= 16:
        raise ValueError('b should be between 8 and 16')
    num_bits_discarded = 32 - b
    m = 1 << b
    hashes = hash_pandas_object(obj, index=False)
    if isinstance(hashes, pd.Series):
        hashes = hashes._values
    hashes = hashes.astype(np.uint32)
    j = hashes >> num_bits_discarded
    first_bit = compute_first_bit(hashes)
    df = pd.DataFrame({'j': j, 'first_bit': first_bit})
    series = df.groupby('j').max()['first_bit']
    return series.reindex(np.arange(m), fill_value=0).values.astype(np.uint8)

def reduce_state(Ms, b):
    if False:
        i = 10
        return i + 15
    m = 1 << b
    Ms = Ms.reshape(len(Ms) // m, m)
    return Ms.max(axis=0)

def estimate_count(Ms, b):
    if False:
        print('Hello World!')
    m = 1 << b
    M = reduce_state(Ms, b)
    alpha = 0.7213 / (1 + 1.079 / m)
    E = alpha * m / (2.0 ** (-M.astype('f8'))).sum() * m
    if E < 2.5 * m:
        V = (M == 0).sum()
        if V:
            return m * np.log(m / V)
    if E > 2 ** 32 / 30.0:
        return -2 ** 32 * np.log1p(-E / 2 ** 32)
    return E