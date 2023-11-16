"""Quasi Monte Carlo support: Sobol sequence.

A TensorFlow implementation of Sobol sequences, a type of quasi-random
low-discrepancy sequence: https://en.wikipedia.org/wiki/Sobol_sequence.
"""
import logging
import os
from typing import Optional, Tuple
import numpy as np
from six.moves import range
import tensorflow.compat.v2 as tf
from tf_quant_finance import types
__all__ = ['load_data', 'sample']
_LN_2 = np.log(2.0)
_MAX_POSITIVE = 2 ** 31 - 1

def sample(dim: int, num_results: types.IntTensor, skip: types.IntTensor=0, validate_args: bool=False, dtype: tf.dtypes.DType=None, name: Optional[str]=None) -> types.RealTensor:
    if False:
        print('Hello World!')
    "Returns num_results samples from the Sobol sequence of dimension dim.\n\n  Uses the original ordering of points, not the more commonly used Gray code\n  ordering. Derived from notes by Joe & Kuo[1].\n\n  [1] describes bitwise operations on binary floats. The implementation below\n  achieves this by transforming the floats into ints, being careful to align\n  the digits so the bitwise operations are correct, then transforming back to\n  floats.\n\n  Args:\n    dim: Positive Python `int` representing each sample's `event_size.`\n    num_results: Positive scalar `Tensor` of dtype int32. The number of Sobol\n      points to return in the output.\n    skip: Positive scalar `Tensor` of dtype int32. The number of initial points\n      of the Sobol sequence to skip.\n    validate_args: Python `bool`. When `True`, input `Tensor's` are checked for\n      validity despite possibly degrading runtime performance. The checks verify\n      that `dim >= 1`, `num_results >= 1`, `skip >= 0` and whether\n      `num_results + skip < 2**31 - 1`. When `False` invalid inputs may silently\n      render incorrect outputs.\n      Default value: False.\n    dtype: Optional `dtype`. The dtype of the output `Tensor` (either `float32`\n      or `float64`).\n      Default value: `None` which maps to the `float32`.\n    name: Python `str` name prefixed to ops created by this function.\n\n  Returns:\n    `Tensor` of samples from Sobol sequence with `shape` [n, dim].\n\n  #### References\n\n  [1]: S. Joe and F. Y. Kuo. Notes on generating Sobol sequences. August 2008.\n       https://web.maths.unsw.edu.au/~fkuo/sobol/joe-kuo-notes.pdf\n  "
    with tf.name_scope(name or 'sobol_sample'):
        num_results = tf.convert_to_tensor(num_results, dtype=tf.int32, name='num_results')
        skip = tf.convert_to_tensor(skip, dtype=tf.int32, name='skip')
        control_dependencies = []
        if validate_args:
            if dim < 1:
                raise ValueError('Dimension must be greater than zero. Supplied {}'.format(dim))
            control_dependencies.append(tf.debugging.assert_greater(num_results, 0, message='Number of results `num_results` must be greater than zero.'))
            control_dependencies.append(tf.debugging.assert_greater(skip, 0, message='`skip` must be non-negative.'))
            control_dependencies.append(tf.debugging.assert_greater(_MAX_POSITIVE - num_results, skip, message=f'Skip too large. Should be smaller than {_MAX_POSITIVE} - num_results'))
        with tf.control_dependencies(control_dependencies):
            if validate_args:
                num_results = tf.math.maximum(num_results, 1, name='fix_num_results')
                skip = tf.math.maximum(skip, 0, name='fix_skip')
            direction_numbers = tf.convert_to_tensor(_compute_direction_numbers(dim), name='direction_numbers')
            max_index = tf.cast(skip, dtype=tf.int64) + tf.cast(num_results, dtype=tf.int64) + 1
            num_digits = tf.cast(tf.math.ceil(tf.math.log(tf.cast(max_index, tf.float64)) / _LN_2), tf.int32)
        direction_numbers = tf.bitwise.left_shift(direction_numbers[:dim, :num_digits], tf.range(num_digits - 1, -1, -1))
        direction_numbers = tf.expand_dims(tf.transpose(a=direction_numbers), 1)
        irange = skip + 1 + tf.range(num_results)
        dig_range = tf.expand_dims(tf.range(num_digits), 1)
        binary_matrix = tf.bitwise.bitwise_and(1, tf.bitwise.right_shift(irange, dig_range))
        binary_matrix = tf.expand_dims(binary_matrix, -1)
        product = direction_numbers * binary_matrix

        def _cond(partial_result, i):
            if False:
                i = 10
                return i + 15
            del partial_result
            return i < num_digits

        def _body(partial_result, i):
            if False:
                print('Hello World!')
            return (tf.bitwise.bitwise_xor(partial_result, product[i, :, :]), i + 1)
        (result, _) = tf.while_loop(_cond, _body, (product[0, :, :], 1))
        dtype = dtype or tf.float32
        one = tf.constant(1, dtype=tf.int64)
        divisor = tf.bitwise.left_shift(one, tf.cast(num_digits, dtype=tf.int64))
        return tf.cast(result, dtype) / tf.cast(divisor, dtype)

def _compute_direction_numbers(dim):
    if False:
        for i in range(10):
            print('nop')
    "Returns array of direction numbers for dimension dim.\n\n  These are the m_kj values in the Joe & Kuo notes[1], not the v_kj values. So\n  these refer to the 'abuse of notation' mentioned in the notes -- it is a\n  matrix of integers, not floats. The variable names below are intended to match\n  the notation in the notes as closely as possible.\n\n  Args:\n    dim: int, dimension.\n\n  Returns:\n    `numpy.array` of direction numbers with `shape` [dim, 32].\n  "
    m = np.empty((dim, 32), dtype=np.int32)
    m[0, :] = np.ones(32, dtype=np.int32)
    for k in range(dim - 1):
        a_k = _PRIMITIVE_POLYNOMIAL_COEFFICIENTS[k]
        deg = np.int32(np.floor(np.log2(a_k)))
        m[k + 1, :deg] = _INITIAL_DIRECTION_NUMBERS[:deg, k]
        for j in range(deg, 32):
            m[k + 1, j] = m[k + 1, j - deg]
            for i in range(deg):
                if a_k >> i & 1:
                    m[k + 1, j] = np.bitwise_xor(m[k + 1, j], m[k + 1, j - deg + i] << deg - i)
    return m

def _get_sobol_data_path():
    if False:
        for i in range(10):
            print('nop')
    "Returns path of file 'new-joe-kuo-6.21201'.\n\n     Location of file 'new-joe-kuo-6.21201' depends on the environment in\n     which this code is executed. In Google internal environment file\n     'new-joe-kuo-6.21201' is accessible using the\n     'third_party/sobol_data/new-joe-kuo-6.21201' file path.\n\n     However, this doesn't work in the pip package. In pip package the directory\n     'third_party' is a subdirectory of directory 'tf_quant_finance' and in\n     this case we construct a file path relative to the __file__ file path.\n\n     If this library is installed in editable mode with `pip install -e .`, then\n     the directory 'third_party` will be at the top level of the repository\n     and need to search a level further up relative to __file__.\n  "
    filename = 'new-joe-kuo-6.21201'
    path1 = os.path.join('third_party', 'sobol_data', filename)
    path2 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'third_party', 'sobol_data', filename))
    path3 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'third_party', 'sobol_data', filename))
    path4 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', 'third_party', 'sobol_data', filename))
    paths = [path1, path2, path3, path4]
    for path in paths:
        if os.path.exists(path):
            return path

def load_data() -> Tuple[types.RealTensor, types.RealTensor]:
    if False:
        while True:
            i = 10
    "Parses file 'new-joe-kuo-6.21201'."
    path = _get_sobol_data_path()
    if path is None:
        logging.warning('Unable to find path to sobol data file.')
        return (NotImplemented, NotImplemented)
    header_line = True
    polynomial_coefficients = np.zeros(shape=(21200,), dtype=np.int64)
    direction_numbers = np.zeros(shape=(18, 21200), dtype=np.int64)
    index = 0
    with open(path) as f:
        for line in f:
            if header_line:
                header_line = False
                continue
            tokens = line.split()
            (s, a) = tokens[1:3]
            polynomial_coefficients[index] = 2 ** int(s) + 2 * int(a) + 1
            for (i, m_i) in enumerate(tokens[3:]):
                direction_numbers[i, index] = int(m_i)
            index += 1
    return (polynomial_coefficients, direction_numbers)
(_PRIMITIVE_POLYNOMIAL_COEFFICIENTS, _INITIAL_DIRECTION_NUMBERS) = load_data()