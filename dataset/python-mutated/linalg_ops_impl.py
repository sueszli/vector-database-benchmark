"""Operations for linear algebra."""
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import compat

def eye(num_rows, num_columns=None, batch_shape=None, dtype=dtypes.float32, name=None):
    if False:
        while True:
            i = 10
    'Construct an identity matrix, or a batch of matrices.\n\n  See `linalg_ops.eye`.\n  '
    with ops.name_scope(name, default_name='eye', values=[num_rows, num_columns, batch_shape]):
        is_square = num_columns is None
        batch_shape = [] if batch_shape is None else batch_shape
        num_columns = num_rows if num_columns is None else num_columns
        if isinstance(num_rows, tensor.Tensor) or isinstance(num_columns, tensor.Tensor):
            diag_size = math_ops.minimum(num_rows, num_columns)
        else:
            if not isinstance(num_rows, compat.integral_types) or not isinstance(num_columns, compat.integral_types):
                raise TypeError(f'Arguments `num_rows` and `num_columns` must be positive integer values. Received: num_rows={num_rows}, num_columns={num_columns}')
            is_square = num_rows == num_columns
            diag_size = np.minimum(num_rows, num_columns)
        if isinstance(batch_shape, tensor.Tensor) or isinstance(diag_size, tensor.Tensor):
            batch_shape = ops.convert_to_tensor(batch_shape, name='shape', dtype=dtypes.int32)
            diag_shape = array_ops.concat((batch_shape, [diag_size]), axis=0)
            if not is_square:
                shape = array_ops.concat((batch_shape, [num_rows, num_columns]), axis=0)
        else:
            batch_shape = list(batch_shape)
            diag_shape = batch_shape + [diag_size]
            if not is_square:
                shape = batch_shape + [num_rows, num_columns]
        diag_ones = array_ops.ones(diag_shape, dtype=dtype)
        if is_square:
            return array_ops.matrix_diag(diag_ones)
        else:
            zero_matrix = array_ops.zeros(shape, dtype=dtype)
            return array_ops.matrix_set_diag(zero_matrix, diag_ones)