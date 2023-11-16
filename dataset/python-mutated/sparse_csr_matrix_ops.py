"""CSR Sparse Matrix Operations."""
import abc
import collections
from tensorflow.python.eager import context
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.linalg.sparse import gen_sparse_csr_matrix_ops as sm_ops
from tensorflow.python.ops.linalg.sparse.gen_sparse_csr_matrix_ops import *
__all__ = ['SparseMatrix', 'CSRSparseMatrix', 'matmul', 'dense_shape_and_type']
__all__ += [_x for _x in dir(sm_ops) if not _x.startswith('_')]

class DenseShapeAndType(collections.namedtuple('DenseShapeAndType', ('shape', 'dtype'))):
    pass

def _get_handle_data(tensor):
    if False:
        while True:
            i = 10
    return resource_variable_ops.get_eager_safe_handle_data(tensor)

def _create_handle_data_proto(shape_proto, dtype_enum):
    if False:
        i = 10
        return i + 15
    'Create handle data based on shape and dtype protos.'
    variant_shape_and_type_data = cpp_shape_inference_pb2.CppShapeInferenceResult.HandleData()
    variant_shape_and_type_data.is_set = True
    variant_shape_and_type_data.shape_and_type.extend([cpp_shape_inference_pb2.CppShapeInferenceResult.HandleShapeAndType(shape=shape_proto, dtype=dtype_enum)])
    return variant_shape_and_type_data

def _make_handle_data(tensor):
    if False:
        return 10
    'Create handle data based on tensor shape and dtype.'
    return _create_handle_data_proto(tensor.shape.as_proto(), tensor.dtype.as_datatype_enum)

def get_shape_and_type(matrix):
    if False:
        print('Hello World!')
    "Return matrix's shape and type if available."
    handle_data = getattr(matrix, '_handle_data', None)
    if handle_data is None:
        return None
    if len(handle_data.shape_and_type) != 1:
        raise ValueError('shape_and_type array in _handle_data must have length one, but saw: %d' % len(handle_data.shape_and_type))
    return handle_data.shape_and_type[0]

def dense_shape_and_type(matrix):
    if False:
        i = 10
        return i + 15
    'Get dense shape and dtype of the tf.Tensor containing the matrix.\n\n  Args:\n    matrix: A `tf.Tensor` of type `tf.variant` storing a sparse matrix.\n\n  Returns:\n    An instance of `ShapeAndType` with properties `shape` (a `tf.TensorShape`)\n    and `dtype` (a `tf.DType`).\n\n  Raises:\n    TypeError: if `matrix` is not a tensor or its dtype is not variant.\n    ValueError: if `matrix` lacks static handle data containing the dense\n      shape and dtype.\n  '
    if not isinstance(matrix, tensor_lib.Tensor):
        raise TypeError('matrix should be a tensor, but saw: %s' % (matrix,))
    if matrix.dtype != dtypes.variant:
        raise TypeError('expected matrix to be type tf.variant, but saw: %s' % (matrix.dtype,))
    handle_data = _get_handle_data(matrix)
    if not handle_data or not handle_data.is_set:
        raise ValueError('matrix has missing handle data: %s' % (matrix,))
    if len(handle_data.shape_and_type) != 1:
        raise ValueError("len(matrix.handle_data.shape_and_type) != 1: '%s'" % (handle_data.shape_and_type,))
    return DenseShapeAndType(tensor_shape.TensorShape(handle_data.shape_and_type[0].shape), dtypes.DType(handle_data.shape_and_type[0].dtype))

def matmul_shape_inference(a, b, c, transpose_a, transpose_b, adjoint_a, adjoint_b):
    if False:
        for i in range(10):
            print('nop')
    "Helper function for matmul to set the result matrix's handle data."
    c_handle = getattr(c, '_handle_data', None)
    a_shape_and_type = get_shape_and_type(a)
    b_shape_and_type = get_shape_and_type(b)
    if c_handle is None and a_shape_and_type is not None and (b_shape_and_type is not None):
        transpose_a = transpose_a or adjoint_a
        transpose_b = transpose_b or adjoint_b
        a_shape = a_shape_and_type.shape
        b_shape = b_shape_and_type.shape
        rank = len(a_shape.dim)
        c_rows = a_shape.dim[rank - (1 if transpose_a else 2)].size
        c_cols = b_shape.dim[rank - (2 if transpose_b else 1)].size
        c_shape = tensor_shape.TensorShape(a_shape)
        c_shape = tensor_shape.TensorShape(c_shape[:rank - 2] + [c_rows, c_cols])
        c_handle = _create_handle_data_proto(c_shape.as_proto(), a_shape_and_type.dtype)
    return c_handle

def matmul(a, b, transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False, name=None):
    if False:
        for i in range(10):
            print('nop')
    'Perform a sparse matrix matmul between `a` and `b`.\n\n  Performs a contraction between `a` and `b` along the two innermost dimensions.\n  If both `a` and `b` are instances of `SparseMatrix`, returns a new instance\n  of `SparseMatrix` (same type as `a`).  If one is not an instance of\n  `SparseMatrix`, returns a dense `Tensor`:\n\n  ```\n  c = opA(a) . opB(b)\n  ```\n  where `opA` (resp. `opB`) is the transpose or hermitian transpose depending\n  on the values of `transpose_a` (resp. `transpose_b`) and `adjoint_a`\n  (resp. `adjoint_b`).\n\n  Args:\n    a: `Tensor` or `SparseMatrix`, having rank `2` or `3`.\n    b: `Tensor` or `SparseMatrix`, having rank `2` or `3`.\n    transpose_a: Python `bool`.\n    transpose_b: Python `bool`.\n    adjoint_a: Python `bool`.\n    adjoint_b: Python `bool`.\n    name: Optional name to use when creating ops.\n\n  Returns:\n    A `SparseMatrix` if both `a` and `b` are instances of `SparseMatrix`,\n    otherwise a dense `Tensor`.\n  '
    if not isinstance(a, SparseMatrix) and (not isinstance(b, SparseMatrix)):
        return math_ops.matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b, adjoint_a=adjoint_a, adjoint_b=adjoint_b, name=name)
    a_matrix = a._matrix if isinstance(a, SparseMatrix) else a
    b_matrix = b._matrix if isinstance(b, SparseMatrix) else b
    with ops.name_scope(name, 'SparseMatrixMatMul', [a_matrix, b_matrix]):
        if isinstance(a, SparseMatrix) and isinstance(b, SparseMatrix):
            if not (isinstance(a, type(b)) or isinstance(b, type(a))):
                raise TypeError("SparseMatrix types don't inherit from each other: %s and %s" % (type(a), type(b)))
            c = sm_ops.sparse_matrix_sparse_mat_mul(a_matrix, b_matrix, transpose_a=transpose_a, transpose_b=transpose_b, adjoint_a=adjoint_a, adjoint_b=adjoint_b, type=a.dtype)
            c_handle = matmul_shape_inference(a_matrix, b_matrix, c, transpose_a, transpose_b, adjoint_a, adjoint_b)
            return a._from_matrix(c, handle_data=c_handle)
        elif isinstance(a, SparseMatrix):
            return sm_ops.sparse_matrix_mat_mul(a_matrix, b, transpose_a=transpose_a, transpose_b=transpose_b, adjoint_a=adjoint_a, adjoint_b=adjoint_b)
        elif not adjoint_a and (not adjoint_b):
            return sm_ops.sparse_matrix_mat_mul(b_matrix, a, transpose_a=not transpose_b, transpose_b=not transpose_a, transpose_output=True)
        elif not transpose_a and (not transpose_b):
            return sm_ops.sparse_matrix_mat_mul(b_matrix, a, adjoint_a=not adjoint_b, adjoint_b=not adjoint_a, transpose_output=True, conjugate_output=True)
        else:
            return sm_ops.sparse_matrix_mat_mul(b_matrix, math_ops.conj(a), transpose_output=True, conjugate_output=adjoint_b)

class SparseMatrix(metaclass=abc.ABCMeta):
    """Abstract class for sparse matrix types."""

    @abc.abstractmethod
    def __init__(self):
        if False:
            while True:
                i = 10
        self._eager_mode = context.executing_eagerly()

    @abc.abstractproperty
    def _matrix(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @abc.abstractmethod
    def _from_matrix(self, matrix, handle_data=None):
        if False:
            print('Hello World!')
        pass

    @abc.abstractmethod
    def to_dense(self):
        if False:
            i = 10
            return i + 15
        pass

    @abc.abstractmethod
    def to_sparse_tensor(self):
        if False:
            while True:
                i = 10
        pass

    @property
    def graph(self):
        if False:
            return 10
        return self._matrix.graph

    @property
    def shape(self):
        if False:
            return 10
        return dense_shape_and_type(self._matrix).shape

    @property
    def dtype(self):
        if False:
            return 10
        return dense_shape_and_type(self._matrix).dtype

    @property
    def eager_handle_data(self):
        if False:
            print('Hello World!')
        "Return the matrix's handle data iff in eager mode."
        return _get_handle_data(self._matrix) if self._eager_mode else None

    def conj(self):
        if False:
            i = 10
            return i + 15
        return self._from_matrix(math_ops.conj(self._matrix), self.eager_handle_data)

    def hermitian_transpose(self):
        if False:
            i = 10
            return i + 15
        'Return the hermitian transpose of the matrix.'
        return self._from_matrix(sm_ops.sparse_matrix_transpose(self._matrix, conjugate=True, type=self.dtype), self.eager_handle_data)

    def nnz(self):
        if False:
            for i in range(10):
                print('nop')
        'Number of stored values, including explicit zeros.'
        return sm_ops.sparse_matrix_nnz(self._matrix)
    nonzero = nnz

    def sorted_indices(self):
        if False:
            print('Hello World!')
        return self.to_sparse_tensor().indices

    def transpose(self):
        if False:
            return 10
        return self._from_matrix(sm_ops.sparse_matrix_transpose(self._matrix, type=self.dtype), self.eager_handle_data)

class CSRSparseMatrix(SparseMatrix):
    """(Optionally batched) CSR Sparse Matrix."""

    def __init__(self, value, indices=None, name=None):
        if False:
            while True:
                i = 10
        'Construct a CSRSparseMatrix from a dense matrix or SparseTensor.\n\n    Args:\n      value: A dense `2D` or `3D` Tensor or `SparseTensor`.\n      indices: The nonzero indices of `value`\n        (if `value` is not a `SparseTensor`).\n      name: Optional op name.\n\n    Raises:\n      ValueError: if `value` is a `SparseTensor` and `indices` is not `None`.\n    '
        del name
        super(CSRSparseMatrix, self).__init__()
        if isinstance(value, sparse_tensor.SparseTensor):
            if indices is not None:
                raise ValueError('indices must be None if value is a SparseTensor.')
            self._dtype = value.dtype
            self._csr_matrix = sm_ops.sparse_tensor_to_csr_sparse_matrix(indices=value.indices, values=value.values, dense_shape=value.dense_shape)
        else:
            value = ops.convert_to_tensor(value)
            self._dtype = value.dtype
            if indices is not None:
                indices = ops.convert_to_tensor(indices, dtype=dtypes.int64)
            else:
                indices = array_ops.stop_gradient(array_ops.where(value))
            self._csr_matrix = sm_ops.dense_to_csr_sparse_matrix(value, indices)
        if self._eager_mode:
            self._csr_matrix._handle_data = _make_handle_data(value)

    @property
    def _matrix(self):
        if False:
            while True:
                i = 10
        return self._csr_matrix

    def _from_matrix(self, matrix, handle_data=None):
        if False:
            return 10
        assert isinstance(matrix, tensor_lib.Tensor) and matrix.dtype == dtypes.variant
        ret = type(self).__new__(type(self))
        ret._dtype = self._dtype
        if self._eager_mode:
            if matrix._handle_data is None:
                matrix._handle_data = handle_data
            assert matrix._handle_data is not None
        ret._csr_matrix = matrix
        return ret

    def to_dense(self):
        if False:
            for i in range(10):
                print('nop')
        return sm_ops.csr_sparse_matrix_to_dense(self._matrix, type=self.dtype)

    def to_sparse_tensor(self):
        if False:
            print('Hello World!')
        r = sm_ops.csr_sparse_matrix_to_sparse_tensor(self._matrix, type=self.dtype)
        return sparse_tensor.SparseTensor(indices=r.indices, values=r.values, dense_shape=r.dense_shape)