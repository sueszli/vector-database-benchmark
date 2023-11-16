"""`LinearOperator` coming from a [[nested] block] circulant matrix."""
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import util as distribution_util
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.ops.linalg import property_hint_util
from tensorflow.python.ops.signal import fft_ops
from tensorflow.python.util.tf_export import tf_export
__all__ = ['LinearOperatorCirculant', 'LinearOperatorCirculant2D', 'LinearOperatorCirculant3D']
_FFT_OP = {1: fft_ops.fft, 2: fft_ops.fft2d, 3: fft_ops.fft3d}
_IFFT_OP = {1: fft_ops.ifft, 2: fft_ops.ifft2d, 3: fft_ops.ifft3d}

def exponential_power_convolution_kernel(grid_shape, length_scale, power=None, divisor=None, zero_inflation=None):
    if False:
        return 10
    'Make an exponentiated convolution kernel.\n\n  In signal processing, a [kernel]\n  (https://en.wikipedia.org/wiki/Kernel_(image_processing)) `h` can be convolved\n  with a signal `x` to filter its spectral content.\n\n  This function makes a `d-dimensional` convolution kernel `h` of shape\n  `grid_shape = [N0, N1, ...]`. For `n` a multi-index with `n[i] < Ni / 2`,\n\n  ```h[n] = exp{sum(|n / (length_scale * grid_shape)|**power) / divisor}.```\n\n  For other `n`, `h` is extended to be circularly symmetric. That is\n\n  ```h[n0 % N0, ...] = h[(-n0) % N0, ...]```\n\n  Since `h` is circularly symmetric and real valued, `H = FFTd[h]` is the\n  spectrum of a symmetric (real) circulant operator `A`.\n\n  #### Example uses\n\n  ```\n  # Matern one-half kernel, d=1.\n  # Will be positive definite without zero_inflation.\n  h = exponential_power_convolution_kernel(\n      grid_shape=[10], length_scale=[0.1], power=1)\n  A = LinearOperatorCirculant(\n      tf.signal.fft(tf.cast(h, tf.complex64)),\n      is_self_adjoint=True, is_positive_definite=True)\n\n  # Gaussian RBF kernel, d=3.\n  # Needs zero_inflation since `length_scale` is long enough to cause aliasing.\n  h = exponential_power_convolution_kernel(\n      grid_shape=[10, 10, 10], length_scale=[0.1, 0.2, 0.2], power=2,\n      zero_inflation=0.15)\n  A = LinearOperatorCirculant3D(\n      tf.signal.fft3d(tf.cast(h, tf.complex64)),\n      is_self_adjoint=True, is_positive_definite=True)\n  ```\n\n  Args:\n    grid_shape: Length `d` (`d` in {1, 2, 3}) list-like of Python integers. The\n      shape of the grid on which the convolution kernel is defined.\n    length_scale: Length `d` `float` `Tensor`. The scale at which the kernel\n      decays in each direction, as a fraction of `grid_shape`.\n    power: Scalar `Tensor` of same `dtype` as `length_scale`, default `2`.\n      Higher (lower) `power` results in nearby points being more (less)\n      correlated, and far away points being less (more) correlated.\n    divisor: Scalar `Tensor` of same `dtype` as `length_scale`. The slope of\n      decay of `log(kernel)` in terms of fractional grid points, along each\n      axis, at `length_scale`, is `power/divisor`. By default, `divisor` is set\n      to `power`. This means, by default, `power=2` results in an exponentiated\n      quadratic (Gaussian) kernel, and `power=1` is a Matern one-half.\n    zero_inflation: Scalar `Tensor` of same `dtype` as `length_scale`, in\n      `[0, 1]`. Let `delta` be the Kronecker delta. That is,\n      `delta[0, ..., 0] = 1` and all other entries are `0`. Then\n      `zero_inflation` modifies the return value via\n      `h --> (1 - zero_inflation) * h + zero_inflation * delta`. This may be\n      needed to ensure a positive definite kernel, especially if `length_scale`\n      is large enough for aliasing and `power > 1`.\n\n  Returns:\n    `Tensor` of shape `grid_shape` with same `dtype` as `length_scale`.\n  '
    nd = len(grid_shape)
    length_scale = tensor_conversion.convert_to_tensor_v2_with_dispatch(length_scale, name='length_scale')
    dtype = length_scale.dtype
    power = 2.0 if power is None else power
    power = tensor_conversion.convert_to_tensor_v2_with_dispatch(power, name='power', dtype=dtype)
    divisor = power if divisor is None else divisor
    divisor = tensor_conversion.convert_to_tensor_v2_with_dispatch(divisor, name='divisor', dtype=dtype)
    zero = math_ops.cast(0.0, dtype)
    one = math_ops.cast(1.0, dtype)
    ts = [math_ops.linspace(zero, one, num=n) for n in grid_shape]
    log_vals = []
    for (i, x) in enumerate(array_ops.meshgrid(*ts, indexing='ij')):
        midpoint = ts[i][math_ops.cast(math_ops.floor(one / 2.0 * grid_shape[i]), dtypes.int32)]
        log_vals.append(-math_ops.abs((x - midpoint) / length_scale[i]) ** power / divisor)
    kernel = math_ops.exp(fft_ops.ifftshift(sum(log_vals), axes=[-i for i in range(1, nd + 1)]))
    if zero_inflation:
        zero_inflation = tensor_conversion.convert_to_tensor_v2_with_dispatch(zero_inflation, name='zero_inflation', dtype=dtype)
        delta = array_ops.pad(array_ops.reshape(one, [1] * nd), [[0, dim - 1] for dim in grid_shape])
        kernel = (1.0 - zero_inflation) * kernel + zero_inflation * delta
    return kernel

class _BaseLinearOperatorCirculant(linear_operator.LinearOperator):
    """Base class for circulant operators.  Not user facing.

  `LinearOperator` acting like a [batch] [[nested] block] circulant matrix.
  """

    def __init__(self, spectrum: tensor.Tensor, block_depth: int, input_output_dtype=dtypes.complex64, is_non_singular: bool=None, is_self_adjoint: bool=None, is_positive_definite: bool=None, is_square: bool=True, parameters=None, name='LinearOperatorCirculant'):
        if False:
            return 10
        'Initialize an `_BaseLinearOperatorCirculant`.\n\n    Args:\n      spectrum:  Shape `[B1,...,Bb] + N` `Tensor`, where `rank(N) in {1, 2, 3}`.\n        Allowed dtypes: `float16`, `float32`, `float64`, `complex64`,\n        `complex128`.  Type can be different than `input_output_dtype`\n      block_depth:  Python integer, either 1, 2, or 3.  Will be 1 for circulant,\n        2 for block circulant, and 3 for nested block circulant.\n      input_output_dtype: `dtype` for input/output.\n      is_non_singular:  Expect that this operator is non-singular.\n      is_self_adjoint:  Expect that this operator is equal to its hermitian\n        transpose.  If `spectrum` is real, this will always be true.\n      is_positive_definite:  Expect that this operator is positive definite,\n        meaning the quadratic form `x^H A x` has positive real part for all\n        nonzero `x`.  Note that we do not require the operator to be\n        self-adjoint to be positive-definite.  See:\n        https://en.wikipedia.org/wiki/Positive-definite_matrix\\\n            #Extension_for_non_symmetric_matrices\n      is_square:  Expect that this operator acts like square [batch] matrices.\n      parameters: Python `dict` of parameters used to instantiate this\n        `LinearOperator`.\n      name:  A name to prepend to all ops created by this class.\n\n    Raises:\n      ValueError:  If `block_depth` is not an allowed value.\n      TypeError:  If `spectrum` is not an allowed type.\n    '
        allowed_block_depths = [1, 2, 3]
        self._name = name
        if block_depth not in allowed_block_depths:
            raise ValueError(f'Argument `block_depth` must be one of {allowed_block_depths}. Received: {block_depth}.')
        self._block_depth = block_depth
        with ops.name_scope(name, values=[spectrum]):
            self._spectrum = self._check_spectrum_and_return_tensor(spectrum)
            if not self.spectrum.dtype.is_complex:
                if is_self_adjoint is False:
                    raise ValueError(f'A real spectrum always corresponds to a self-adjoint operator. Expected argument `is_self_adjoint` to be True when `spectrum.dtype.is_complex` = True. Received: {is_self_adjoint}.')
                is_self_adjoint = True
            if is_square is False:
                raise ValueError(f'A [[nested] block] circulant operator is always square. Expected argument `is_square` to be True. Received: {is_square}.')
            is_square = True
            super(_BaseLinearOperatorCirculant, self).__init__(dtype=dtypes.as_dtype(input_output_dtype), is_non_singular=is_non_singular, is_self_adjoint=is_self_adjoint, is_positive_definite=is_positive_definite, is_square=is_square, parameters=parameters, name=name)

    def _check_spectrum_and_return_tensor(self, spectrum):
        if False:
            while True:
                i = 10
        'Static check of spectrum.  Then return `Tensor` version.'
        spectrum = linear_operator_util.convert_nonref_to_tensor(spectrum, name='spectrum')
        if spectrum.shape.ndims is not None:
            if spectrum.shape.ndims < self.block_depth:
                raise ValueError(f'Argument `spectrum` must have at least {self.block_depth} dimensions. Received: {spectrum}.')
        return spectrum

    @property
    def block_depth(self):
        if False:
            i = 10
            return i + 15
        'Depth of recursively defined circulant blocks defining this `Operator`.\n\n    With `A` the dense representation of this `Operator`,\n\n    `block_depth = 1` means `A` is symmetric circulant.  For example,\n\n    ```\n    A = |w z y x|\n        |x w z y|\n        |y x w z|\n        |z y x w|\n    ```\n\n    `block_depth = 2` means `A` is block symmetric circulant with symmetric\n    circulant blocks.  For example, with `W`, `X`, `Y`, `Z` symmetric circulant,\n\n    ```\n    A = |W Z Y X|\n        |X W Z Y|\n        |Y X W Z|\n        |Z Y X W|\n    ```\n\n    `block_depth = 3` means `A` is block symmetric circulant with block\n    symmetric circulant blocks.\n\n    Returns:\n      Python `integer`.\n    '
        return self._block_depth

    def block_shape_tensor(self):
        if False:
            return 10
        'Shape of the block dimensions of `self.spectrum`.'
        return self._block_shape_tensor()

    def _block_shape_tensor(self, spectrum_shape=None):
        if False:
            print('Hello World!')
        if self.block_shape.is_fully_defined():
            return linear_operator_util.shape_tensor(self.block_shape.as_list(), name='block_shape')
        spectrum_shape = array_ops.shape(self.spectrum) if spectrum_shape is None else spectrum_shape
        return spectrum_shape[-self.block_depth:]

    def _linop_adjoint(self) -> '_BaseLinearOperatorCirculant':
        if False:
            i = 10
            return i + 15
        spectrum = self.spectrum
        if spectrum.dtype.is_complex:
            spectrum = math_ops.conj(spectrum)
        return _BaseLinearOperatorCirculant(spectrum=spectrum, block_depth=self.block_depth, is_non_singular=self.is_non_singular, is_self_adjoint=self.is_self_adjoint, is_positive_definite=self.is_positive_definite, is_square=True)

    def _linop_inverse(self) -> '_BaseLinearOperatorCirculant':
        if False:
            while True:
                i = 10
        return _BaseLinearOperatorCirculant(spectrum=1.0 / self.spectrum, block_depth=self.block_depth, is_non_singular=self.is_non_singular, is_self_adjoint=self.is_self_adjoint, is_positive_definite=self.is_positive_definite, is_square=True, input_output_dtype=self.dtype)

    def _linop_matmul(self, left_operator: '_BaseLinearOperatorCirculant', right_operator: linear_operator.LinearOperator) -> linear_operator.LinearOperator:
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(right_operator, _BaseLinearOperatorCirculant) or not isinstance(left_operator, type(right_operator)):
            return super()._linop_matmul(left_operator, right_operator)
        return _BaseLinearOperatorCirculant(spectrum=left_operator.spectrum * right_operator.spectrum, block_depth=left_operator.block_depth, is_non_singular=property_hint_util.combined_non_singular_hint(left_operator, right_operator), is_self_adjoint=property_hint_util.combined_commuting_self_adjoint_hint(left_operator, right_operator), is_positive_definite=property_hint_util.combined_commuting_positive_definite_hint(left_operator, right_operator), is_square=True)

    def _linop_solve(self, left_operator: '_BaseLinearOperatorCirculant', right_operator: linear_operator.LinearOperator) -> linear_operator.LinearOperator:
        if False:
            i = 10
            return i + 15
        if not isinstance(right_operator, _BaseLinearOperatorCirculant) or not isinstance(left_operator, type(right_operator)):
            return super()._linop_solve(left_operator, right_operator)
        return _BaseLinearOperatorCirculant(spectrum=right_operator.spectrum / left_operator.spectrum, block_depth=left_operator.block_depth, is_non_singular=property_hint_util.combined_non_singular_hint(left_operator, right_operator), is_self_adjoint=property_hint_util.combined_commuting_self_adjoint_hint(left_operator, right_operator), is_positive_definite=property_hint_util.combined_commuting_positive_definite_hint(left_operator, right_operator), is_square=True)

    @property
    def block_shape(self):
        if False:
            print('Hello World!')
        return self.spectrum.shape[-self.block_depth:]

    @property
    def spectrum(self) -> tensor.Tensor:
        if False:
            for i in range(10):
                print('nop')
        return self._spectrum

    def _vectorize_then_blockify(self, matrix):
        if False:
            for i in range(10):
                print('nop')
        'Shape batch matrix to batch vector, then blockify trailing dimensions.'
        vec = distribution_util.rotate_transpose(matrix, shift=1)
        if vec.shape.is_fully_defined() and self.block_shape.is_fully_defined():
            vec_leading_shape = vec.shape[:-1]
            final_shape = vec_leading_shape.concatenate(self.block_shape)
        else:
            vec_leading_shape = array_ops.shape(vec)[:-1]
            final_shape = array_ops.concat((vec_leading_shape, self.block_shape_tensor()), 0)
        return array_ops.reshape(vec, final_shape)

    def _unblockify(self, x):
        if False:
            return 10
        'Flatten the trailing block dimensions.'
        if x.shape.is_fully_defined():
            x_shape = x.shape.as_list()
            x_leading_shape = x_shape[:-self.block_depth]
            x_block_shape = x_shape[-self.block_depth:]
            flat_shape = x_leading_shape + [np.prod(x_block_shape)]
        else:
            x_shape = array_ops.shape(x)
            x_leading_shape = x_shape[:-self.block_depth]
            x_block_shape = x_shape[-self.block_depth:]
            flat_shape = array_ops.concat((x_leading_shape, [math_ops.reduce_prod(x_block_shape)]), 0)
        return array_ops.reshape(x, flat_shape)

    def _unblockify_then_matricize(self, vec):
        if False:
            i = 10
            return i + 15
        'Flatten the block dimensions then reshape to a batch matrix.'
        vec_flat = self._unblockify(vec)
        matrix = distribution_util.rotate_transpose(vec_flat, shift=-1)
        return matrix

    def _fft(self, x):
        if False:
            i = 10
            return i + 15
        'FFT along the last self.block_depth dimensions of x.\n\n    Args:\n      x: `Tensor` with floating or complex `dtype`.\n        Should be in the form returned by self._vectorize_then_blockify.\n\n    Returns:\n      `Tensor` with `dtype` `complex64`.\n    '
        x_complex = _to_complex(x)
        return _FFT_OP[self.block_depth](x_complex)

    def _ifft(self, x):
        if False:
            return 10
        'IFFT along the last self.block_depth dimensions of x.\n\n    Args:\n      x: `Tensor` with floating or complex dtype.  Should be in the form\n        returned by self._vectorize_then_blockify.\n\n    Returns:\n      `Tensor` with `dtype` `complex64`.\n    '
        x_complex = _to_complex(x)
        return _IFFT_OP[self.block_depth](x_complex)

    def convolution_kernel(self, name='convolution_kernel'):
        if False:
            i = 10
            return i + 15
        'Convolution kernel corresponding to `self.spectrum`.\n\n    The `D` dimensional DFT of this kernel is the frequency domain spectrum of\n    this operator.\n\n    Args:\n      name:  A name to give this `Op`.\n\n    Returns:\n      `Tensor` with `dtype` `self.dtype`.\n    '
        with self._name_scope(name):
            h = self._ifft(_to_complex(self.spectrum))
            return math_ops.cast(h, self.dtype)

    def _shape(self):
        if False:
            while True:
                i = 10
        s_shape = self._spectrum.shape
        batch_shape = s_shape[:-self.block_depth]
        trailing_dims = s_shape[-self.block_depth:]
        if trailing_dims.is_fully_defined():
            n = np.prod(trailing_dims.as_list())
        else:
            n = None
        n_x_n = tensor_shape.TensorShape([n, n])
        return batch_shape.concatenate(n_x_n)

    def _shape_tensor(self, spectrum=None):
        if False:
            print('Hello World!')
        spectrum = self.spectrum if spectrum is None else spectrum
        s_shape = array_ops.shape(spectrum)
        batch_shape = s_shape[:-self.block_depth]
        trailing_dims = s_shape[-self.block_depth:]
        n = math_ops.reduce_prod(trailing_dims)
        n_x_n = [n, n]
        return array_ops.concat((batch_shape, n_x_n), 0)

    def assert_hermitian_spectrum(self, name='assert_hermitian_spectrum'):
        if False:
            print('Hello World!')
        'Returns an `Op` that asserts this operator has Hermitian spectrum.\n\n    This operator corresponds to a real-valued matrix if and only if its\n    spectrum is Hermitian.\n\n    Args:\n      name:  A name to give this `Op`.\n\n    Returns:\n      An `Op` that asserts this operator has Hermitian spectrum.\n    '
        eps = np.finfo(self.dtype.real_dtype.as_numpy_dtype).eps
        with self._name_scope(name):
            max_err = eps * self.domain_dimension_tensor()
            imag_convolution_kernel = math_ops.imag(self.convolution_kernel())
            return check_ops.assert_less(math_ops.abs(imag_convolution_kernel), max_err, message='Spectrum was not Hermitian')

    def _assert_non_singular(self):
        if False:
            print('Hello World!')
        return linear_operator_util.assert_no_entries_with_modulus_zero(self.spectrum, message='Singular operator:  Spectrum contained zero values.')

    def _assert_positive_definite(self):
        if False:
            print('Hello World!')
        message = 'Not positive definite:  Real part of spectrum was not all positive.'
        return check_ops.assert_positive(math_ops.real(self.spectrum), message=message)

    def _assert_self_adjoint(self):
        if False:
            return 10
        return linear_operator_util.assert_zero_imag_part(self.spectrum, message='Not self-adjoint:  The spectrum contained non-zero imaginary part.')

    def _broadcast_batch_dims(self, x, spectrum):
        if False:
            return 10
        'Broadcast batch dims of batch matrix `x` and spectrum.'
        spectrum = tensor_conversion.convert_to_tensor_v2_with_dispatch(spectrum, name='spectrum')
        batch_shape = self._batch_shape_tensor(shape=self._shape_tensor(spectrum=spectrum))
        spec_mat = array_ops.reshape(spectrum, array_ops.concat((batch_shape, [-1, 1]), axis=0))
        (x, spec_mat) = linear_operator_util.broadcast_matrix_batch_dims((x, spec_mat))
        x_batch_shape = array_ops.shape(x)[:-2]
        spectrum_shape = array_ops.shape(spectrum)
        spectrum = array_ops.reshape(spec_mat, array_ops.concat((x_batch_shape, self._block_shape_tensor(spectrum_shape=spectrum_shape)), axis=0))
        return (x, spectrum)

    def _cond(self):
        if False:
            while True:
                i = 10
        abs_singular_values = math_ops.abs(self._unblockify(self.spectrum))
        return math_ops.reduce_max(abs_singular_values, axis=-1) / math_ops.reduce_min(abs_singular_values, axis=-1)

    def _eigvals(self):
        if False:
            for i in range(10):
                print('nop')
        return tensor_conversion.convert_to_tensor_v2_with_dispatch(self._unblockify(self.spectrum))

    def _matmul(self, x, adjoint=False, adjoint_arg=False):
        if False:
            return 10
        x = linalg.adjoint(x) if adjoint_arg else x
        spectrum = _to_complex(self.spectrum)
        if adjoint:
            spectrum = math_ops.conj(spectrum)
        x = math_ops.cast(x, spectrum.dtype)
        (x, spectrum) = self._broadcast_batch_dims(x, spectrum)
        x_vb = self._vectorize_then_blockify(x)
        fft_x_vb = self._fft(x_vb)
        block_vector_result = self._ifft(spectrum * fft_x_vb)
        y = self._unblockify_then_matricize(block_vector_result)
        return math_ops.cast(y, self.dtype)

    def _determinant(self):
        if False:
            return 10
        axis = [-(i + 1) for i in range(self.block_depth)]
        det = math_ops.reduce_prod(self.spectrum, axis=axis)
        return math_ops.cast(det, self.dtype)

    def _log_abs_determinant(self):
        if False:
            i = 10
            return i + 15
        axis = [-(i + 1) for i in range(self.block_depth)]
        lad = math_ops.reduce_sum(math_ops.log(math_ops.abs(self.spectrum)), axis=axis)
        return math_ops.cast(lad, self.dtype)

    def _solve(self, rhs, adjoint=False, adjoint_arg=False):
        if False:
            print('Hello World!')
        rhs = linalg.adjoint(rhs) if adjoint_arg else rhs
        spectrum = _to_complex(self.spectrum)
        if adjoint:
            spectrum = math_ops.conj(spectrum)
        (rhs, spectrum) = self._broadcast_batch_dims(rhs, spectrum)
        rhs_vb = self._vectorize_then_blockify(rhs)
        fft_rhs_vb = self._fft(rhs_vb)
        solution_vb = self._ifft(fft_rhs_vb / spectrum)
        x = self._unblockify_then_matricize(solution_vb)
        return math_ops.cast(x, self.dtype)

    def _diag_part(self):
        if False:
            print('Hello World!')
        if self.shape.is_fully_defined():
            diag_shape = self.shape[:-1]
            diag_size = self.domain_dimension.value
        else:
            diag_shape = self.shape_tensor()[:-1]
            diag_size = self.domain_dimension_tensor()
        ones_diag = array_ops.ones(diag_shape, dtype=self.dtype)
        diag_value = self.trace() / math_ops.cast(diag_size, self.dtype)
        return diag_value[..., array_ops.newaxis] * ones_diag

    def _trace(self):
        if False:
            i = 10
            return i + 15
        if self.spectrum.shape.is_fully_defined():
            spec_rank = self.spectrum.shape.ndims
            axis = np.arange(spec_rank - self.block_depth, spec_rank, dtype=np.int32)
        else:
            spec_rank = array_ops.rank(self.spectrum)
            axis = math_ops.range(spec_rank - self.block_depth, spec_rank)
        re_d_value = math_ops.reduce_sum(math_ops.real(self.spectrum), axis=axis)
        if not self.dtype.is_complex:
            return math_ops.cast(re_d_value, self.dtype)
        if self.is_self_adjoint:
            im_d_value = array_ops.zeros_like(re_d_value)
        else:
            im_d_value = math_ops.reduce_sum(math_ops.imag(self.spectrum), axis=axis)
        return math_ops.cast(math_ops.complex(re_d_value, im_d_value), self.dtype)

    @property
    def _composite_tensor_fields(self):
        if False:
            print('Hello World!')
        return ('spectrum', 'input_output_dtype')

    @property
    def _experimental_parameter_ndims_to_matrix_ndims(self):
        if False:
            print('Hello World!')
        return {'spectrum': self.block_depth}

@tf_export('linalg.LinearOperatorCirculant')
@linear_operator.make_composite_tensor
class LinearOperatorCirculant(_BaseLinearOperatorCirculant):
    """`LinearOperator` acting like a circulant matrix.

  This operator acts like a circulant matrix `A` with
  shape `[B1,...,Bb, N, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `N x N` matrix.  This matrix `A` is not materialized, but for
  purposes of broadcasting this shape will be relevant.

  #### Description in terms of circulant matrices

  Circulant means the entries of `A` are generated by a single vector, the
  convolution kernel `h`: `A_{mn} := h_{m-n mod N}`.  With `h = [w, x, y, z]`,

  ```
  A = |w z y x|
      |x w z y|
      |y x w z|
      |z y x w|
  ```

  This means that the result of matrix multiplication `v = Au` has `Lth` column
  given circular convolution between `h` with the `Lth` column of `u`.

  #### Description in terms of the frequency spectrum

  There is an equivalent description in terms of the [batch] spectrum `H` and
  Fourier transforms.  Here we consider `A.shape = [N, N]` and ignore batch
  dimensions.  Define the discrete Fourier transform (DFT) and its inverse by

  ```
  DFT[ h[n] ] = H[k] := sum_{n = 0}^{N - 1} h_n e^{-i 2pi k n / N}
  IDFT[ H[k] ] = h[n] = N^{-1} sum_{k = 0}^{N - 1} H_k e^{i 2pi k n / N}
  ```

  From these definitions, we see that

  ```
  H[0] = sum_{n = 0}^{N - 1} h_n
  H[1] = "the first positive frequency"
  H[N - 1] = "the first negative frequency"
  ```

  Loosely speaking, with `*` element-wise multiplication, matrix multiplication
  is equal to the action of a Fourier multiplier: `A u = IDFT[ H * DFT[u] ]`.
  Precisely speaking, given `[N, R]` matrix `u`, let `DFT[u]` be the `[N, R]`
  matrix with `rth` column equal to the DFT of the `rth` column of `u`.
  Define the `IDFT` similarly.
  Matrix multiplication may be expressed columnwise:

  ```(A u)_r = IDFT[ H * (DFT[u])_r ]```

  #### Operator properties deduced from the spectrum.

  Letting `U` be the `kth` Euclidean basis vector, and `U = IDFT[u]`.
  The above formulas show that`A U = H_k * U`.  We conclude that the elements
  of `H` are the eigenvalues of this operator.   Therefore

  * This operator is positive definite if and only if `Real{H} > 0`.

  A general property of Fourier transforms is the correspondence between
  Hermitian functions and real valued transforms.

  Suppose `H.shape = [B1,...,Bb, N]`.  We say that `H` is a Hermitian spectrum
  if, with `%` meaning modulus division,

  ```H[..., n % N] = ComplexConjugate[ H[..., (-n) % N] ]```

  * This operator corresponds to a real matrix if and only if `H` is Hermitian.
  * This operator is self-adjoint if and only if `H` is real.

  See e.g. "Discrete-Time Signal Processing", Oppenheim and Schafer.

  #### Example of a self-adjoint positive definite operator

  ```python
  # spectrum is real ==> operator is self-adjoint
  # spectrum is positive ==> operator is positive definite
  spectrum = [6., 4, 2]

  operator = LinearOperatorCirculant(spectrum)

  # IFFT[spectrum]
  operator.convolution_kernel()
  ==> [4 + 0j, 1 + 0.58j, 1 - 0.58j]

  operator.to_dense()
  ==> [[4 + 0.0j, 1 - 0.6j, 1 + 0.6j],
       [1 + 0.6j, 4 + 0.0j, 1 - 0.6j],
       [1 - 0.6j, 1 + 0.6j, 4 + 0.0j]]
  ```

  #### Example of defining in terms of a real convolution kernel

  ```python
  # convolution_kernel is real ==> spectrum is Hermitian.
  convolution_kernel = [1., 2., 1.]]
  spectrum = tf.signal.fft(tf.cast(convolution_kernel, tf.complex64))

  # spectrum is Hermitian ==> operator is real.
  # spectrum is shape [3] ==> operator is shape [3, 3]
  # We force the input/output type to be real, which allows this to operate
  # like a real matrix.
  operator = LinearOperatorCirculant(spectrum, input_output_dtype=tf.float32)

  operator.to_dense()
  ==> [[ 1, 1, 2],
       [ 2, 1, 1],
       [ 1, 2, 1]]
  ```

  #### Example of Hermitian spectrum

  ```python
  # spectrum is shape [3] ==> operator is shape [3, 3]
  # spectrum is Hermitian ==> operator is real.
  spectrum = [1, 1j, -1j]

  operator = LinearOperatorCirculant(spectrum)

  operator.to_dense()
  ==> [[ 0.33 + 0j,  0.91 + 0j, -0.24 + 0j],
       [-0.24 + 0j,  0.33 + 0j,  0.91 + 0j],
       [ 0.91 + 0j, -0.24 + 0j,  0.33 + 0j]
  ```

  #### Example of forcing real `dtype` when spectrum is Hermitian

  ```python
  # spectrum is shape [4] ==> operator is shape [4, 4]
  # spectrum is real ==> operator is self-adjoint
  # spectrum is Hermitian ==> operator is real
  # spectrum has positive real part ==> operator is positive-definite.
  spectrum = [6., 4, 2, 4]

  # Force the input dtype to be float32.
  # Cast the output to float32.  This is fine because the operator will be
  # real due to Hermitian spectrum.
  operator = LinearOperatorCirculant(spectrum, input_output_dtype=tf.float32)

  operator.shape
  ==> [4, 4]

  operator.to_dense()
  ==> [[4, 1, 0, 1],
       [1, 4, 1, 0],
       [0, 1, 4, 1],
       [1, 0, 1, 4]]

  # convolution_kernel = tf.signal.ifft(spectrum)
  operator.convolution_kernel()
  ==> [4, 1, 0, 1]
  ```

  #### Performance

  Suppose `operator` is a `LinearOperatorCirculant` of shape `[N, N]`,
  and `x.shape = [N, R]`.  Then

  * `operator.matmul(x)` is `O(R*N*Log[N])`
  * `operator.solve(x)` is `O(R*N*Log[N])`
  * `operator.determinant()` involves a size `N` `reduce_prod`.

  If instead `operator` and `x` have shape `[B1,...,Bb, N, N]` and
  `[B1,...,Bb, N, R]`, every operation increases in complexity by `B1*...*Bb`.

  #### Matrix property hints

  This `LinearOperator` is initialized with boolean flags of the form `is_X`,
  for `X = non_singular, self_adjoint, positive_definite, square`.
  These have the following meaning:

  * If `is_X == True`, callers should expect the operator to have the
    property `X`.  This is a promise that should be fulfilled, but is *not* a
    runtime assert.  For example, finite floating point precision may result
    in these promises being violated.
  * If `is_X == False`, callers should expect the operator to not have `X`.
  * If `is_X == None` (the default), callers should have no expectation either
    way.

  References:
    Toeplitz and Circulant Matrices - A Review:
      [Gray, 2006](https://www.nowpublishers.com/article/Details/CIT-006)
      ([pdf](https://ee.stanford.edu/~gray/toeplitz.pdf))
  """

    def __init__(self, spectrum: tensor.Tensor, input_output_dtype=dtypes.complex64, is_non_singular: bool=None, is_self_adjoint: bool=None, is_positive_definite: bool=None, is_square: bool=True, name='LinearOperatorCirculant'):
        if False:
            while True:
                i = 10
        'Initialize an `LinearOperatorCirculant`.\n\n    This `LinearOperator` is initialized to have shape `[B1,...,Bb, N, N]`\n    by providing `spectrum`, a `[B1,...,Bb, N]` `Tensor`.\n\n    If `input_output_dtype = DTYPE`:\n\n    * Arguments to methods such as `matmul` or `solve` must be `DTYPE`.\n    * Values returned by all methods, such as `matmul` or `determinant` will be\n      cast to `DTYPE`.\n\n    Note that if the spectrum is not Hermitian, then this operator corresponds\n    to a complex matrix with non-zero imaginary part.  In this case, setting\n    `input_output_dtype` to a real type will forcibly cast the output to be\n    real, resulting in incorrect results!\n\n    If on the other hand the spectrum is Hermitian, then this operator\n    corresponds to a real-valued matrix, and setting `input_output_dtype` to\n    a real type is fine.\n\n    Args:\n      spectrum:  Shape `[B1,...,Bb, N]` `Tensor`.  Allowed dtypes: `float16`,\n        `float32`, `float64`, `complex64`, `complex128`.  Type can be different\n        than `input_output_dtype`\n      input_output_dtype: `dtype` for input/output.\n      is_non_singular:  Expect that this operator is non-singular.\n      is_self_adjoint:  Expect that this operator is equal to its hermitian\n        transpose.  If `spectrum` is real, this will always be true.\n      is_positive_definite:  Expect that this operator is positive definite,\n        meaning the quadratic form `x^H A x` has positive real part for all\n        nonzero `x`.  Note that we do not require the operator to be\n        self-adjoint to be positive-definite.  See:\n        https://en.wikipedia.org/wiki/Positive-definite_matrix\\\n            #Extension_for_non_symmetric_matrices\n      is_square:  Expect that this operator acts like square [batch] matrices.\n      name:  A name to prepend to all ops created by this class.\n    '
        parameters = dict(spectrum=spectrum, input_output_dtype=input_output_dtype, is_non_singular=is_non_singular, is_self_adjoint=is_self_adjoint, is_positive_definite=is_positive_definite, is_square=is_square, name=name)
        super(LinearOperatorCirculant, self).__init__(spectrum, block_depth=1, input_output_dtype=input_output_dtype, is_non_singular=is_non_singular, is_self_adjoint=is_self_adjoint, is_positive_definite=is_positive_definite, is_square=is_square, parameters=parameters, name=name)

    def _linop_adjoint(self) -> 'LinearOperatorCirculant':
        if False:
            for i in range(10):
                print('nop')
        spectrum = self.spectrum
        if spectrum.dtype.is_complex:
            spectrum = math_ops.conj(spectrum)
        return LinearOperatorCirculant(spectrum=spectrum, is_non_singular=self.is_non_singular, is_self_adjoint=self.is_self_adjoint, is_positive_definite=self.is_positive_definite, is_square=True)

    def _linop_inverse(self) -> 'LinearOperatorCirculant':
        if False:
            for i in range(10):
                print('nop')
        return LinearOperatorCirculant(spectrum=1.0 / self.spectrum, is_non_singular=self.is_non_singular, is_self_adjoint=self.is_self_adjoint, is_positive_definite=self.is_positive_definite, is_square=True, input_output_dtype=self.dtype)

    def _linop_solve(self, left_operator: 'LinearOperatorCirculant', right_operator: linear_operator.LinearOperator) -> linear_operator.LinearOperator:
        if False:
            print('Hello World!')
        if not isinstance(right_operator, LinearOperatorCirculant):
            return super()._linop_solve(left_operator, right_operator)
        return LinearOperatorCirculant(spectrum=right_operator.spectrum / left_operator.spectrum, is_non_singular=property_hint_util.combined_non_singular_hint(left_operator, right_operator), is_self_adjoint=property_hint_util.combined_commuting_self_adjoint_hint(left_operator, right_operator), is_positive_definite=property_hint_util.combined_commuting_positive_definite_hint(left_operator, right_operator), is_square=True)

@tf_export('linalg.LinearOperatorCirculant2D')
@linear_operator.make_composite_tensor
class LinearOperatorCirculant2D(_BaseLinearOperatorCirculant):
    """`LinearOperator` acting like a block circulant matrix.

  This operator acts like a block circulant matrix `A` with
  shape `[B1,...,Bb, N, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `N x N` matrix.  This matrix `A` is not materialized, but for
  purposes of broadcasting this shape will be relevant.

  #### Description in terms of block circulant matrices

  If `A` is block circulant, with block sizes `N0, N1` (`N0 * N1 = N`):
  `A` has a block circulant structure, composed of `N0 x N0` blocks, with each
  block an `N1 x N1` circulant matrix.

  For example, with `W`, `X`, `Y`, `Z` each circulant,

  ```
  A = |W Z Y X|
      |X W Z Y|
      |Y X W Z|
      |Z Y X W|
  ```

  Note that `A` itself will not in general be circulant.

  #### Description in terms of the frequency spectrum

  There is an equivalent description in terms of the [batch] spectrum `H` and
  Fourier transforms.  Here we consider `A.shape = [N, N]` and ignore batch
  dimensions.

  If `H.shape = [N0, N1]`, (`N0 * N1 = N`):
  Loosely speaking, matrix multiplication is equal to the action of a
  Fourier multiplier:  `A u = IDFT2[ H DFT2[u] ]`.
  Precisely speaking, given `[N, R]` matrix `u`, let `DFT2[u]` be the
  `[N0, N1, R]` `Tensor` defined by re-shaping `u` to `[N0, N1, R]` and taking
  a two dimensional DFT across the first two dimensions.  Let `IDFT2` be the
  inverse of `DFT2`.  Matrix multiplication may be expressed columnwise:

  ```(A u)_r = IDFT2[ H * (DFT2[u])_r ]```

  #### Operator properties deduced from the spectrum.

  * This operator is positive definite if and only if `Real{H} > 0`.

  A general property of Fourier transforms is the correspondence between
  Hermitian functions and real valued transforms.

  Suppose `H.shape = [B1,...,Bb, N0, N1]`, we say that `H` is a Hermitian
  spectrum if, with `%` indicating modulus division,

  ```
  H[..., n0 % N0, n1 % N1] = ComplexConjugate[ H[..., (-n0) % N0, (-n1) % N1 ].
  ```

  * This operator corresponds to a real matrix if and only if `H` is Hermitian.
  * This operator is self-adjoint if and only if `H` is real.

  See e.g. "Discrete-Time Signal Processing", Oppenheim and Schafer.

  ### Example of a self-adjoint positive definite operator

  ```python
  # spectrum is real ==> operator is self-adjoint
  # spectrum is positive ==> operator is positive definite
  spectrum = [[1., 2., 3.],
              [4., 5., 6.],
              [7., 8., 9.]]

  operator = LinearOperatorCirculant2D(spectrum)

  # IFFT[spectrum]
  operator.convolution_kernel()
  ==> [[5.0+0.0j, -0.5-.3j, -0.5+.3j],
       [-1.5-.9j,        0,        0],
       [-1.5+.9j,        0,        0]]

  operator.to_dense()
  ==> Complex self adjoint 9 x 9 matrix.
  ```

  #### Example of defining in terms of a real convolution kernel,

  ```python
  # convolution_kernel is real ==> spectrum is Hermitian.
  convolution_kernel = [[1., 2., 1.], [5., -1., 1.]]
  spectrum = tf.signal.fft2d(tf.cast(convolution_kernel, tf.complex64))

  # spectrum is shape [2, 3] ==> operator is shape [6, 6]
  # spectrum is Hermitian ==> operator is real.
  operator = LinearOperatorCirculant2D(spectrum, input_output_dtype=tf.float32)
  ```

  #### Performance

  Suppose `operator` is a `LinearOperatorCirculant` of shape `[N, N]`,
  and `x.shape = [N, R]`.  Then

  * `operator.matmul(x)` is `O(R*N*Log[N])`
  * `operator.solve(x)` is `O(R*N*Log[N])`
  * `operator.determinant()` involves a size `N` `reduce_prod`.

  If instead `operator` and `x` have shape `[B1,...,Bb, N, N]` and
  `[B1,...,Bb, N, R]`, every operation increases in complexity by `B1*...*Bb`.

  #### Matrix property hints

  This `LinearOperator` is initialized with boolean flags of the form `is_X`,
  for `X = non_singular, self_adjoint, positive_definite, square`.
  These have the following meaning
  * If `is_X == True`, callers should expect the operator to have the
    property `X`.  This is a promise that should be fulfilled, but is *not* a
    runtime assert.  For example, finite floating point precision may result
    in these promises being violated.
  * If `is_X == False`, callers should expect the operator to not have `X`.
  * If `is_X == None` (the default), callers should have no expectation either
    way.
  """

    def __init__(self, spectrum: tensor.Tensor, input_output_dtype=dtypes.complex64, is_non_singular: bool=None, is_self_adjoint: bool=None, is_positive_definite: bool=None, is_square: bool=True, name='LinearOperatorCirculant2D'):
        if False:
            i = 10
            return i + 15
        'Initialize an `LinearOperatorCirculant2D`.\n\n    This `LinearOperator` is initialized to have shape `[B1,...,Bb, N, N]`\n    by providing `spectrum`, a `[B1,...,Bb, N0, N1]` `Tensor` with `N0*N1 = N`.\n\n    If `input_output_dtype = DTYPE`:\n\n    * Arguments to methods such as `matmul` or `solve` must be `DTYPE`.\n    * Values returned by all methods, such as `matmul` or `determinant` will be\n      cast to `DTYPE`.\n\n    Note that if the spectrum is not Hermitian, then this operator corresponds\n    to a complex matrix with non-zero imaginary part.  In this case, setting\n    `input_output_dtype` to a real type will forcibly cast the output to be\n    real, resulting in incorrect results!\n\n    If on the other hand the spectrum is Hermitian, then this operator\n    corresponds to a real-valued matrix, and setting `input_output_dtype` to\n    a real type is fine.\n\n    Args:\n      spectrum:  Shape `[B1,...,Bb, N0, N1]` `Tensor`.  Allowed dtypes:\n        `float16`, `float32`, `float64`, `complex64`, `complex128`.\n        Type can be different than `input_output_dtype`\n      input_output_dtype: `dtype` for input/output.\n      is_non_singular:  Expect that this operator is non-singular.\n      is_self_adjoint:  Expect that this operator is equal to its hermitian\n        transpose.  If `spectrum` is real, this will always be true.\n      is_positive_definite:  Expect that this operator is positive definite,\n        meaning the quadratic form `x^H A x` has positive real part for all\n        nonzero `x`.  Note that we do not require the operator to be\n        self-adjoint to be positive-definite.  See:\n        https://en.wikipedia.org/wiki/Positive-definite_matrix\\\n            #Extension_for_non_symmetric_matrices\n      is_square:  Expect that this operator acts like square [batch] matrices.\n      name:  A name to prepend to all ops created by this class.\n    '
        parameters = dict(spectrum=spectrum, input_output_dtype=input_output_dtype, is_non_singular=is_non_singular, is_self_adjoint=is_self_adjoint, is_positive_definite=is_positive_definite, is_square=is_square, name=name)
        super(LinearOperatorCirculant2D, self).__init__(spectrum, block_depth=2, input_output_dtype=input_output_dtype, is_non_singular=is_non_singular, is_self_adjoint=is_self_adjoint, is_positive_definite=is_positive_definite, is_square=is_square, parameters=parameters, name=name)

    def _linop_adjoint(self) -> 'LinearOperatorCirculant2D':
        if False:
            return 10
        spectrum = self.spectrum
        if spectrum.dtype.is_complex:
            spectrum = math_ops.conj(spectrum)
        return LinearOperatorCirculant2D(spectrum=spectrum, is_non_singular=self.is_non_singular, is_self_adjoint=self.is_self_adjoint, is_positive_definite=self.is_positive_definite, is_square=True)

    def _linop_inverse(self) -> 'LinearOperatorCirculant2D':
        if False:
            for i in range(10):
                print('nop')
        return LinearOperatorCirculant2D(spectrum=1.0 / self.spectrum, is_non_singular=self.is_non_singular, is_self_adjoint=self.is_self_adjoint, is_positive_definite=self.is_positive_definite, is_square=True, input_output_dtype=self.dtype)

    def _linop_solve(self, left_operator: 'LinearOperatorCirculant2D', right_operator: linear_operator.LinearOperator) -> linear_operator.LinearOperator:
        if False:
            i = 10
            return i + 15
        if not isinstance(right_operator, LinearOperatorCirculant2D):
            return super()._linop_solve(left_operator, right_operator)
        return LinearOperatorCirculant2D(spectrum=right_operator.spectrum / left_operator.spectrum, is_non_singular=property_hint_util.combined_non_singular_hint(left_operator, right_operator), is_self_adjoint=property_hint_util.combined_commuting_self_adjoint_hint(left_operator, right_operator), is_positive_definite=property_hint_util.combined_commuting_positive_definite_hint(left_operator, right_operator), is_square=True)

@tf_export('linalg.LinearOperatorCirculant3D')
@linear_operator.make_composite_tensor
class LinearOperatorCirculant3D(_BaseLinearOperatorCirculant):
    """`LinearOperator` acting like a nested block circulant matrix.

  This operator acts like a block circulant matrix `A` with
  shape `[B1,...,Bb, N, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `N x N` matrix.  This matrix `A` is not materialized, but for
  purposes of broadcasting this shape will be relevant.

  #### Description in terms of block circulant matrices

  If `A` is nested block circulant, with block sizes `N0, N1, N2`
  (`N0 * N1 * N2 = N`):
  `A` has a block structure, composed of `N0 x N0` blocks, with each
  block an `N1 x N1` block circulant matrix.

  For example, with `W`, `X`, `Y`, `Z` each block circulant,

  ```
  A = |W Z Y X|
      |X W Z Y|
      |Y X W Z|
      |Z Y X W|
  ```

  Note that `A` itself will not in general be circulant.

  #### Description in terms of the frequency spectrum

  There is an equivalent description in terms of the [batch] spectrum `H` and
  Fourier transforms.  Here we consider `A.shape = [N, N]` and ignore batch
  dimensions.

  If `H.shape = [N0, N1, N2]`, (`N0 * N1 * N2 = N`):
  Loosely speaking, matrix multiplication is equal to the action of a
  Fourier multiplier:  `A u = IDFT3[ H DFT3[u] ]`.
  Precisely speaking, given `[N, R]` matrix `u`, let `DFT3[u]` be the
  `[N0, N1, N2, R]` `Tensor` defined by re-shaping `u` to `[N0, N1, N2, R]` and
  taking a three dimensional DFT across the first three dimensions.  Let `IDFT3`
  be the inverse of `DFT3`.  Matrix multiplication may be expressed columnwise:

  ```(A u)_r = IDFT3[ H * (DFT3[u])_r ]```

  #### Operator properties deduced from the spectrum.

  * This operator is positive definite if and only if `Real{H} > 0`.

  A general property of Fourier transforms is the correspondence between
  Hermitian functions and real valued transforms.

  Suppose `H.shape = [B1,...,Bb, N0, N1, N2]`, we say that `H` is a Hermitian
  spectrum if, with `%` meaning modulus division,

  ```
  H[..., n0 % N0, n1 % N1, n2 % N2]
    = ComplexConjugate[ H[..., (-n0) % N0, (-n1) % N1, (-n2) % N2] ].
  ```

  * This operator corresponds to a real matrix if and only if `H` is Hermitian.
  * This operator is self-adjoint if and only if `H` is real.

  See e.g. "Discrete-Time Signal Processing", Oppenheim and Schafer.

  ### Examples

  See `LinearOperatorCirculant` and `LinearOperatorCirculant2D` for examples.

  #### Performance

  Suppose `operator` is a `LinearOperatorCirculant` of shape `[N, N]`,
  and `x.shape = [N, R]`.  Then

  * `operator.matmul(x)` is `O(R*N*Log[N])`
  * `operator.solve(x)` is `O(R*N*Log[N])`
  * `operator.determinant()` involves a size `N` `reduce_prod`.

  If instead `operator` and `x` have shape `[B1,...,Bb, N, N]` and
  `[B1,...,Bb, N, R]`, every operation increases in complexity by `B1*...*Bb`.

  #### Matrix property hints

  This `LinearOperator` is initialized with boolean flags of the form `is_X`,
  for `X = non_singular, self_adjoint, positive_definite, square`.
  These have the following meaning
  * If `is_X == True`, callers should expect the operator to have the
    property `X`.  This is a promise that should be fulfilled, but is *not* a
    runtime assert.  For example, finite floating point precision may result
    in these promises being violated.
  * If `is_X == False`, callers should expect the operator to not have `X`.
  * If `is_X == None` (the default), callers should have no expectation either
    way.
  """

    def __init__(self, spectrum: tensor.Tensor, input_output_dtype=dtypes.complex64, is_non_singular: bool=None, is_self_adjoint: bool=None, is_positive_definite: bool=None, is_square: bool=True, name='LinearOperatorCirculant3D'):
        if False:
            i = 10
            return i + 15
        'Initialize an `LinearOperatorCirculant`.\n\n    This `LinearOperator` is initialized to have shape `[B1,...,Bb, N, N]`\n    by providing `spectrum`, a `[B1,...,Bb, N0, N1, N2]` `Tensor`\n    with `N0*N1*N2 = N`.\n\n    If `input_output_dtype = DTYPE`:\n\n    * Arguments to methods such as `matmul` or `solve` must be `DTYPE`.\n    * Values returned by all methods, such as `matmul` or `determinant` will be\n      cast to `DTYPE`.\n\n    Note that if the spectrum is not Hermitian, then this operator corresponds\n    to a complex matrix with non-zero imaginary part.  In this case, setting\n    `input_output_dtype` to a real type will forcibly cast the output to be\n    real, resulting in incorrect results!\n\n    If on the other hand the spectrum is Hermitian, then this operator\n    corresponds to a real-valued matrix, and setting `input_output_dtype` to\n    a real type is fine.\n\n    Args:\n      spectrum:  Shape `[B1,...,Bb, N0, N1, N2]` `Tensor`.  Allowed dtypes:\n        `float16`, `float32`, `float64`, `complex64`, `complex128`.\n        Type can be different than `input_output_dtype`\n      input_output_dtype: `dtype` for input/output.\n      is_non_singular:  Expect that this operator is non-singular.\n      is_self_adjoint:  Expect that this operator is equal to its hermitian\n        transpose.  If `spectrum` is real, this will always be true.\n      is_positive_definite:  Expect that this operator is positive definite,\n        meaning the real part of all eigenvalues is positive.  We do not require\n        the operator to be self-adjoint to be positive-definite.  See:\n        https://en.wikipedia.org/wiki/Positive-definite_matrix\n            #Extension_for_non_symmetric_matrices\n      is_square:  Expect that this operator acts like square [batch] matrices.\n      name:  A name to prepend to all ops created by this class.\n    '
        parameters = dict(spectrum=spectrum, input_output_dtype=input_output_dtype, is_non_singular=is_non_singular, is_self_adjoint=is_self_adjoint, is_positive_definite=is_positive_definite, is_square=is_square, name=name)
        super(LinearOperatorCirculant3D, self).__init__(spectrum, block_depth=3, input_output_dtype=input_output_dtype, is_non_singular=is_non_singular, is_self_adjoint=is_self_adjoint, is_positive_definite=is_positive_definite, is_square=is_square, parameters=parameters, name=name)

    def _linop_adjoint(self) -> 'LinearOperatorCirculant3D':
        if False:
            for i in range(10):
                print('nop')
        spectrum = self.spectrum
        if spectrum.dtype.is_complex:
            spectrum = math_ops.conj(spectrum)
        return LinearOperatorCirculant3D(spectrum=spectrum, is_non_singular=self.is_non_singular, is_self_adjoint=self.is_self_adjoint, is_positive_definite=self.is_positive_definite, is_square=True)

    def _linop_inverse(self) -> 'LinearOperatorCirculant3D':
        if False:
            print('Hello World!')
        return LinearOperatorCirculant3D(spectrum=1.0 / self.spectrum, is_non_singular=self.is_non_singular, is_self_adjoint=self.is_self_adjoint, is_positive_definite=self.is_positive_definite, is_square=True, input_output_dtype=self.dtype)

    def _linop_solve(self, left_operator: 'LinearOperatorCirculant3D', right_operator: linear_operator.LinearOperator) -> linear_operator.LinearOperator:
        if False:
            return 10
        if not isinstance(right_operator, LinearOperatorCirculant3D):
            return super()._linop_solve(left_operator, right_operator)
        return LinearOperatorCirculant3D(spectrum=right_operator.spectrum / left_operator.spectrum, is_non_singular=property_hint_util.combined_non_singular_hint(left_operator, right_operator), is_self_adjoint=property_hint_util.combined_commuting_self_adjoint_hint(left_operator, right_operator), is_positive_definite=property_hint_util.combined_commuting_positive_definite_hint(left_operator, right_operator), is_square=True)

def _to_complex(x):
    if False:
        return 10
    if x.dtype.is_complex:
        return x
    dtype = dtypes.complex64
    if x.dtype == dtypes.float64:
        dtype = dtypes.complex128
    return math_ops.cast(x, dtype)