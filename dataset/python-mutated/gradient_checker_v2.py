"""Gradient checker for functions.

The gradient checker verifies numerically that an function properly
computes the gradients
"""
import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export

def _product(t):
    if False:
        while True:
            i = 10
    if isinstance(t, int):
        return t
    else:
        y = 1
        for x in t:
            y *= x
        return y

def _eval_indexed_slices(a):
    if False:
        i = 10
        return i + 15
    "Converts IndexedSlices to IndexedSlicesValue with numpy indices/values.\n\n  When eager execution is enabled, converts IndexedSlices\n  to IndexedSlicesValue with numpy indices/values.\n\n  Args:\n    a: any value.\n\n  Returns:\n    If a is IndexedSlices and eager execution is enabled, calls numpy() on a's\n    fields. Otherwise returns a unchanged.\n  "
    if isinstance(a, indexed_slices.IndexedSlices) and context.executing_eagerly():
        return indexed_slices.IndexedSlicesValue(indices=[x.numpy() for x in a.indices], values=[x.numpy() for x in a.values], dense_shape=a.dense_shape)
    return a

def _to_numpy(a):
    if False:
        print('Hello World!')
    'Converts Tensors, EagerTensors, and IndexedSlicesValue to numpy arrays.\n\n  Args:\n    a: any value.\n\n  Returns:\n    If a is EagerTensor or Tensor, returns the evaluation of a by calling\n    numpy() or run(). If a is IndexedSlicesValue, constructs the corresponding\n    dense numpy array. Otherwise returns a unchanged.\n  '
    if isinstance(a, ops.EagerTensor):
        return a.numpy()
    if isinstance(a, tensor.Tensor):
        sess = ops.get_default_session()
        return sess.run(a)
    if isinstance(a, indexed_slices.IndexedSlicesValue):
        arr = np.zeros(a.dense_shape)
        assert len(a.values) == len(a.indices), 'IndexedSlicesValue has %s value slices but %s indices\n%s' % (a.values, a.indices, a)
        for (values_slice, index) in zip(a.values, a.indices):
            assert 0 <= index < len(arr), 'IndexedSlicesValue has invalid index %s\n%s' % (index, a)
            arr[index] += values_slice
        return arr
    return a

def _prepare(f, xs_dtypes, xs_shapes):
    if False:
        for i in range(10):
            print('nop')
    "Return a function that executes 'f'.\n\n    In TF 2.x, this is the same as `f`.\n    In TF 1.x, returns a Python function that executes the graph defined by `f`\n    in a Session.\n\n  Args:\n    f: the function.\n    xs_dtypes: dtypes of f's arguments.\n    xs_shapes: shapes of f's arguments.\n\n  Returns:\n  "
    if context.executing_eagerly():

        def decorated_eager(*xs_data):
            if False:
                for i in range(10):
                    print('nop')
            return f(*map(ops.convert_to_tensor, xs_data))
        return decorated_eager
    xs = [array_ops.placeholder(x_dtype, shape=x_shape) for (x_dtype, x_shape) in zip(xs_dtypes, xs_shapes)]
    y = f(*xs)
    sess = ops.get_default_session()

    def decorated_graph(*xs_data):
        if False:
            return 10
        xs_data = [_to_numpy(a) for a in xs_data]
        return sess.run(y, feed_dict=dict(zip(xs, xs_data)))
    return decorated_graph

def _compute_theoretical_jacobian(f, y_shape, y_dtype, xs, param):
    if False:
        while True:
            i = 10
    'Computes the theoretical Jacobian for f regarding xs[param].\n\n  One can think of the relation among f, xs and y as y = f(xs).\n\n  Args:\n    f: the function.\n    y_shape: the shape of the result.\n    y_dtype: the dtype of the result.\n    xs: a list of tensors.\n    param: the index of the target parameter.\n\n  Returns:\n    A 2-d numpy array representing the Jacobian. It has "y_size" rows\n    and "x_size" columns where "x_size" is the number of elements in xs[param]\n    and "y_size" is the number of elements in the result.\n\n  Raises:\n    ValueError: If result is empty but the gradient is nonzero.\n  '
    x = xs[param]
    x_shape = tuple(x.shape) + (2,) if x.dtype.is_complex else x.shape
    y_factor = 2 if y_dtype.is_complex else 1
    x_size = _product(x_shape)
    x_val_size = _product(x_shape[1:])
    y_size = _product(y_shape) * y_factor
    jacobian = np.zeros((y_size, x_size), dtype=x.dtype.real_dtype.as_numpy_dtype)
    dy_data = np.zeros(y_shape, dtype=y_dtype.as_numpy_dtype)
    dy_data_flat = dy_data.ravel().view(y_dtype.real_dtype.as_numpy_dtype)
    grad_fn_unprep = backprop.gradients_function(f, [param])
    grad_fn = _prepare(lambda dy, *xs: grad_fn_unprep(*xs, dy=dy), [y_dtype] + [z.dtype for z in xs], [None] + [z.shape for z in xs])
    for row in range(y_size):
        dy_data_flat[row] = 1
        grad = _to_numpy(grad_fn(dy_data, *xs)[0])
        grad = _eval_indexed_slices(grad)
        if isinstance(grad, indexed_slices.IndexedSlicesValue):
            for (i, v) in zip(grad.indices, grad.values):
                c_begin = i * x_val_size
                c_end = c_begin + x_val_size
                jacobian[row, c_begin:c_end] += v.flat
        elif grad is not None:
            jacobian[row, :] = grad.ravel().view(jacobian.dtype)
        dy_data_flat[row] = 0
    if y_size == 0:
        grad = _to_numpy(grad_fn(dy_data, *xs)[0])
        if grad.shape != x.shape:
            raise ValueError('Empty gradient has wrong shape: expected %s, got %s' % (x.shape, grad.shape))
        if np.any(grad):
            raise ValueError('Empty tensor with nonzero gradients')
    logging.vlog(1, 'Theoretical Jacobian =\n%s', jacobian)
    return jacobian

def _compute_numeric_jacobian(f, y_size, y_dtype, xs, param, delta):
    if False:
        print('Hello World!')
    'Computes the numeric Jacobian for f regarding xs[param].\n\n  One can think of the relation among f, xs and y as y = f(xs).\n\n  Args:\n    f: the function.\n    y_size: the number of elements of the result.\n    y_dtype: the dtype of the result.\n    xs: a list of tensors.\n    param: the index of the target parameter.\n    delta: the amount of perturbation we give to the input.\n\n  Returns:\n    A 2-d numpy array representing the Jacobian. It has "y_size" rows\n    and "x_size" columns where "x_size" is the number of elements in xs[param]\n    and "y_size" is the number of elements in the result.\n  '
    x_shape = xs[param].shape
    x_dtype = xs[param].dtype
    x_size = _product(x_shape) * (2 if x_dtype.is_complex else 1)
    y_size = y_size * (2 if y_dtype.is_complex else 1)
    x_dtype = x_dtype.real_dtype.as_numpy_dtype
    y_dtype = y_dtype.real_dtype.as_numpy_dtype
    xs_dtypes = [x.dtype for x in xs]
    xs_shapes = [x.shape for x in xs]
    xs = [np.asarray(_to_numpy(x)) for x in xs]
    x = xs[param]
    scale = np.asarray(2 * delta, dtype=y_dtype)[()]
    jacobian = np.zeros((y_size, x_size), dtype=x_dtype)
    f = _prepare(f, xs_dtypes, xs_shapes)
    for col in range(x_size):
        original = x.ravel().view(x_dtype)[col]
        x.ravel().view(x_dtype)[col] += delta
        y_pos = _to_numpy(f(*xs))
        x.ravel().view(x_dtype)[col] = original
        x.ravel().view(x_dtype)[col] -= delta
        y_neg = _to_numpy(f(*xs))
        x.ravel().view(x_dtype)[col] = original
        diff = (y_pos - y_neg) / scale
        jacobian[:, col] = diff.ravel().view(y_dtype)
    logging.vlog(1, 'Numeric Jacobian =\n%s', jacobian)
    return jacobian

def _compute_gradient(f, y_shape, y_dtype, xs, param, delta):
    if False:
        print('Hello World!')
    'Computes the theoretical and numerical jacobian.'
    x = xs[param]
    t = x.dtype
    allowed_types = [dtypes.float16, dtypes.bfloat16, dtypes.float32, dtypes.float64, dtypes.complex64, dtypes.complex128]
    assert t.base_dtype in allowed_types, 'Cannot compute gradient for unsupported type %s of argument %s' % (t.name, param)
    t2 = y_dtype
    assert t2.base_dtype in allowed_types, 'Cannot compute gradient for unsupported type %s of y' % t2.name
    y_size = _product(y_shape)
    jacob_t = _compute_theoretical_jacobian(f, y_shape, y_dtype, xs, param)
    jacob_n = _compute_numeric_jacobian(f, y_size, y_dtype, xs, param, delta)
    return (jacob_t, jacob_n)

def _compute_gradient_list(f, xs, delta):
    if False:
        print('Hello World!')
    'Compute gradients for a list of x values.'
    xs = [ops.convert_to_tensor(x) for x in xs]
    xs_dtypes = [x.dtype for x in xs]
    xs_shapes = [x.shape for x in xs]
    f_temp = _prepare(f, xs_dtypes, xs_shapes)
    y = f_temp(*xs)
    return tuple(zip(*[_compute_gradient(f, y.shape, dtypes.as_dtype(y.dtype), xs, i, delta) for i in range(len(xs))]))

@tf_export('test.compute_gradient', v1=[])
def compute_gradient(f, x, delta=None):
    if False:
        return 10
    'Computes the theoretical and numeric Jacobian of `f`.\n\n  With y = f(x), computes the theoretical and numeric Jacobian dy/dx.\n\n  Args:\n    f: the function.\n    x: the arguments for the function as a list or tuple of values convertible\n      to a Tensor.\n    delta: (optional) perturbation used to compute numeric Jacobian.\n\n  Returns:\n    A pair of lists, where the first is a list of 2-d numpy arrays representing\n    the theoretical Jacobians for each argument, and the second list is the\n    numerical ones. Each 2-d array has "y_size" rows\n    and "x_size" columns where "x_size" is the number of elements in the\n    corresponding argument and "y_size" is the number of elements in f(x).\n\n  Raises:\n    ValueError: If result is empty but the gradient is nonzero.\n    ValueError: If x is not list, but any other type.\n\n  Example:\n\n  >>> @tf.function\n  ... def test_func(x):\n  ...   return x*x\n  ...\n  >>>\n  >>> class MyTest(tf.test.TestCase):\n  ...\n  ...   def test_gradient_of_test_func(self):\n  ...     theoretical, numerical = tf.test.compute_gradient(test_func, [1.0])\n  ...     # ((array([[2.]], dtype=float32),),\n  ...     #  (array([[2.000004]], dtype=float32),))\n  ...     self.assertAllClose(theoretical, numerical)\n\n  '
    if not isinstance(x, (list, tuple)):
        raise ValueError('`x` must be a list or tuple of values convertible to a Tensor (arguments to `f`), not a %s' % type(x))
    if delta is None:
        delta = 1.0 / 1024
    return _compute_gradient_list(f, x, delta)

def max_error(grad1, grad2):
    if False:
        while True:
            i = 10
    'Computes maximum elementwise gap.\n\n  Computes the maximum elementwise gap between two lists of tensors of the same\n  shape.\n\n  Args:\n    grad1: a lists of tensors.\n    grad2: a lists of tensors with the same shape as grad1.\n\n  Returns:\n    The maximum elementwise gap between the two.\n  '
    error = 0
    for (j_t, j_n) in zip(grad1, grad2):
        if j_t.size or j_n.size:
            error = np.maximum(error, np.fabs(j_t - j_n).max())
    return error