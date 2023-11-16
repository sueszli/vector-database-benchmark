"""Script Language Operators."""
import functools
import threading
import traceback
import weakref
import numpy as np
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.eager import backprop
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.lib.core import _pywrap_py_func
from tensorflow.python.ops import autograph_ops
from tensorflow.python.ops import gen_script_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
tape_cache = {}

def _maybe_copy_to_context_device(tensor, device_name):
    if False:
        i = 10
        return i + 15
    "Copy an EagerTensor to the current device if it's not on `device_name`."
    in_device = tensor.backing_device
    if device_name == in_device:
        return tensor
    else:
        return tensor._copy()

class EagerFunc:
    """A wrapper for a function owned by an EagerPyFunc."""

    def __init__(self, func, Tout, is_grad_func):
        if False:
            for i in range(10):
                print('nop')
        'Constructs an EagerFunc.\n\n    Args:\n      func: The function to wrap.\n      Tout: A list of datatypes for the output; an empty list if the output is\n        None.\n      is_grad_func: Whether this EagerFunc is the gradient of another\n        EagerPyFunc.\n    '
        self._func = func
        self._out_dtypes = Tout
        self._is_grad_func = is_grad_func
        self._support_graph_mode_gradient = False

    def set_support_graph_mode_gradient(self):
        if False:
            while True:
                i = 10
        'Indicates the object shall support gradient ops.\n\n    This function is internally used by _EagerPyFuncGrad to support\n    graph mode gradient of EagerFunc via tf.gradient().\n    '
        self._support_graph_mode_gradient = True

    def _convert(self, value, dtype):
        if False:
            i = 10
            return i + 15
        'Converts `value` to a tensor of type `dtype`, with error checking.\n\n    Args:\n      value: The tensor to convert.\n      dtype: The desired dtype.\n\n    Returns:\n      A tensor of type `dtype`, or a zeros tensor if value is None and\n      this function is in fact a gradient function.\n\n    Raises:\n      RuntimeError: if `value` is a variable.\n    '
        if isinstance(value, resource_variable_ops.ResourceVariable):
            raise RuntimeError(f'Attempting to return a variable from an eagerly executed py_func. Only numeric data structures like Tensors or NumPy arrays should be returned; to return the value of a variable, make sure to obtain the Tensor backing it by calling `.read_value()` on the variable in question: {value}')
        if value is None and self._is_grad_func:
            return constant_op.constant(0.0, dtype=dtype)
        return ops.convert_to_tensor(value, dtype=dtype)

    def __call__(self, device, token, args):
        if False:
            i = 10
            return i + 15
        'Calls `self._func` in eager mode, recording the tape if needed.'
        use_tape_cache = self._support_graph_mode_gradient or record.could_possibly_record()
        if use_tape_cache:
            with backprop.GradientTape() as tape:
                for tensor in args:
                    for t in nest.flatten(tensor):
                        if backprop_util.IsTrainable(t):
                            tape.watch(t)
                outputs = self._call(device, args)
            tape_cache[compat.as_bytes(token)] = (tape, args, outputs)
        else:
            outputs = self._call(device, args)
        return outputs

    def _call(self, device, args):
        if False:
            print('Hello World!')
        'Passes `args` to `self._func`, which is executed eagerly.'
        with context.eager_mode():
            ret = self._func(*args)
            device_name = device
            if device_name is None:
                device_name = '/job:localhost/replica:0/task:0/device:CPU:0'
            with ops.device(device):
                if isinstance(ret, (tuple, list)):
                    outputs = [_maybe_copy_to_context_device(self._convert(x, dtype=dtype), device_name) for (x, dtype) in zip(ret, self._out_dtypes)]
                elif ret is None:
                    outputs = None
                else:
                    outputs = _maybe_copy_to_context_device(self._convert(ret, dtype=self._out_dtypes[0]), device_name)
        return outputs

class FuncRegistry:
    """A helper class to keep track of registered py functions.

  FuncRegistry keeps a map from unique tokens (string) to python
  functions, which takes numpy arrays and outputs numpy arrays.
  """

    def __init__(self):
        if False:
            return 10
        self._lock = threading.Lock()
        self._unique_id = 0
        self._funcs = weakref.WeakValueDictionary()

    @property
    def _ctx(self):
        if False:
            while True:
                i = 10
        context.ensure_initialized()
        return context.context()._handle

    def insert(self, func):
        if False:
            i = 10
            return i + 15
        'Registers `func` and returns a unique token for this entry.'
        token = self._next_unique_token()
        self._funcs[token] = func
        return token

    def remove(self, token):
        if False:
            while True:
                i = 10
        'Removes the registered function corresponding to `token`.'
        self._funcs.pop(token, None)

    def get(self, token, default=None):
        if False:
            print('Hello World!')
        'Gets the registered function corresponding to `token`.'
        return self._funcs.get(token, default)

    @staticmethod
    def _convert(value, dtype=None):
        if False:
            print('Hello World!')
        'Converts an arg to numpy, avoiding dangerous string and unicode dtypes.\n\n    Numpy pads with zeros when using string and unicode dtypes if different\n    components of a tensor have different lengths.  This is bad: ignoring the\n    padding is wrong for text data, and removing the padding is wrong for binary\n    data.  To avoid this bug, we redo the conversion using an object dtype.\n    Additionally, we convert unicode strings to (byte-)strings for\n    compatibility.\n\n    Args:\n      value: Value to convert to a numpy array.\n      dtype: (Optional.) Desired NumPy type for the returned value.\n\n    Returns:\n      A numpy array.\n    '
        result = np.asarray(value, dtype=dtype, order='C')
        if result.dtype.char == 'S' and result is not value:
            return np.asarray(value, order='C', dtype=object)
        elif result.dtype.char == 'U' and result is not value:
            value = np.vectorize(lambda x: x.encode('utf8'))(value)
            return np.asarray(value, order='C', dtype=object)
        elif result.dtype.char == 'U':
            return result.astype(np.bytes_)
        else:
            return result

    def __call__(self, token, device, args):
        if False:
            return 10
        "Calls the registered function for `token` with args.\n\n    Args:\n      token: A key into this `FuncRegistry` identifying which function to call.\n      device: Name of the device on which outputs of `token`'s corresponding\n        operation should be placed. Used iff the function registered for `token`\n        is an EagerPyFunc.\n      args: The arguments to pass to the function registered for `token`.\n\n    Returns:\n      The output of the function registered for `token`.\n\n    Raises:\n      ValueError: if no function is registered for `token`.\n    "
        func = self.get(token, None)
        if func is None:
            raise ValueError(f'Could not find callback with key={token} in the registry.')
        if isinstance(func, EagerFunc):
            return func(device, token, args)
        else:
            ret = func(*args)
            if isinstance(ret, bytes):
                ret = [ret]
            if isinstance(ret, (tuple, list)):
                return [self._convert(x) for x in ret]
            else:
                return self._convert(ret)

    def size(self):
        if False:
            return 10
        'Returns how many functions are currently registered.'
        return len(self._funcs)

    def _next_unique_token(self):
        if False:
            while True:
                i = 10
        'Returns a unique token.'
        with self._lock:
            uid = self._unique_id
            self._unique_id += 1
        return 'pyfunc_%d' % uid
_py_funcs = FuncRegistry()
_pywrap_py_func.initialize_py_trampoline(_py_funcs)

def _internal_py_func(func, inp, Tout, stateful=None, use_eager_py_func=False, is_grad_func=False, name=None):
    if False:
        i = 10
        return i + 15
    'See documentation for py_func and eager_py_func.'
    if not callable(func):
        raise ValueError(f'Expected func to be callable. Received func={func} of type {type(func)}.')
    original_func = func
    func = autograph.do_not_convert(func)
    inp = variable_utils.convert_variables_to_tensors(list(inp))
    is_list_or_tuple = isinstance(Tout, (list, tuple))
    Tout = Tout if is_list_or_tuple else [Tout]
    Tout = [_as_dtype_or_type_spec(t) for t in Tout]
    handle_composite_tensors = use_eager_py_func and (any((isinstance(v, composite_tensor.CompositeTensor) for v in inp)) or any((isinstance(t, type_spec.TypeSpec) for t in Tout)))
    if handle_composite_tensors:
        (func, inp, Tout, out_structure) = _wrap_for_composites(func, inp, Tout)
    if use_eager_py_func:
        func = EagerFunc(func, Tout, is_grad_func)
    if tf_inspect.isfunction(original_func):
        original_func.ag_dnc_wrapper__ = func
    token = _py_funcs.insert(func)
    graph = ops.get_default_graph()
    while True:
        current_graph = graph
        if isinstance(graph, function._FuncGraph):
            graph = graph._outer_graph
        elif isinstance(graph, func_graph.FuncGraph):
            graph = graph.outer_graph
        if graph is current_graph:
            break
    if not hasattr(graph, '_py_funcs_used_in_graph'):
        graph._py_funcs_used_in_graph = []
    graph._py_funcs_used_in_graph.append(func)
    if use_eager_py_func:
        result = gen_script_ops.eager_py_func(input=inp, token=token, is_async=context.is_async(), Tout=Tout, name=name)
    elif stateful:
        result = gen_script_ops.py_func(input=inp, token=token, Tout=Tout, name=name)
    else:
        result = gen_script_ops.py_func_stateless(input=inp, token=token, Tout=Tout, name=name)
    if handle_composite_tensors and Tout:
        result = nest.pack_sequence_as(out_structure, result, expand_composites=True)
    return result if is_list_or_tuple else result[0]

@ops.RegisterGradient('EagerPyFunc')
def _EagerPyFuncGrad(op, *dy):
    if False:
        while True:
            i = 10
    'Computes the gradient of an EagerPyFunc.'
    token = op.get_attr('token')

    def eagerly_executed_grad(*dy):
        if False:
            i = 10
            return i + 15
        (tape, eager_inputs, eager_outputs) = tape_cache.pop(compat.as_bytes(token))
        return tape.gradient(eager_outputs, eager_inputs, output_gradients=dy)
    with ops.control_dependencies(op.outputs):
        gradient_op = _internal_py_func(func=eagerly_executed_grad, inp=dy, Tout=[tensor.dtype for tensor in op.inputs], use_eager_py_func=True, is_grad_func=True)
    if not context.executing_eagerly():
        func = _py_funcs.get(token.decode())
        assert isinstance(func, EagerFunc), f'EagerPyFuncGrad called on a non-EagerFunc object: {func}.'
        func.set_support_graph_mode_gradient()
    return gradient_op

def _check_args_and_maybe_make_decorator(script_op, script_op_name, func=None, inp=None, Tout=None, **kwargs):
    if False:
        return 10
    'Checks the arguments and returns a decorator if func is None.'
    if Tout is None:
        raise TypeError(f"Missing required argument: 'Tout'\n  If using {script_op_name} as a decorator, set `Tout`\n  **by name** above the function:\n  `@{script_op_name}(Tout=tout)`")
    if func is None:
        if inp is not None:
            raise TypeError(f"Don't set the `inp` argument when using {script_op_name} as a decorator (`func=None`).")

        def py_function_decorator(fun):
            if False:
                print('Hello World!')

            @functools.wraps(fun)
            def py_function_wrapper(*args):
                if False:
                    return 10
                return script_op(fun, inp=args, Tout=Tout, **kwargs)
            return py_function_wrapper
        return py_function_decorator
    if inp is None:
        raise TypeError(f'Missing argument `inp`:\n  You must set the `inp` argument (the list of arguments to the\n  function), unless you use `{script_op_name}` as a decorator(`func=None`).')
    return None

@tf_export('py_function')
@dispatch.add_dispatch_support
def eager_py_func(func=None, inp=None, Tout=None, name=None):
    if False:
        print('Hello World!')
    "Wraps a python function into a TensorFlow op that executes it eagerly.\n\n  Using `tf.py_function` inside a `tf.function` allows you to run a python\n  function using eager execution, inside the `tf.function`'s graph.\n  This has two main affects:\n\n  1. This allows you to use nofunc=None, inp=None, Tout=Nonen tensorflow code\n  inside your `tf.function`.\n  2. It allows you to run python control logic in a `tf.function` without\n  relying on `tf.autograph` to convert the code to use tensorflow control logic\n  (tf.cond, tf.while_loop).\n\n  Both of these features can be useful for debgging.\n\n  Since `tf.py_function` operates on `Tensor`s it is still\n  differentiable (once).\n\n  There are two ways to use this function:\n\n  ### As a decorator\n\n  Use `tf.py_function` as a decorator to ensure the function always runs\n  eagerly.\n\n  When using `tf.py_function` as a decorator:\n\n  * you must set `Tout`\n  * you may set `name`\n  * you must not set `func` or `inp`\n\n  For example, you might use `tf.py_function` to\n  implement the log huber function.\n\n  >>> @tf.py_function(Tout=tf.float32)\n  ... def py_log_huber(x, m):\n  ...   print('Running with eager execution.')\n  ...   if tf.abs(x) <= m:\n  ...     return x**2\n  ...   else:\n  ...     return m**2 * (1 - 2 * tf.math.log(m) + tf.math.log(x**2))\n\n  Under eager execution the function operates normally:\n\n  >>> x = tf.constant(1.0)\n  >>> m = tf.constant(2.0)\n  >>>\n  >>> print(py_log_huber(x,m).numpy())\n  Running with eager execution.\n  1.0\n\n  Inside a `tf.function` the `tf.py_function` is not converted to a `tf.Graph`.:\n\n  >>> @tf.function\n  ... def tf_wrapper(x):\n  ...   print('Tracing.')\n  ...   m = tf.constant(2.0)\n  ...   return py_log_huber(x,m)\n\n  The `tf.py_function` only executes eagerly, and only when the `tf.function`\n  is called:\n\n  >>> print(tf_wrapper(x).numpy())\n  Tracing.\n  Running with eager execution.\n  1.0\n  >>> print(tf_wrapper(x).numpy())\n  Running with eager execution.\n  1.0\n\n\n  Gradients work as exeppcted:\n\n  >>> with tf.GradientTape() as t:\n  ...   t.watch(x)\n  ...   y = tf_wrapper(x)\n  Running with eager execution.\n  >>>\n  >>> t.gradient(y, x).numpy()\n  2.0\n\n  ### Inplace\n\n  You can also skip the decorator and use `tf.py_function` inplace.\n  This form can a useful shortcut if you don't control the function's source,\n  but it is harder to read.\n\n  >>> # No decorator\n  >>> def log_huber(x, m):\n  ...   if tf.abs(x) <= m:\n  ...     return x**2\n  ...   else:\n  ...     return m**2 * (1 - 2 * tf.math.log(m) + tf.math.log(x**2))\n  >>>\n  >>> x = tf.constant(1.0)\n  >>> m = tf.constant(2.0)\n  >>>\n  >>> tf.py_function(func=log_huber, inp=[x, m], Tout=tf.float32).numpy()\n  1.0\n\n  ### More info\n\n  You can also use `tf.py_function` to debug your models at runtime\n  using Python tools, i.e., you can isolate portions of your code that\n  you want to debug, wrap them in Python functions and insert `pdb` tracepoints\n  or print statements as desired, and wrap those functions in\n  `tf.py_function`.\n\n  For more information on eager execution, see the\n  [Eager guide](https://tensorflow.org/guide/eager).\n\n  `tf.py_function` is similar in spirit to `tf.numpy_function`, but unlike\n  the latter, the former lets you use TensorFlow operations in the wrapped\n  Python function. In particular, while `tf.compat.v1.py_func` only runs on CPUs\n  and wraps functions that take NumPy arrays as inputs and return NumPy arrays\n  as outputs, `tf.py_function` can be placed on GPUs and wraps functions\n  that take Tensors as inputs, execute TensorFlow operations in their bodies,\n  and return Tensors as outputs.\n\n  Note: We recommend to avoid using `tf.py_function` outside of prototyping\n  and experimentation due to the following known limitations:\n\n  * Calling `tf.py_function` will acquire the Python Global Interpreter Lock\n    (GIL) that allows only one thread to run at any point in time. This will\n    preclude efficient parallelization and distribution of the execution of the\n    program.\n\n  * The body of the function (i.e. `func`) will not be serialized in a\n    `GraphDef`. Therefore, you should not use this function if you need to\n    serialize your model and restore it in a different environment.\n\n  * The operation must run in the same address space as the Python program\n    that calls `tf.py_function()`. If you are using distributed\n    TensorFlow, you must run a `tf.distribute.Server` in the same process as the\n    program that calls `tf.py_function()` and you must pin the created\n    operation to a device in that server (e.g. using `with tf.device():`).\n\n  * Currently `tf.py_function` is not compatible with XLA. Calling\n    `tf.py_function` inside `tf.function(jit_compile=True)` will raise an\n    error.\n\n  Args:\n    func: A Python function that accepts `inp` as arguments, and returns a value\n      (or list of values) whose type is described by `Tout`. Do not set `func`\n      when using `tf.py_function` as a decorator.\n    inp: Input arguments for `func`.  A list whose elements are `Tensor`s or\n      `CompositeTensors` (such as `tf.RaggedTensor`); or a single `Tensor` or\n      `CompositeTensor`. Do not set `inp` when using `tf.py_function` as a\n      decorator.\n    Tout: The type(s) of the value(s) returned by `func`.  One of the following.\n      * If `func` returns a `Tensor` (or a value that can be converted to a\n        Tensor): the `tf.DType` for that value. \n      * If `func` returns a `CompositeTensor`: The `tf.TypeSpec` for that value.\n      * If `func` returns `None`: the empty list (`[]`). \n      * If `func` returns a list of `Tensor` and `CompositeTensor` values: a\n        corresponding list of `tf.DType`s and `tf.TypeSpec`s for each value.\n    name: A name for the operation (optional).\n\n  Returns:\n    * If `func` is `None` this returns a decorator that will ensure the\n    decorated function will always run with eager execution even if called\n    from a `tf.function`/`tf.Graph`.\n    * If used `func` is not `None` this executes `func` with eager execution\n    and returns the result: a `Tensor`, `CompositeTensor`, or list of\n    `Tensor` and `CompositeTensor`; or an empty list if `func` returns `None`.\n  "
    decorator = _check_args_and_maybe_make_decorator(eager_py_func, 'tf.py_function', func=func, inp=inp, Tout=Tout, name=name)
    if decorator is not None:
        return decorator
    if ops.executing_eagerly_outside_functions():
        with ops.device(context.context().host_address_space()):
            return _internal_py_func(func=func, inp=inp, Tout=Tout, use_eager_py_func=True, name=name)
    return _internal_py_func(func=func, inp=inp, Tout=Tout, use_eager_py_func=True, name=name)

def py_func_common(func, inp, Tout, stateful=True, name=None):
    if False:
        print('Hello World!')
    'Wraps a python function and uses it as a TensorFlow op.\n\n  Given a python function `func`, which takes numpy arrays as its\n  arguments and returns numpy arrays as its outputs, wrap this function as an\n  operation in a TensorFlow graph. The following snippet constructs a simple\n  TensorFlow graph that invokes the `np.sinh()` NumPy function as a operation\n  in the graph:\n\n  ```python\n  def my_func(x):\n    # x will be a numpy array with the contents of the placeholder below\n    return np.sinh(x)\n  input = tf.compat.v1.placeholder(tf.float32)\n  y = tf.compat.v1.py_func(my_func, [input], tf.float32)\n  ```\n\n  **N.B.** The `tf.compat.v1.py_func()` operation has the following known\n  limitations:\n\n  * The body of the function (i.e. `func`) will not be serialized in a\n    `GraphDef`. Therefore, you should not use this function if you need to\n    serialize your model and restore it in a different environment.\n\n  * The operation must run in the same address space as the Python program\n    that calls `tf.compat.v1.py_func()`. If you are using distributed\n    TensorFlow, you\n    must run a `tf.distribute.Server` in the same process as the program that\n    calls\n    `tf.compat.v1.py_func()` and you must pin the created operation to a device\n    in that\n    server (e.g. using `with tf.device():`).\n\n  Note: It produces tensors of unknown shape and rank as shape inference\n    does not work on arbitrary Python code.\n    If you need the shape, you need to set it based on statically\n    available information.\n\n    E.g.\n    ```python\n    import tensorflow as tf\n    import numpy as np\n\n    def make_synthetic_data(i):\n        return np.cast[np.uint8](i) * np.ones([20,256,256,3],\n                dtype=np.float32) / 10.\n\n    def preprocess_fn(i):\n        ones = tf.py_function(make_synthetic_data,[i],tf.float32)\n        ones.set_shape(tf.TensorShape([None, None, None, None]))\n        ones = tf.image.resize(ones, [224,224])\n        return ones\n\n    ds = tf.data.Dataset.range(10)\n    ds = ds.map(preprocess_fn)\n    ```\n\n  Args:\n    func: A Python function, which accepts `ndarray` objects as arguments and\n      returns a list of `ndarray` objects (or a single `ndarray`). This function\n      must accept as many arguments as there are tensors in `inp`, and these\n      argument types will match the corresponding `tf.Tensor` objects in `inp`.\n      The returns `ndarray`s must match the number and types defined `Tout`.\n      Important Note: Input and output numpy `ndarray`s of `func` are not\n        guaranteed to be copies. In some cases their underlying memory will be\n        shared with the corresponding TensorFlow tensors. In-place modification\n        or storing `func` input or return values in python datastructures\n        without explicit (np.)copy can have non-deterministic consequences.\n    inp: A list of `Tensor` objects.\n    Tout: A list or tuple of tensorflow data types or a single tensorflow data\n      type if there is only one, indicating what `func` returns.\n    stateful: (Boolean.) If True, the function should be considered stateful. If\n      a function is stateless, when given the same input it will return the same\n      output and have no observable side effects. Optimizations such as common\n      subexpression elimination are only performed on stateless operations.\n    name: A name for the operation (optional).\n\n  Returns:\n    A list of `Tensor` or a single `Tensor` which `func` computes.\n\n  @compatibility(TF2)\n\n  This name was deprecated and removed in TF2, but `tf.numpy_function` is a\n  near-exact replacement, just drop the `stateful` argument (all\n  `tf.numpy_function` calls are considered stateful). It is compatible with\n  eager execution and `tf.function`.\n\n  `tf.py_function` is a close but not an exact replacement, passing TensorFlow\n  tensors to the wrapped function instead of NumPy arrays, which provides\n  gradients and can take advantage of accelerators.\n\n  Before:\n\n  >>> def fn_using_numpy(x):\n  ...   x[0] = 0.\n  ...   return x\n  >>> tf.compat.v1.py_func(fn_using_numpy, inp=[tf.constant([1., 2.])],\n  ...     Tout=tf.float32, stateful=False)\n  <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0., 2.], dtype=float32)>\n\n  After:\n\n  >>> tf.numpy_function(fn_using_numpy, inp=[tf.constant([1., 2.])],\n  ...     Tout=tf.float32)\n  <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0., 2.], dtype=float32)>\n\n  @end_compatibility\n\n  '
    if context.executing_eagerly():
        result = func(*[np.array(x) for x in inp])
        result = nest.flatten(result)
        result = [x if x is None else ops.convert_to_tensor(x) for x in result]
        if len(result) == 1:
            (result,) = result
        return result
    if ops.executing_eagerly_outside_functions():
        with ops.device(context.context().host_address_space()):
            return _internal_py_func(func=func, inp=inp, Tout=Tout, stateful=stateful, use_eager_py_func=False, name=name)
    return _internal_py_func(func=func, inp=inp, Tout=Tout, stateful=stateful, use_eager_py_func=False, name=name)

@deprecation.deprecated(date=None, instructions="tf.py_func is deprecated in TF V2. Instead, there are two\n    options available in V2.\n    - tf.py_function takes a python function which manipulates tf eager\n    tensors instead of numpy arrays. It's easy to convert a tf eager tensor to\n    an ndarray (just call tensor.numpy()) but having access to eager tensors\n    means `tf.py_function`s can use accelerators such as GPUs as well as\n    being differentiable using a gradient tape.\n    - tf.numpy_function maintains the semantics of the deprecated tf.py_func\n    (it is not differentiable, and manipulates numpy arrays). It drops the\n    stateful argument making all functions stateful.\n    ")
@tf_export(v1=['py_func'])
@dispatch.add_dispatch_support
def py_func(func, inp, Tout, stateful=True, name=None):
    if False:
        for i in range(10):
            print('nop')
    return py_func_common(func, inp, Tout, stateful, name=name)
py_func.__doc__ = '%s' % py_func_common.__doc__

@tf_export('numpy_function')
@dispatch.add_dispatch_support
def numpy_function(func=None, inp=None, Tout=None, stateful=True, name=None):
    if False:
        return 10
    "Wraps a python function and uses it as a TensorFlow op.\n\n  Given a python function `func` wrap this function as an operation in a\n  `tf.function`. `func` must take numpy arrays as its arguments and\n  return numpy arrays as its outputs.\n\n  There are two ways to use `tf.numpy_function`.\n\n  ### As a decorator\n\n  When using `tf.numpy_function` as a decorator:\n\n  * you must set `Tout`\n  * you may set `name`\n  * you must not set `func` or `inp`\n\n  >>> @tf.numpy_function(Tout=tf.float32)\n  ... def my_numpy_func(x):\n  ...   # x will be a numpy array with the contents of the input to the\n  ...   # tf.function\n  ...   print(f'executing eagerly, {x=}')\n  ...   return np.sinh(x)\n\n  The function runs eagerly:\n\n  >>> my_numpy_func(1.0).numpy()\n  executing eagerly, x=1.0\n  1.17520\n\n  The behavior doesn't change inside a `tf.function`:\n\n  >>> @tf.function(input_signature=[tf.TensorSpec(None, tf.float32)])\n  ... def tf_function(input):\n  ...   y = tf.numpy_function(my_numpy_func, [input], tf.float32)\n  ...   return y\n  >>> tf_function(tf.constant(1.)).numpy()\n  executing eagerly, x=array(1.)\n  1.17520\n\n  ### Inplace\n\n  This form can be useful if you don't control the function's source,\n  but it is harder to read.\n\n  Here is the same function with no decorator:\n\n  >>> def my_func(x):\n  ...   # x will be a numpy array with the contents of the input to the\n  ...   # tf.function\n  ...   print(f'executing eagerly, {x=}')\n  ...   return np.sinh(x)\n\n  To run `tf.numpy_function` inplace, pass the function, its inputs, and the\n  output type in a single call to `tf.numpy_function`:\n\n  >>> tf.numpy_function(my_func, [tf.constant(1.0)], tf.float32)\n  executing eagerly, x=array(1.)\n  1.17520\n\n  ### More info\n\n  Comparison to `tf.py_function`:\n  `tf.py_function` and `tf.numpy_function` are very similar, except that\n  `tf.numpy_function` takes numpy arrays, and not `tf.Tensor`s. If you want the\n  function to contain `tf.Tensors`, and have any TensorFlow operations executed\n  in the function be differentiable, please use `tf.py_function`.\n\n  Note: We recommend to avoid using `tf.numpy_function` outside of\n  prototyping and experimentation due to the following known limitations:\n\n  * Calling `tf.numpy_function` will acquire the Python Global Interpreter Lock\n    (GIL) that allows only one thread to run at any point in time. This will\n    preclude efficient parallelization and distribution of the execution of the\n    program. Therefore, you are discouraged to use `tf.numpy_function` outside\n    of prototyping and experimentation.\n\n  * The body of the function (i.e. `func`) will not be serialized in a\n    `tf.SavedModel`. Therefore, you should not use this function if you need to\n    serialize your model and restore it in a different environment.\n\n  * The operation must run in the same address space as the Python program\n    that calls `tf.numpy_function()`. If you are using distributed\n    TensorFlow, you must run a `tf.distribute.Server` in the same process as the\n    program that calls `tf.numpy_function`  you must pin the created\n    operation to a device in that server (e.g. using `with tf.device():`).\n\n  * Currently `tf.numpy_function` is not compatible with XLA. Calling\n    `tf.numpy_function` inside `tf.function(jit_compile=True)` will raise an\n    error.\n\n  * Since the function takes numpy arrays, you cannot take gradients\n    through a numpy_function. If you require something that is differentiable,\n    please consider using tf.py_function.\n\n  Args:\n    func: A Python function, which accepts `numpy.ndarray` objects as arguments\n      and returns a list of `numpy.ndarray` objects (or a single\n      `numpy.ndarray`). This function must accept as many arguments as there are\n      tensors in `inp`, and these argument types will match the corresponding\n      `tf.Tensor` objects in `inp`. The returns `numpy.ndarray`s must match the\n      number and types defined `Tout`. Important Note: Input and output\n      `numpy.ndarray`s of `func` are not guaranteed to be copies. In some cases\n      their underlying memory will be shared with the corresponding TensorFlow\n      tensors. In-place modification or storing `func` input or return values in\n      python datastructures without explicit (np.)copy can have\n      non-deterministic consequences.\n    inp: A list of `tf.Tensor` objects.\n    Tout: A list or tuple of tensorflow data types or a single tensorflow data\n      type if there is only one, indicating what `func` returns.\n    stateful: (Boolean.) Setting this argument to False tells the runtime to\n      treat the function as stateless, which enables certain optimizations. A\n      function is stateless when given the same input it will return the same\n      output and have no side effects; its only purpose is to have a return\n      value. The behavior for a stateful function with the `stateful` argument\n      False is undefined. In particular, caution should be taken when mutating\n      the input arguments as this is a stateful operation.\n    name: (Optional) A name for the operation.\n\n  Returns:\n    * If `func` is `None` this returns a decorator that will ensure the\n      decorated function will always run with eager execution even if called\n      from a `tf.function`/`tf.Graph`.\n    * If used `func` is not `None` this executes `func` with eager execution\n      and returns the result: A single or list of `tf.Tensor` which `func`\n      computes.\n  "
    decorator = _check_args_and_maybe_make_decorator(numpy_function, 'tf.numpy_function', func=func, inp=inp, Tout=Tout, stateful=stateful, name=name)
    if decorator is not None:
        return decorator
    return py_func_common(func, inp, Tout, stateful=stateful, name=name)

def _as_dtype_or_type_spec(t):
    if False:
        i = 10
        return i + 15
    return t if isinstance(t, type_spec.TypeSpec) else dtypes.as_dtype(t)

def _wrap_for_composites(func, inp, Tout):
    if False:
        print('Hello World!')
    "Wraps user inputs to support composite tensors for `py_function`.\n\n  1. Flattens `inp` to a list of Tensors (by flattening any composite tensors).\n  2. Creates a wrapper fuction for `func` that expects flat inputs and:\n     - Packs the inputs into the input structure expected by `func`.\n     - Calls `func` with the packed inputs.\n     - Checks that `func`'s output matches `Tout`.\n     - Flattens func`'s output to a list of Tensors (flattening any composite\n       tensors).\n\n  Args:\n    func: The function to wrap (`func` argument to `py_function`).\n    inp: The input arguments for func (`inp` argument to `py_function`).\n    Tout: The expected output types for func (`Tout` argument to `py_function).\n\n  Returns:\n    A tuple `(func, inp, Tout, out_structure)`, where `func` is the wrapped\n    function, `inp` is the flattened inputs, `Tout` is the list of expected\n    dtypes for the flattened outputs, and `out_structure` is the expected\n    output structure (which can be used to pack the output tensors).\n  "
    in_structure = [v if isinstance(v, composite_tensor.CompositeTensor) else 1 for v in inp]
    inp = nest.flatten_up_to(in_structure, inp, expand_composites=True)
    out_structure = Tout
    Tout = [v.dtype if isinstance(v, tensor_spec.TensorSpec) else v for v in nest.flatten(Tout, expand_composites=True)]

    def wrapped_func(*flat_inp):
        if False:
            i = 10
            return i + 15
        structured_inp = nest.pack_sequence_as(in_structure, flat_inp, expand_composites=True)
        out = func(*structured_inp)
        if not out_structure:
            return []
        if not isinstance(out, (list, tuple)):
            out = [out]
        flat_out = []
        for (elt, expected_type) in zip(out, out_structure):
            if isinstance(expected_type, type_spec.TypeSpec) and (not isinstance(expected_type, tensor_spec.TensorSpec)):
                if not expected_type.is_compatible_with(elt):
                    raise ValueError(f'py_function: func={func} returned {out!r}, which did not match Tout={out_structure!r}.\nIn particular, {elt!r} is not compatible with {expected_type!r}.')
                flat_out.extend(nest.flatten(elt, expand_composites=True))
            else:
                if isinstance(elt, composite_tensor.CompositeTensor):
                    raise ValueError(f'py_function: func={func} returned {out!r}, which did not match Tout={out_structure!r}.\nIn particular, {elt!r} is not a Tensor.')
                flat_out.append(elt)
        return flat_out
    return (wrapped_func, inp, Tout, out_structure)
ops.NotDifferentiable('PyFunc')
ops.NotDifferentiable('PyFuncStateless')