"""Decorator to overrides the gradient for a function."""
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import op_selector
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
VAR_OP_TYPES = ['VariableV2', 'VarHandleOp']

@tf_export('custom_gradient')
def custom_gradient(f=None):
    if False:
        while True:
            i = 10
    "Decorator to define a function with a custom gradient.\n\n  This decorator allows fine grained control over the gradients of a sequence\n  for operations.  This may be useful for multiple reasons, including providing\n  a more efficient or numerically stable gradient for a sequence of operations.\n\n  For example, consider the following function that commonly occurs in the\n  computation of cross entropy and log likelihoods:\n\n  ```python\n  def log1pexp(x):\n    return tf.math.log(1 + tf.exp(x))\n  ```\n\n  Due to numerical instability, the gradient of this function evaluated at x=100\n  is NaN.  For example:\n\n  ```python\n  with tf.GradientTape() as tape:\n    tape.watch(x)\n    y=log1pexp(x)\n  dy_dx = tape.gradient(y, x) # Will be NaN when evaluated.\n  ```\n\n  The gradient expression can be analytically simplified to provide numerical\n  stability:\n\n  ```python\n  @tf.custom_gradient\n  def log1pexp(x):\n    e = tf.exp(x)\n    def grad(upstream):\n      return upstream * (1 - 1 / (1 + e))\n    return tf.math.log(1 + e), grad\n  ```\n\n  With this definition, the gradient `dy_dx` at `x = 100` will be correctly\n  evaluated as 1.0.\n\n  The variable `upstream` is defined as the upstream gradient. i.e. the gradient\n  from all the layers or functions originating from this layer. The above\n  example has no upstream functions, therefore `upstream = dy/dy = 1.0`.\n\n  Assume that `x_i` is `log1pexp` in the forward pass `x_1 = x_1(x_0)`,\n  `x_2 = x_2(x_1)`, ..., `x_i = x_i(x_i-1)`, ..., `x_n = x_n(x_n-1)`. By\n  chain rule we know that `dx_n/dx_0 = dx_n/dx_n-1 * dx_n-1/dx_n-2 * ... *\n  dx_i/dx_i-1 * ... * dx_1/dx_0`.\n\n  In this case the gradient of our current function defined as\n  `dx_i/dx_i-1 = (1 - 1 / (1 + e))`. The upstream gradient `upstream` would be\n  `dx_n/dx_n-1 * dx_n-1/dx_n-2 * ... * dx_i+1/dx_i`. The upstream gradient\n  multiplied by the current gradient is then passed downstream.\n\n  In case the function takes multiple variables as input, the `grad`\n  function must also return  the same number of variables.\n  We take the function `z = x * y` as an example.\n\n  >>> @tf.custom_gradient\n  ... def bar(x, y):\n  ...   def grad(upstream):\n  ...     dz_dx = y\n  ...     dz_dy = x\n  ...     return upstream * dz_dx, upstream * dz_dy\n  ...   z = x * y\n  ...   return z, grad\n  >>> x = tf.constant(2.0, dtype=tf.float32)\n  >>> y = tf.constant(3.0, dtype=tf.float32)\n  >>> with tf.GradientTape(persistent=True) as tape:\n  ...   tape.watch(x)\n  ...   tape.watch(y)\n  ...   z = bar(x, y)\n  >>> z\n  <tf.Tensor: shape=(), dtype=float32, numpy=6.0>\n  >>> tape.gradient(z, x)\n  <tf.Tensor: shape=(), dtype=float32, numpy=3.0>\n  >>> tape.gradient(z, y)\n  <tf.Tensor: shape=(), dtype=float32, numpy=2.0>\n\n  Nesting custom gradients can lead to unintuitive results. The default\n  behavior does not correspond to n-th order derivatives. For example\n\n  ```python\n  @tf.custom_gradient\n  def op(x):\n    y = op1(x)\n    @tf.custom_gradient\n    def grad_fn(dy):\n      gdy = op2(x, y, dy)\n      def grad_grad_fn(ddy):  # Not the 2nd order gradient of op w.r.t. x.\n        return op3(x, y, dy, ddy)\n      return gdy, grad_grad_fn\n    return y, grad_fn\n  ```\n\n  The function `grad_grad_fn` will be calculating the first order gradient\n  of `grad_fn` with respect to `dy`, which is used to generate forward-mode\n  gradient graphs from backward-mode gradient graphs, but is not the same as\n  the second order gradient of `op` with respect to `x`.\n\n  Instead, wrap nested `@tf.custom_gradients` in another function:\n\n  ```python\n  @tf.custom_gradient\n  def op_with_fused_backprop(x):\n    y, x_grad = fused_op(x)\n    def first_order_gradient(dy):\n      @tf.custom_gradient\n      def first_order_custom(unused_x):\n        def second_order_and_transpose(ddy):\n          return second_order_for_x(...), gradient_wrt_dy(...)\n        return x_grad, second_order_and_transpose\n      return dy * first_order_custom(x)\n    return y, first_order_gradient\n  ```\n\n  Additional arguments to the inner `@tf.custom_gradient`-decorated function\n  control the expected return values of the innermost function.\n\n  The examples above illustrate how to specify custom gradients for functions\n  which do not read from variables. The following example uses variables, which\n  require special handling because they are effectively inputs of the forward\n  function.\n\n  >>> weights = tf.Variable(tf.ones([2]))  # Trainable variable weights\n  >>> @tf.custom_gradient\n  ... def linear_poly(x):\n  ...   # Creating polynomial\n  ...   poly = weights[1] * x + weights[0]\n  ...\n  ...   def grad_fn(dpoly, variables):\n  ...     # dy/dx = weights[1] and we need to left multiply dpoly\n  ...     grad_xs = dpoly * weights[1]  # Scalar gradient\n  ...\n  ...     grad_vars = []  # To store gradients of passed variables\n  ...     assert variables is not None\n  ...     assert len(variables) == 1\n  ...     assert variables[0] is weights\n  ...     # Manually computing dy/dweights\n  ...     dy_dw = dpoly * tf.stack([x ** 1, x ** 0])\n  ...     grad_vars.append(\n  ...         tf.reduce_sum(tf.reshape(dy_dw, [2, -1]), axis=1)\n  ...     )\n  ...     return grad_xs, grad_vars\n  ...   return poly, grad_fn\n  >>> x = tf.constant([1., 2., 3.])\n  >>> with tf.GradientTape(persistent=True) as tape:\n  ...   tape.watch(x)\n  ...   poly = linear_poly(x)\n  >>> poly # poly = x + 1\n  <tf.Tensor: shape=(3,),\n    dtype=float32,\n    numpy=array([2., 3., 4.], dtype=float32)>\n  >>> tape.gradient(poly, x)  # conventional scalar gradient dy/dx\n  <tf.Tensor: shape=(3,),\n    dtype=float32,\n    numpy=array([1., 1., 1.], dtype=float32)>\n  >>> tape.gradient(poly, weights)\n  <tf.Tensor: shape=(2,), dtype=float32, numpy=array([6., 3.], dtype=float32)>\n\n  Above example illustrates usage of trainable variable `weights`.\n  In the example, the inner `grad_fn` accepts an extra `variables` input\n  parameter and also returns an extra `grad_vars` output. That extra argument\n  is passed if the forward function reads any variables. You need to\n  compute the gradient w.r.t. each of those `variables` and output it as a list\n  of `grad_vars`. Note here that default value of `variables` is set to `None`\n  when no variables are used in the forward function.\n\n  It should be noted `tf.GradientTape` is still watching the forward pass of a\n  `tf.custom_gradient`, and will use the ops it watches. As a consequence,\n  calling `tf.function` while the tape is still watching leads\n  to a gradient graph being built. If an op is used in `tf.function` without\n  registered gradient, a `LookupError` will be raised.\n\n  Users can insert `tf.stop_gradient` to customize this behavior. This\n  is demonstrated in the example below. `tf.random.shuffle` does not have a\n  registered gradient. As a result `tf.stop_gradient` is used to avoid the\n  `LookupError`.\n\n  ```python\n  x = tf.constant([0.3, 0.5], dtype=tf.float32)\n\n  @tf.custom_gradient\n  def test_func_with_stop_grad(x):\n    @tf.function\n    def _inner_func():\n      # Avoid exception during the forward pass\n      return tf.stop_gradient(tf.random.shuffle(x))\n      # return tf.random.shuffle(x)  # This will raise\n\n    res = _inner_func()\n    def grad(upstream):\n      return upstream  # Arbitrarily defined custom gradient\n    return res, grad\n\n  with tf.GradientTape() as g:\n    g.watch(x)\n    res = test_func_with_stop_grad(x)\n\n  g.gradient(res, x)\n  ```\n\n  See also `tf.RegisterGradient` which registers a gradient function for a\n  primitive TensorFlow operation. `tf.custom_gradient` on the other hand allows\n  for fine grained control over the gradient computation of a sequence of\n  operations.\n\n  Note that if the decorated function uses `Variable`s, the enclosing variable\n  scope must be using\n  [ResourceVariables](https://www.tensorflow.org/guide/migrate/tf1_vs_tf2#resourcevariables_instead_of_referencevariables).\n\n  Args:\n    f: function `f(*x)` that returns a tuple `(y, grad_fn)` where:\n       - `x` is a sequence of (nested structures of) `Tensor` inputs to the\n         function.\n       - `y` is a (nested structure of) `Tensor` outputs of applying TensorFlow\n         operations in `f` to `x`.\n       - `grad_fn` is a function with the signature `g(*grad_ys)` which returns\n         a list of `Tensor`s the same size as (flattened) `x` - the derivatives\n         of `Tensor`s in `y` with respect to the `Tensor`s in `x`.  `grad_ys` is\n         a sequence of `Tensor`s the same size as (flattened) `y` holding the\n         initial value gradients for each `Tensor` in `y`.\n\n         In a pure mathematical sense, a vector-argument vector-valued function\n         `f`'s derivatives should be its Jacobian matrix `J`. Here we are\n         expressing the Jacobian `J` as a function `grad_fn` which defines how\n         `J` will transform a vector `grad_ys` when left-multiplied with it\n         (`grad_ys * J`, the vector-Jacobian product, or VJP). This functional\n         representation of a matrix is convenient to use for chain-rule\n         calculation (in e.g. the back-propagation algorithm).\n\n         If `f` uses `Variable`s (that are not part of the\n         inputs), i.e. through `get_variable`, then `grad_fn` should have\n         signature `g(*grad_ys, variables=None)`, where `variables` is a list of\n         the `Variable`s, and return a 2-tuple `(grad_xs, grad_vars)`, where\n         `grad_xs` is the same as above, and `grad_vars` is a `list<Tensor>`\n         with the derivatives of `Tensor`s in `y` with respect to the variables\n         (that is, grad_vars has one Tensor per variable in variables).\n\n  Returns:\n    A function `h(x)` which returns the same value as `f(x)[0]` and whose\n    gradient (as calculated by `tf.gradients`) is determined by `f(x)[1]`.\n  "
    if f is None:
        return lambda f: custom_gradient(f=f)

    @Bind.decorator
    def decorated(wrapped, args, kwargs):
        if False:
            return 10
        'Decorated function with custom gradient.'
        if context.executing_eagerly():
            return _eager_mode_decorator(wrapped, args, kwargs)
        else:
            return _graph_mode_decorator(wrapped, args, kwargs)
    return tf_decorator.make_decorator(f, decorated(f))

class Bind:
    """When called evaluates `d(f, args, kwargs)` but supports binding `f`.

  >>> @Bind.decorator
  ... def my_decorator(f, args, kwargs):
  ...   print("my_decorator called with", args, kwargs)
  ...   return f(*args, **kwargs)

  >>> class Foo:
  ...   @my_decorator
  ...   def bar(self, a, b, c):
  ...     return a * b * c

  >>> Foo.bar(None, 1, 2, c=3)
  my_decorator called with (None, 1, 2) {'c': 3}
  6

  >>> foo = Foo()
  >>> foo.bar(1, 2, c=3)
  my_decorator called with (1, 2) {'c': 3}
  6
  """

    @classmethod
    def decorator(cls, d):
        if False:
            while True:
                i = 10
        return lambda f: Bind(f, d)

    def __init__(self, f, d):
        if False:
            for i in range(10):
                print('nop')
        self._f = f
        self._d = d

    def __get__(self, instance, owner):
        if False:
            i = 10
            return i + 15
        if instance is not None:
            f = self._f.__get__(instance, owner)
            return tf_decorator.make_decorator(f, Bind(f, self._d))
        else:
            return self

    def __call__(self, *a, **k):
        if False:
            return 10
        return self._d(self._f, a, k)

def get_variable_by_name(var_name):
    if False:
        return 10
    'Given a variable name, retrieves a handle on the tensorflow Variable.'
    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)

    def _filter_fn(item):
        if False:
            for i in range(10):
                print('nop')
        try:
            return var_name == item.op.name
        except AttributeError:
            return False
    candidate_vars = list(filter(_filter_fn, global_vars))
    if len(candidate_vars) >= 1:
        candidate_vars = [v for v in candidate_vars if v.trainable]
    else:
        raise ValueError('Unsuccessful at finding variable {}.'.format(var_name))
    if len(candidate_vars) == 1:
        return candidate_vars[0]
    elif len(candidate_vars) > 1:
        raise ValueError('Unsuccessful at finding trainable variable {}. Number of candidates: {}. Candidates: {}'.format(var_name, len(candidate_vars), candidate_vars))
    else:
        return None

def _get_dependent_variables(input_ops, output_ops):
    if False:
        print('Hello World!')
    'Finds variables involved in the subgraph between input_ops and output_ops.\n\n  Args:\n    input_ops: Flattened list of input ops\n    output_ops: Flattened list of output ops\n\n  Returns:\n    A list of variables\n  '
    output_ops = nest.map_structure(gen_array_ops.identity, output_ops)
    inbetween_ops = op_selector.get_backward_walk_ops(seed_ops=output_ops, stop_at_ts=input_ops, inclusive=False, only_differentiable=True)
    var_ops = (op for op in inbetween_ops if op.type in VAR_OP_TYPES)
    var_names = (op.name for op in var_ops)
    tf_vars = (get_variable_by_name(var_name) for var_name in var_names)
    tf_vars = [v for v in tf_vars if v is not None]
    return tf_vars

def generate_name():
    if False:
        while True:
            i = 10
    return 'CustomGradient-%s' % ops.uid()

def _graph_mode_decorator(f, args, kwargs):
    if False:
        while True:
            i = 10
    'Implement custom gradient decorator for graph mode.'
    if kwargs:
        raise ValueError('The custom_gradient decorator currently supports keywords arguments only when eager execution is enabled.')
    name = generate_name()
    args = variable_utils.convert_variables_to_tensors(args)
    args = nest.map_structure(ops.convert_to_tensor, args, expand_composites=True)
    current_var_scope = variable_scope.get_variable_scope()
    before_vars = set([v.ref() for v in current_var_scope.global_variables() + current_var_scope.local_variables()])
    with record.VariableWatcher() as variable_watcher:
        (result, grad_fn) = f(*args)
    flat_args = composite_tensor_gradient.get_flat_tensors_for_gradients(nest.flatten(args))
    flat_result = composite_tensor_gradient.get_flat_tensors_for_gradients(nest.flatten(result))
    flat_result_len = len(flat_result)
    after_vars = set([v.ref() for v in current_var_scope.global_variables() + current_var_scope.local_variables()])
    new_vars = after_vars - before_vars
    new_vars_list = [v.deref() for v in new_vars]
    for v in new_vars_list:
        if not resource_variable_ops.is_resource_variable(v):
            raise TypeError('All variables used by a function wrapped with @custom_gradient must be `ResourceVariable`s. Ensure that no `variable_scope` is created with `use_resource=False`.')
    variables_in_tape = frozenset([v.ref() for v in variable_watcher.watched_variables()])
    graphs = {getattr(o, 'graph', None) for o in flat_result}
    graphs.discard(None)
    if graphs:
        if len(graphs) > 1:
            raise ValueError('All custom_gradient outputs should be from the same graph')
        output_graph = graphs.pop()
        filtered_input_tensors = []
        for i in flat_args:
            if i.graph == output_graph:
                filtered_input_tensors.append(i)
    else:
        filtered_input_tensors = flat_args
    variables_in_subgraph = frozenset([v.ref() for v in _get_dependent_variables(input_ops=filtered_input_tensors, output_ops=flat_result)])
    variables = sorted([v.deref() for v in variables_in_subgraph.union(variables_in_tape)], key=lambda v: v.name)
    grad_argspec = tf_inspect.getfullargspec(grad_fn)
    variables_in_signature = 'variables' in grad_argspec.args or 'variables' in grad_argspec.kwonlyargs or grad_argspec.varkw
    if variables and (not variables_in_signature):
        raise TypeError("@tf.custom_gradient grad_fn must accept keyword argument 'variables', since function uses variables: {}".format(variables))
    if variables_in_signature and (not variables):
        logging.vlog(1, "@custom_gradient grad_fn has 'variables' in signature, but no ResourceVariables were used on the forward pass.")
    all_tensors = flat_result + flat_args + variables

    def tape_grad_fn(*result_grad_components):
        if False:
            return 10
        'Custom grad fn wrapper.'
        result_grads = composite_tensor_gradient.replace_flat_tensors_for_gradients(nest.flatten(result), result_grad_components[:flat_result_len])
        if not isinstance(result_grads, (list, tuple)):
            result_grads = [result_grads]
        if variables:
            (input_grads, variable_grads) = grad_fn(*result_grads, variables=variables)
            if len(variable_grads) != len(variables):
                raise ValueError('Must return gradient for each variable from @custom_gradient grad_fn.')
        else:
            input_grads = grad_fn(*result_grads)
            variable_grads = []
        input_grads = composite_tensor_gradient.get_flat_tensors_for_gradients(nest.flatten(input_grads))
        return [None] * flat_result_len + input_grads + variable_grads

    @ops.RegisterGradient(name)
    def internal_grad_fn(unused_op, *result_grads):
        if False:
            print('Hello World!')
        'Custom grad fn wrapper.'
        return tape_grad_fn(*result_grads)
    original_tensors = all_tensors
    with ops.get_default_graph().gradient_override_map({'IdentityN': name}):
        all_tensors = array_ops.identity_n(all_tensors)
    original_tensors = [ops.convert_to_tensor(x) for x in original_tensors]
    for (i, t) in enumerate(original_tensors):
        if t.dtype == dtypes.resource and hasattr(t, '_handle_data'):
            all_tensors[i]._handle_data = t._handle_data
    record.record_operation(f.__name__, all_tensors, original_tensors, tape_grad_fn)
    for (ot, t) in zip(original_tensors, all_tensors):
        handle_data_util.copy_handle_data(ot, t)
    flat_result = composite_tensor_gradient.replace_flat_tensors_for_gradients(nest.flatten(result), all_tensors[:flat_result_len])
    return nest.pack_sequence_as(result, flat_result)

def _eager_mode_decorator(f, args, kwargs):
    if False:
        while True:
            i = 10
    'Implement custom gradient decorator for eager mode.'
    with record.VariableWatcher() as variable_watcher:
        (result, grad_fn) = f(*args, **kwargs)
    flat_args = composite_tensor_gradient.get_flat_tensors_for_gradients(nest.flatten(args))
    flat_kwargs = composite_tensor_gradient.get_flat_tensors_for_gradients(nest.flatten(kwargs))
    all_inputs = flat_args + flat_kwargs
    variables = [v.deref() for v in set((v.ref() for v in variable_watcher.watched_variables())) if all((v.deref() is not i for i in all_inputs))]
    grad_argspec = tf_inspect.getfullargspec(grad_fn)
    if variables and 'variables' not in grad_argspec.args and ('variables' not in grad_argspec.kwonlyargs) and (not grad_argspec.varkw):
        raise TypeError("@tf.custom_gradient grad_fn must accept keyword argument 'variables', since function uses variables: {}".format(variables))
    flat_result = composite_tensor_gradient.get_flat_tensors_for_gradients(nest.flatten(result))
    flat_result = [gen_array_ops.identity(x) for x in flat_result]
    input_tensors = [ops.convert_to_tensor(x) for x in flat_args + list(variables)]
    recorded_inputs = input_tensors
    arg_count = len(flat_args)

    def actual_grad_fn(*result_grad_components):
        if False:
            while True:
                i = 10
        'Custom grad fn wrapper.'
        result_grads = composite_tensor_gradient.replace_flat_tensors_for_gradients(nest.flatten(result), result_grad_components)
        if not isinstance(result_grads, (list, tuple)):
            result_grads = [result_grads]
        if variables:
            (input_grads, variable_grads) = grad_fn(*result_grads, variables=variables)
            if len(variable_grads) != len(variables):
                raise ValueError('Must return gradient for each variable from @custom_gradient grad_fn.')
        else:
            input_grads = grad_fn(*result_grads)
            variable_grads = []
        flat_grads = composite_tensor_gradient.get_flat_tensors_for_gradients(nest.flatten(input_grads))
        if len(flat_grads) != arg_count:
            raise ValueError(f'custom_gradient function expected to return {arg_count} gradients, but returned {len(flat_grads)} instead.')
        return flat_grads + variable_grads
    record.record_operation(f.__name__, flat_result, recorded_inputs, actual_grad_fn)
    flat_result = composite_tensor_gradient.replace_flat_tensors_for_gradients(nest.flatten(result), flat_result)
    return nest.pack_sequence_as(result, flat_result)

@tf_export('recompute_grad')
def recompute_grad(f):
    if False:
        return 10
    'Defines a function as a recompute-checkpoint for the tape auto-diff.\n\n  Tape checkpointing is a technique to reduce the memory consumption of the\n  auto-diff tape:\n\n  - Without tape checkpointing operations and intermediate values are\n  recorded to the tape for use in the backward pass.\n\n  - With tape checkpointing, only the function call and its inputs are\n  recorded. During back-propagation the `recompute_grad` custom gradient\n  (`tf.custom_gradient`) recomputes the function under a localized Tape object.\n  This recomputation of the function during backpropagation performs redundant\n  calculation, but reduces the overall memory usage of the Tape.\n\n  >>> y = tf.Variable(1.0)\n\n  >>> def my_function(x):\n  ...   tf.print(\'running\')\n  ...   z = x*y\n  ...   return z\n\n  >>> my_function_recompute = tf.recompute_grad(my_function)\n\n  >>> with tf.GradientTape() as tape:\n  ...   r = tf.constant(1.0)\n  ...   for i in range(4):\n  ...     r = my_function_recompute(r)\n  running\n  running\n  running\n  running\n\n  >>> grad = tape.gradient(r, [y])\n  running\n  running\n  running\n  running\n\n  Without `recompute_grad`, the tape contains all intermitate steps, and no\n  recomputation is performed.\n\n  >>> with tf.GradientTape() as tape:\n  ...   r = tf.constant(1.0)\n  ...   for i in range(4):\n  ...     r = my_function(r)\n  running\n  running\n  running\n  running\n\n  >>> grad = tape.gradient(r, [y])\n\n\n  If `f` was a `tf.keras` `Model` or `Layer` object, methods and attributes\n  such as `f.variables` are not available on the returned function `g`.\n  Either keep a reference of `f` , or use `g.__wrapped__` for accessing\n  these variables and methods.\n\n\n  >>> def print_running_and_return(x):\n  ...   tf.print("running")\n  ...   return x\n\n  >>> model = tf.keras.Sequential([\n  ...   tf.keras.layers.Lambda(print_running_and_return),\n  ...   tf.keras.layers.Dense(2)\n  ... ])\n\n  >>> model_recompute = tf.recompute_grad(model)\n\n  >>> with tf.GradientTape(persistent=True) as tape:\n  ...   r = tf.constant([[1,2]])\n  ...   for i in range(4):\n  ...     r = model_recompute(r)\n  running\n  running\n  running\n  running\n\n  >>> grad = tape.gradient(r, model.variables)\n  running\n  running\n  running\n  running\n\n  Alternatively, use the `__wrapped__` attribute to access the original\n  model object.\n\n  >>> grad = tape.gradient(r, model_recompute.__wrapped__.variables)\n  running\n  running\n  running\n  running\n\n\n  Args:\n    f: function `f(*x)` that returns a `Tensor` or sequence of `Tensor` outputs.\n\n  Returns:\n    A function `g` wrapping `f` that defines a custom gradient, which recomputes\n    `f` on the backwards pass of a gradient call.\n  '

    @custom_gradient
    def inner(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Inner function closure for calculating gradients.'
        current_var_scope = variable_scope.get_variable_scope()
        with record.stop_recording():
            result = f(*args, **kwargs)

        def grad_wrapper(*wrapper_args, variables=None):
            if False:
                while True:
                    i = 10
            'Wrapper function to accomodate lack of kwargs in graph mode custom_gradient.'

            @custom_gradient
            def inner_recompute_grad(*dresult):
                if False:
                    while True:
                        i = 10
                'Nested custom gradient function for computing grads in reverse and forward mode autodiff.'
                with backprop.GradientTape() as t:
                    id_args = nest.map_structure(gen_array_ops.identity, args)
                    assert len(dresult) >= 1
                    if not context.executing_eagerly():
                        elem = math_ops.reduce_max(array_ops.reshape(dresult[0], [-1])[:1])
                        elem_bool = math_ops.cast(elem, dtypes.bool)
                        dresult_dep = array_ops.where_v2(elem_bool == elem_bool, 0.0, float('nan'))
                        id_args = nest.map_structure(lambda x: x + math_ops.cast(dresult_dep, x.dtype), id_args)
                    t.watch(id_args)
                    if variables is not None:
                        t.watch(variables)
                    with variable_scope.variable_scope(current_var_scope):
                        recomputed_result = f(*id_args, **kwargs)
                kw_vars = []
                if variables is not None:
                    kw_vars = list(variables)
                grads = t.gradient(recomputed_result, list(id_args) + kw_vars, output_gradients=dresult, unconnected_gradients=UnconnectedGradients.ZERO)

                def transpose(*t_args, **t_kwargs):
                    if False:
                        i = 10
                        return i + 15
                    'Gradient function calculation for forward mode autodiff.'
                    raise NotImplementedError('recompute_grad tried to transpose grad of {}. Consider not using recompute_grad in forward modeautodiff'.format(f.__name__))
                return ((grads[:len(id_args)], grads[len(id_args):]), transpose)
            return inner_recompute_grad(*wrapper_args)
        return (result, grad_wrapper)
    return tf_decorator.make_decorator(f, inner)

@tf_export('grad_pass_through')
def grad_pass_through(f):
    if False:
        return 10
    'Creates a grad-pass-through op with the forward behavior provided in f.\n\n  Use this function to wrap any op, maintaining its behavior in the forward\n  pass, but replacing the original op in the backward graph with an identity.\n  For example:\n\n  ```python\n  x = tf.Variable(1.0, name="x")\n  z = tf.Variable(3.0, name="z")\n\n  with tf.GradientTape() as tape:\n    # y will evaluate to 9.0\n    y = tf.grad_pass_through(x.assign)(z**2)\n  # grads will evaluate to 6.0\n  grads = tape.gradient(y, z)\n  ```\n\n  Another example is a \'differentiable\' moving average approximation, where\n  gradients are allowed to flow into the last value fed to the moving average,\n  but the moving average is still used for the forward pass:\n\n  ```python\n  x = ... # Some scalar value\n  # A moving average object, we don\'t need to know how this is implemented\n  moving_average = MovingAverage()\n  with backprop.GradientTape() as tape:\n    # mavg_x will evaluate to the current running average value\n    mavg_x = tf.grad_pass_through(moving_average)(x)\n  grads = tape.gradient(mavg_x, x) # grads will evaluate to 1.0\n  ```\n\n  Args:\n    f: function `f(*x)` that returns a `Tensor` or nested structure of `Tensor`\n      outputs.\n\n  Returns:\n    A function `h(x)` which returns the same values as `f(x)` and whose\n    gradients are the same as those of an identity function.\n  '

    @custom_gradient
    def _grad_pass_through_op(*args, **kwargs):
        if False:
            return 10

        def grad(*args, **kwargs):
            if False:
                print('Hello World!')
            variables = kwargs.get('variables')
            if variables is not None:
                return (args, [None] * len(variables))
            return args
        return (f(*args, **kwargs), grad)
    return tf_decorator.make_decorator(f, _grad_pass_through_op)