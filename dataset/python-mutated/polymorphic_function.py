"""API for defining graph functions with some additional eager semantics.

tf.function utilizes varying configurations of tracing compilation to allow
initializing `tf.Variable`s with subgraphs of the function. For example:

```python
class M(tf.Module):
  def __init__(self):
    self.v_opinit = None
    self.v_arginit = None

  @tf.function
  def __call__(self, x):
    # Variables are only created on the first call to the function. This is a
    # common pattern in layer libraries.
    if self.v_opinit is None:
      # self.v_opinit will outlive the function call, but `tf.ones` is traced as
      # part of the function body before the `tf.Variable` object is
      # created. This subgraph is easy to lift out of the function.
      self.v_opinit = tf.Variable(tf.ones([]))

      # If arguments feed into variable initialization, it can be very tricky to
      # disentangle from the rest of the function. We don't attempt it.
      self.v_arginit = tf.Variable(tf.ones(tf.shape(x)) * tf.constant(2.))
    return self.v_opinit + self.v_arginit + x
```

These patterns using tracing compilation directly throw an error asking
the user to put the variable's initializer in a lambda. With tf.function they
work with eager semantics either by lifting the subgraph out of the function and
using it to initialize the variable, or by initializing variables on the first
call to the function (if they weren't already initialized by something else,
e.g. a checkpoint API). The latter requires tf.conds, and is not well supported
by TF-XLA, so we only do it when necessary.

Since these patterns are relatively common in layer libraries, we expose the
wrapper in this file as `tf.function`. The defun concept in quarantine.py is a
legacy internal API.

In order to support these variable initialization patterns, tf.function defines
a variable subtype (UnliftedInitializerVariable) which collects the input
subgraph. This type of variable replaces the regular variable type on the first
tf.function trace. To exclude initializers from the function body (the `tf.ones`
ops above and associated assignment operations), tf.function traces a second
time if it sees variables on the first call.
"""
import dataclasses
import functools
import os
import threading
import types as types_lib
import warnings
import weakref
from google.protobuf import text_format as _text_format
from google.protobuf.message import DecodeError
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.function import trace_type
from tensorflow.core.function.capture import capture_container
from tensorflow.core.function.polymorphism import function_cache
from tensorflow.python.distribute.parallel_device import parallel_device
from tensorflow.python.eager import context
from tensorflow.python.eager import lift_to_graph
from tensorflow.python.eager import monitoring
from tensorflow.python.eager.polymorphic_function import attributes as attributes_lib
from tensorflow.python.eager.polymorphic_function import autograph_util
from tensorflow.python.eager.polymorphic_function import compiler_ir
from tensorflow.python.eager.polymorphic_function import eager_function_run
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.eager.polymorphic_function import tf_method_target
from tensorflow.python.eager.polymorphic_function import tracing_compilation
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import core
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.tf_export import tf_export
FREQUENT_TRACING_WARNING_MAX_CALL_HISTORY = 10
FREQUENT_TRACING_WARNING_THRESHOLD = 5
FREQUENT_TRACING_WARNING_MAX_WARNING_PER_DETECTOR = 2
_tf_function_counter = monitoring.Counter('/tensorflow/core/tf_function_counter', 'Counter for the number of tf.functions created when Eager execution is enabled.', 'jit_compile')

class _FrequentTracingDetector(object):
    """Class keeping track of how many recent calls triggered tracing."""
    __slots__ = ['_calls_per_tracings', '_call_count', '_total_warning_count']

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._calls_per_tracings = []
        self._total_warning_count = 0
        self._call_count = 0

    def called_with_tracing(self, function_name, omit_warning):
        if False:
            return 10
        "Updates the list of most recent calls' tracing information.\n\n    Warns the user when recent calls caused retracing too often.\n\n    Args:\n      function_name: the python function being traced.\n      omit_warning: If 'True', this call will not warn the user even if\n        retracing happens too often.\n    "
        self._call_count += 1
        self._calls_per_tracings.append(1)
        while self._calls_per_tracings:
            if self._call_count - self._calls_per_tracings[0] > FREQUENT_TRACING_WARNING_MAX_CALL_HISTORY:
                self._call_count -= self._calls_per_tracings.pop(0)
            else:
                break
        if omit_warning or self._total_warning_count >= FREQUENT_TRACING_WARNING_MAX_WARNING_PER_DETECTOR:
            return
        if len(self._calls_per_tracings) >= FREQUENT_TRACING_WARNING_THRESHOLD:
            self._total_warning_count += 1
            logging.warning('{} out of the last {} calls to {} triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.'.format(len(self._calls_per_tracings), self._call_count, function_name))

    def called_without_tracing(self):
        if False:
            return 10
        if not self._calls_per_tracings:
            self._calls_per_tracings = [0]
        self._calls_per_tracings[-1] += 1
        self._call_count += 1

class _FrequentTracingDetectorManager(object):
    """Class for the management of all _FrequentTracingDetector objects."""
    __slots__ = ['_detectors', '_lock']

    def __init__(self):
        if False:
            return 10
        self._detectors = weakref.WeakKeyDictionary()
        self._lock = threading.Lock()

    def _get_detector(self, key):
        if False:
            for i in range(10):
                print('nop')
        if key not in self._detectors:
            self._detectors[key] = _FrequentTracingDetector()
        return self._detectors[key]

    def called_without_tracing(self, key):
        if False:
            print('Hello World!')
        with self._lock:
            detector = self._get_detector(key)
            detector.called_without_tracing()

    def called_with_tracing(self, key, function_name, omit_warning):
        if False:
            while True:
                i = 10
        with self._lock:
            detector = self._get_detector(key)
            detector.called_with_tracing(function_name, omit_warning)
_frequent_tracing_detector_manager = _FrequentTracingDetectorManager()

class UnliftedInitializerVariable(resource_variable_ops.UninitializedVariable):
    """Variable which does not lift its initializer out of function context.

  Instances of this variable, when created, build a graph which runs their
  initializer inside a tf.cond(is_initialized) block.

  This can only be created during tracing compilation called from
  (eventually) eager mode. That is, non-function-building graphs are not
  supported.
  """

    def __init__(self, initial_value=None, trainable=None, caching_device=None, name=None, dtype=None, constraint=None, add_initializers_to=None, synchronization=None, aggregation=None, shape=None, **unused_kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Creates a variable.\n\n    Args:\n      initial_value: A `Tensor`, or Python object convertible to a `Tensor`,\n        which is the initial value for the Variable. The initial value must have\n        a shape specified unless `validate_shape` is set to False. Can also be a\n        callable with no argument that returns the initial value when called.\n        (Note that initializer functions from init_ops.py must first be bound to\n        a shape before being used here.)\n      trainable: If `True`, GradientTapes automatically watch uses of this\n        Variable.\n      caching_device: Optional device string or function describing where the\n        Variable should be cached for reading.  Defaults to the Variable's\n        device.  If not `None`, caches on another device.  Typical use is to\n        cache on the device where the Ops using the Variable reside, to\n        deduplicate copying through `Switch` and other conditional statements.\n      name: Optional name for the variable. Defaults to `'Variable'` and gets\n        uniquified automatically.\n      dtype: If set, initial_value will be converted to the given type. If None,\n        either the datatype will be kept (if initial_value is a Tensor) or\n        float32 will be used (if it is a Python object convertible to a Tensor).\n      constraint: An optional projection function to be applied to the variable\n        after being updated by an `Optimizer` (e.g. used to implement norm\n        constraints or value constraints for layer weights). The function must\n        take as input the unprojected Tensor representing the value of the\n        variable and return the Tensor for the projected value (which must have\n        the same shape). Constraints are not safe to use when doing asynchronous\n        distributed training.\n      add_initializers_to: if not None and not in legacy graph mode, the\n        initializer tensor will be added to this map in addition to adding the\n        assignment to the function.\n      synchronization: Indicates when a distributed variable will be aggregated.\n        Accepted values are constants defined in the class\n        `tf.VariableSynchronization`. By default the synchronization is set to\n        `AUTO` and the current `DistributionStrategy` chooses when to\n        synchronize.\n      aggregation: Indicates how a distributed variable will be aggregated.\n        Accepted values are constants defined in the class\n        `tf.VariableAggregation`.\n      shape: (optional) The shape of this variable. If None, the shape of\n        `initial_value` will be used. When setting this argument to\n        `tf.TensorShape(None)` (representing an unspecified shape), the variable\n        can be assigned with values of different shapes.\n\n    Raises:\n      ValueError: If the initial value is not specified, or does not have a\n        shape and `validate_shape` is `True`.\n      RuntimeError: If called outside of a function definition.\n    "
        with ops.init_scope():
            self._in_graph_mode = not context.executing_eagerly()
        if not ops.inside_function():
            resource_variable_ops.ResourceVariable.__init__(self, initial_value=initial_value, trainable=trainable, caching_device=caching_device, name=name, dtype=dtype, constraint=constraint)
            return
        if initial_value is None:
            raise ValueError('`initial_value` must be a Tensor or a Python object convertible to a Tensor. Got None.')
        init_from_fn = callable(initial_value)
        if constraint is not None and (not callable(constraint)):
            raise ValueError(f'`constraint` with type {type(constraint)} must be a callable.')
        with ops.name_scope(name, 'Variable', [] if init_from_fn else [initial_value]) as scope_name:
            with ops.name_scope('Initializer'):
                if init_from_fn:
                    initial_value = initial_value()
                if isinstance(initial_value, trackable.CheckpointInitialValue):
                    self._maybe_initialize_trackable()
                    self._update_uid = initial_value.checkpoint_position.restore_uid
                    initial_value = initial_value.wrapped_value
                initial_value = ops.convert_to_tensor(initial_value, name='initial_value', dtype=dtype)
            assert initial_value is not None
            if shape is None:
                shape = initial_value.shape
        super().__init__(trainable=trainable, caching_device=caching_device, name=name, shape=shape, dtype=initial_value.dtype, constraint=constraint, synchronization=synchronization, aggregation=aggregation, extra_handle_data=initial_value, **unused_kwargs)
        with ops.name_scope(scope_name):
            if self._in_graph_mode:
                with ops.init_scope():
                    outer_graph = ops.get_default_graph()
                func_graph = ops.get_default_graph()
                function_placeholders = func_graph.inputs + func_graph.internal_captures
                placeholder_ops = set([tensor.op for tensor in function_placeholders])
                lifted_initializer = lift_to_graph.lift_to_graph([initial_value], outer_graph, disallowed_placeholders=placeholder_ops)[initial_value]
                with ops.init_scope():
                    self._initial_value = lifted_initializer
                    with ops.name_scope('IsInitialized'):
                        self._is_initialized_op = resource_variable_ops.var_is_initialized_op(self._handle)
                    if initial_value is not None:
                        with ops.name_scope('Assign') as n, ops.colocate_with(self._handle):
                            self._initializer_op = resource_variable_ops.assign_variable_op(self._handle, lifted_initializer, name=n)
            elif context.executing_eagerly():
                with ops.name_scope('Assign') as n, ops.colocate_with(self._handle):
                    resource_variable_ops.assign_variable_op(self._handle, initial_value, name=n)
            else:
                if add_initializers_to is not None:
                    add_initializers_to.append((self, initial_value))

                def assign_fn():
                    if False:
                        print('Hello World!')
                    with ops.name_scope('Assign') as n, ops.colocate_with(self._handle):
                        resource_variable_ops.assign_variable_op(self._handle, initial_value, name=n)
                    return ops.convert_to_tensor(1)

                def not_assign_fn():
                    if False:
                        print('Hello World!')
                    return ops.convert_to_tensor(0)
                graph = ops.get_default_graph()
                graph.capture(self._handle, shape=())
                cond.cond(resource_variable_ops.var_is_initialized_op(self._handle), not_assign_fn, assign_fn)
JIT_COMPILE_FUNCTIONS = os.getenv('TF_FUNCTION_JIT_COMPILE_DEFAULT', 'false').lower() in ('true', '1')

def _evaluate_var_is_initialized(variables):
    if False:
        print('Hello World!')
    'Compute booleans indicating whether each variable is initialized.'
    with ops.init_scope():
        var_is_initialized = []
        for v in variables:
            var_is_initialized.append(resource_variable_ops.var_is_initialized_op(v.handle))
        try:
            return array_ops_stack.stack(var_is_initialized).numpy()
        except errors.UnimplementedError:
            for (index, v) in enumerate(variables):
                try:
                    numpy_value = var_is_initialized[index].numpy()
                except errors.UnimplementedError:
                    components = parallel_device.unpack(var_is_initialized[index])
                    with ops.device(None):
                        components = array_ops_stack.stack(components)
                        all_initialized = math_ops.reduce_all(components).numpy()
                        any_initialized = math_ops.reduce_any(components).numpy()
                    if all_initialized != any_initialized:
                        raise NotImplementedError(f"Some but not all components of a parallel variable {v!r} were initialized between their creation in a tf.function and the function's trace having completed. This is not supported; consider initializing either all or none of the components, or moving initialization out of the function.")
                    numpy_value = all_initialized
                var_is_initialized[index] = numpy_value
    return var_is_initialized

class OptionalXlaContext:
    """Wrapper for XLA context optionally applied under a context manager."""

    def __init__(self, is_compiled):
        if False:
            for i in range(10):
                print('nop')
        wrap = is_compiled and (not control_flow_util.GraphOrParentsInXlaContext(ops.get_default_graph()))
        self.xla_context = control_flow_ops.XLAControlFlowContext() if wrap else None

    def __enter__(self):
        if False:
            return 10
        if self.xla_context:
            self.xla_context.Enter()

    def __exit__(self, t, value, traceback):
        if False:
            print('Hello World!')
        if self.xla_context:
            self.xla_context.Exit()

@tf_export('__internal__.function.Function', v1=[])
class Function(core.PolymorphicFunction, trackable.Trackable):
    """A `tf.types.experimental.PolymorphicFunction` created by `tf.function`.

  Currently, individual methods/attributes under this class are not guaranteed
  by the TF API contract, and are subject to future changes.

  (Previously also known as `tf.types.experimental.GenericFunction`)
  """

    def __init__(self, python_function, name, input_signature=None, autograph=True, jit_compile=None, reduce_retracing=False, experimental_implements=None, experimental_autograph_options=None, experimental_attributes=None):
        if False:
            for i in range(10):
                print('nop')
        "Initializes a `Function`.\n\n    Args:\n      python_function: the function to be wrapped.\n      name: the name given to it.\n      input_signature: See the documentation for `tf.function`.\n      autograph: See the documentation for `tf.function`.\n      jit_compile: See the documentation for `tf.function`.\n      reduce_retracing: See the documentation for `tf.function`.\n      experimental_implements: See the documentation for `tf.function`.\n      experimental_autograph_options: See the documentation for `tf.function`.\n      experimental_attributes: See the documentation for `tf.function`.\n\n    Raises:\n      ValueError: if `input_signature` is not None and the `python_function`'s\n        argspec has keyword arguments.\n    "
        self._lock = threading.RLock()
        self._python_function = python_function
        (self._function_type, self._default_values) = function_type_utils.make_function_type(python_function, input_signature)
        self._function_cache = function_cache.FunctionCache()
        self._function_captures = capture_container.FunctionCaptures()
        self._attributes = {}
        if experimental_implements is not None:
            self._attributes = self._create_implements_attribute(experimental_implements)
        if experimental_attributes is not None:
            self._attributes.update(experimental_attributes)
        for attribute in self._attributes:
            if attribute not in attributes_lib.POLYMORPHIC_FUNCTION_ALLOWLIST:
                raise ValueError(f'`{attribute} is not supported by tf.function as an attribute.')
        self._is_pure = self._attributes and attributes_lib.IMPLEMENTS in self._attributes
        self._shared_rendezvous = None
        self._autograph = autograph
        self._experimental_autograph_options = experimental_autograph_options
        self._reduce_retracing = reduce_retracing
        self._jit_compile = jit_compile
        self._created_variables = None
        self._variable_creation_config = None
        self._no_variable_creation_config = None
        self._descriptor_cache = weakref.WeakKeyDictionary()
        self._name = name
        self._key_for_call_stats = self._get_key_for_call_stats()
        self._omit_frequent_tracing_warning = False
        ops._tf_function_api_gauge.get_cell().set(True)

    @property
    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return self._name

    def __getstate__(self):
        if False:
            for i in range(10):
                print('nop')
        'Custom pickling, to omit unpickleable objects.'
        result = self.__dict__.copy()
        del result['_lock']
        del result['_descriptor_cache']
        del result['_key_for_call_stats']
        return result

    def __setstate__(self, state):
        if False:
            for i in range(10):
                print('nop')
        'Restore from pickled state.'
        self.__dict__ = state
        self._lock = threading.RLock()
        self._descriptor_cache = weakref.WeakKeyDictionary()
        self._key_for_call_stats = self._get_key_for_call_stats()

    def _get_key_for_call_stats(self):
        if False:
            return 10
        'Returns key instance to track call stats and retracings.\n\n    The key instance a best-effort to preserve global consistency.\n    '
        target_function = self._python_function
        while hasattr(target_function, '__wrapped__'):
            target_function = target_function.__wrapped__
        if hasattr(target_function, '__func__'):
            target_function = target_function.__func__
        if hasattr(target_function, '__code__'):
            return target_function.__code__
        return self._python_function

    def _generate_scoped_tracing_options(self, scope, scope_type):
        if False:
            return 10
        'Creates TracingOptions for variable creator scopes.'
        weak_wrapped_fn = None
        compile_with_xla = self._jit_compile

        def wrapped_fn(*args, **kwds):
            if False:
                while True:
                    i = 10
            'Wraps `self._python_function` in a variable creator scope.'
            default_graph = ops.get_default_graph()
            with default_graph._variable_creator_scope(scope, priority=50):
                with OptionalXlaContext(compile_with_xla):
                    out = weak_wrapped_fn().__wrapped__(*args, **kwds)
                return out
        weak_wrapped_fn = weakref.ref(wrapped_fn)
        return self._generate_tracing_options(tf_decorator.make_decorator(self._python_function, wrapped_fn), scope_type)

    def _create_implements_attribute(self, implements_arg):
        if False:
            print('Hello World!')
        'Creates the attribute value corresponding to attribute_lib.IMPLEMENTS.'
        attributes = {}
        if isinstance(implements_arg, str):
            try:
                attr_value = attr_value_pb2.AttrValue()
                nameattrlist = attr_value_pb2.NameAttrList()
                _text_format.Merge(implements_arg, nameattrlist)
                attr_value.func.CopyFrom(nameattrlist)
                attributes[attributes_lib.IMPLEMENTS] = attr_value
            except (_text_format.ParseError, DecodeError):
                attributes[attributes_lib.IMPLEMENTS] = implements_arg
        return attributes

    def _generate_tracing_options(self, fn, scope_type):
        if False:
            return 10
        'Return a TracingOptions catered to the input function.'
        attributes = self._attributes.copy()
        share = self._shared_rendezvous
        if share is not None:
            attributes[attributes_lib.SHARED_RENDEZVOUS] = share
        if self._jit_compile is not None:
            attributes[attributes_lib.XLA_COMPILE] = bool(self._jit_compile)
            if self._jit_compile:
                attributes[attributes_lib.NO_INLINE] = True
        if self._autograph:
            fn = autograph_util.py_func_from_autograph(fn, self._experimental_autograph_options)
        return tracing_compilation.TracingOptions(fn, self._name, polymorphic_type=self._function_type, default_values=self._default_values, scope_type=scope_type, attributes=attributes, autograph=self._autograph, reduce_retracing=self._reduce_retracing, autograph_options=self._experimental_autograph_options, function_cache=self._function_cache, function_captures=self._function_captures, lock=self._lock)

    def _initialize(self, args, kwds, add_initializers_to=None):
        if False:
            i = 10
            return i + 15
        "Initializes, on the first call.\n\n    Creates two `Function`s, one that will allow creation of variables\n    and one that won't.\n\n    Additionally runs a trace for the `Function` that allows creation\n    of variables.\n\n    Args:\n      args: Arguments to the underlying python callable.\n      kwds: Keyword arguments to the python callable.\n      add_initializers_to: Where to collect variable initializers, if not None.\n    "
        created_variables = []

        def variable_capturing_scope(next_creator, **kwds):
            if False:
                return 10
            'Creates UnliftedInitializerVariables and saves references to them.'
            enable_variable_lifting = kwds.get('experimental_enable_variable_lifting')
            if enable_variable_lifting is None:
                enable_variable_lifting = True
            if not enable_variable_lifting:
                return next_creator(**kwds)
            v = UnliftedInitializerVariable(add_initializers_to=add_initializers_to, **kwds)
            created_variables.append(weakref.ref(v))
            return v
        self._created_variables = created_variables
        self._variable_creation_config = self._generate_scoped_tracing_options(variable_capturing_scope, tracing_compilation.ScopeType.VARIABLE_CREATION)
        self._concrete_variable_creation_fn = tracing_compilation.trace_function(args, kwds, self._variable_creation_config)

        def invalid_creator_scope(*unused_args, **unused_kwds):
            if False:
                return 10
            'Disables variable creation.'
            raise ValueError('tf.function only supports singleton tf.Variables created on the first call. Make sure the tf.Variable is only created once or created outside tf.function. See https://www.tensorflow.org/guide/function#creating_tfvariables for more information.')
        self._no_variable_creation_config = self._generate_scoped_tracing_options(invalid_creator_scope, tracing_compilation.ScopeType.NO_VARIABLE_CREATION)

    def _clone(self, python_function):
        if False:
            while True:
                i = 10
        'Clone the function with different python function.'
        f = Function(python_function=self._python_function if python_function is None else python_function, name=self._name, input_signature=self.input_signature, autograph=self._autograph, jit_compile=self._jit_compile, reduce_retracing=self._reduce_retracing, experimental_attributes=self._attributes, experimental_autograph_options=self._experimental_autograph_options)
        if self._shared_rendezvous:
            f._shared_rendezvous = self._shared_rendezvous
        return f

    def _decorate(self, decorator):
        if False:
            return 10
        'Allows the captured Python function to be decorated in place.\n\n    This method is only safe to call when the Function has not been called by a\n    user. It makes sense to use this method to push a decorator into the\n    function rather than wrapping the function in the decorator.\n\n    We use this in tf.Module to allow user annotated `tf.functions` to remain as\n    `Function` objects but still automatically enter the Module name_scope\n    when they are evaluated like all other methods.\n\n    Args:\n      decorator: A callable accepting a single argument which is the function\n        to decorate and returning a callable result.\n\n    Raises:\n      ValueError: If the function has been called a ValueError is raised.\n    '
        if self._variable_creation_config is not None or self._no_variable_creation_config is not None:
            raise ValueError('Functions cannot be decorated after they have been traced.')
        self._python_function = decorator(self._python_function)
        (self._function_type, self._default_values) = function_type_utils.make_function_type(self._python_function, self.input_signature)

    def _get_tracing_count(self):
        if False:
            return 10
        return self.experimental_get_tracing_count()

    def experimental_get_tracing_count(self):
        if False:
            print('Hello World!')
        'Returns the number of times the function has been traced.\n\n    For more information on when a function is traced and when it is\n    traced multiple times see https://www.tensorflow.org/guide/function.\n    Example:\n\n    >>> @tf.function\n    ... def double(a):\n    ...   return a + a\n    >>> double(tf.constant(1))\n    >>> double(tf.constant(2))\n    >>> double.experimental_get_tracing_count()\n    1\n    >>> double(tf.constant("a"))\n    >>> double.experimental_get_tracing_count()\n    2\n\n\n    The first time experimental_get_tracing_count is called\n    it returns 1, as the function is traced the first\n    time it is called, and the second time the same graph is used\n    since we\'re calling it with a parameter of the same type.\n\n    The second time experimental_get_tracing_count is called\n    it returns 2, as we called double with a\n    different argument type, and so it was traced again.\n\n    '
        return len(self._function_cache)

    @property
    def _run_functions_eagerly(self):
        if False:
            print('Hello World!')
        return eager_function_run.RUN_FUNCTIONS_EAGERLY

    @traceback_utils.filter_traceback
    def __call__(self, *args, **kwds):
        if False:
            for i in range(10):
                print('nop')
        if self._run_functions_eagerly:
            with trace.Trace(self._name, tf_function_call='eager'):
                return self._python_function(*args, **kwds)
        if self._created_variables is None:
            compiled = bool(self._jit_compile and (not control_flow_util.GraphOrParentsInXlaContext(ops.get_default_graph())))
            if ops.executing_eagerly_outside_functions() and (context.executing_eagerly() or compiled):
                _tf_function_counter.get_cell(str(int(compiled))).increase_by(1)
        tracing_count = self.experimental_get_tracing_count()
        with trace.Trace(self._name) as tm:
            compiler = 'xla' if self._jit_compile else 'nonXla'
            with OptionalXlaContext(self._jit_compile):
                result = self._call(*args, **kwds)
            new_tracing_count = self.experimental_get_tracing_count()
            without_tracing = tracing_count == new_tracing_count
            execution_mode = 'notTraced' if without_tracing else 'traced'
            tm.set_metadata(tf_function_call=execution_mode + '-' + compiler, tracing_count=new_tracing_count)
        if context.executing_eagerly():
            if without_tracing:
                _frequent_tracing_detector_manager.called_without_tracing(self._key_for_call_stats)
            else:
                _frequent_tracing_detector_manager.called_with_tracing(self._key_for_call_stats, self._python_function, self._omit_frequent_tracing_warning)
        return result

    def _call(self, *args, **kwds):
        if False:
            i = 10
            return i + 15
        'Calls the graph function.'
        self._lock.acquire()
        bound_args = function_type_utils.canonicalize_function_inputs(args, kwds, self._function_type, self._default_values, self._is_pure)
        (args, kwds) = (bound_args.args, bound_args.kwargs)
        if self._created_variables:
            self._lock.release()
            return tracing_compilation.call_function(args, kwds, self._no_variable_creation_config)
        elif self._variable_creation_config is not None:
            self._lock.release()
            results = tracing_compilation.call_function(args, kwds, self._variable_creation_config)
            if self._created_variables:
                raise ValueError('Creating variables on a non-first call to a function decorated with tf.function.')
            return results
        try:
            initializers = []
            self._initialize(args, kwds, add_initializers_to=initializers)
        finally:
            self._lock.release()
        if self._created_variables:
            try:
                self._initialize_uninitialized_variables(initializers)
            except lift_to_graph.UnliftableError:
                pass
            else:
                return tracing_compilation.call_function(args, kwds, self._no_variable_creation_config)
        else:
            bound_args = self._concrete_variable_creation_fn.function_type.bind(*args, **kwds)
            filtered_flat_args = self._concrete_variable_creation_fn.function_type.unpack_inputs(bound_args)
            return self._concrete_variable_creation_fn._call_flat(filtered_flat_args, self._concrete_variable_creation_fn.captured_inputs)

        def fn_with_cond(inner_args, inner_kwds):
            if False:
                while True:
                    i = 10
            "Conditionally runs initialization if it's needed."
            condition = True
            for (v, _) in initializers:
                condition = math_ops.logical_and(condition, resource_variable_ops.var_is_initialized_op(v.handle))
            return cond.cond(condition, lambda : tracing_compilation.call_function(inner_args, inner_kwds, self._no_variable_creation_config), lambda : self._concrete_variable_creation_fn(*inner_args, **inner_kwds))
        if self._jit_compile:
            raise errors.UnimplementedError(None, None, 'We failed to lift variable creations out of this tf.function, so this tf.function cannot be run on XLA. A possible workaround is to move variable creation outside of the XLA compiled function.')
        (canon_args, canon_kwds) = (bound_args.args, bound_args.kwargs)
        options = tracing_compilation.TracingOptions(fn_with_cond, 'fn_with_cond')
        return tracing_compilation.call_function((canon_args, canon_kwds), {}, options)

    def experimental_get_compiler_ir(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        context.ensure_initialized()
        if not self._jit_compile:
            raise ValueError("Compiler IR can only be returned for functions marked with 'jit_compile=True'")
        is_tensor_spec = lambda x: isinstance(x, tensor_spec.TensorSpec)

        def _check_inputs(args, kwargs):
            if False:
                while True:
                    i = 10
            all_inputs = list(args) + list(kwargs.values())
            if not all_inputs:
                return
            if any(map(is_tensor_spec, all_inputs)) and any(map(lambda x: not is_tensor_spec(x), all_inputs)):
                raise ValueError('experimental_get_compiler_ir supports either (1) all inputs are TensorSpec  or (2) all inputs are tf.Tensor/python variables')
        _check_inputs(args, kwargs)
        if len(args) + len(kwargs.values()) > 0 and all(map(is_tensor_spec, args)) and all(map(is_tensor_spec, kwargs.values())):
            concrete_fn = self.get_concrete_function(*args, **kwargs)
            return compiler_ir.from_concrete_function(concrete_fn)
        concrete_fn = self.get_concrete_function(*args, **kwargs)
        fn_name = concrete_fn.name
        bound_args = function_type_utils.canonicalize_function_inputs(args, kwargs, concrete_fn.function_type)
        filtered_flat_args = concrete_fn.function_type.unpack_inputs(bound_args)

        def compiler_ir_generator(stage='hlo', device_name=None, platform_name=None):
            if False:
                i = 10
                return i + 15
            'Gets the compiler IR bytes.\n\n      Args:\n        stage: The exported stage for the given function.\n        device_name: The name of the device with the form as\n          "/job:localhost/replica:0/task:0/device:CPU:0", "/device:TPU:0" etc.\n          When this is used, actual device is used for getting the compiler IR.\n        platform_name: The name of the platform, e.g. "TPU". When this is used,\n          no actual device is needed but the compiler IR is obtained as if using\n          that device. The scenarios supported are more limited.\n\n      Returns:\n        The compiler IR bytes.\n      '
            if device_name is not None:
                if platform_name is not None:
                    raise ValueError('device_name and platform_name cannot be provided at the same time.')
                warnings.warn('device_name is being deprecated. Use platform_name.')
            device_name = compiler_ir.maybe_get_device_name(device_name)
            res_bytes = context.context().get_compiler_ir(device_name=device_name, platform_name=platform_name, function_name=fn_name, flat_args=list(filtered_flat_args), captured_inputs=concrete_fn.captured_inputs, stage=stage)
            if stage in ('hlo_serialized', 'optimized_hlo_serialized', 'optimized_hlo_proto_serialized'):
                return res_bytes
            else:
                return res_bytes.decode('utf-8')
        return compiler_ir_generator

    @property
    def python_function(self):
        if False:
            while True:
                i = 10
        'The python function wrapped in this tf.function.'
        return self._python_function

    @property
    def input_signature(self):
        if False:
            print('Hello World!')
        return function_type_utils.to_input_signature(self._function_type)

    @property
    def function_spec(self):
        if False:
            i = 10
            return i + 15
        return function_type_utils.FunctionSpec(self._function_type, self._default_values, False, self._name, self._jit_compile)

    @property
    def function_type(self):
        if False:
            print('Hello World!')
        return self._function_type

    def pretty_printed_concrete_signatures(self, verbose=True):
        if False:
            print('Hello World!')
        joiner = '\n\n' if verbose else '\n'
        return joiner.join([c.pretty_printed_signature(verbose=verbose) for c in self._list_all_concrete_functions()])

    def _initialize_uninitialized_variables(self, initializers):
        if False:
            print('Hello World!')
        'Make and call a `ConcreteFunction` which initializes variables.'
        if not initializers:
            return
        var_is_initialized = _evaluate_var_is_initialized([v for (v, _) in initializers])

        def initialize_variables():
            if False:
                return 10
            op_map = object_identity.ObjectIdentityDictionary()
            inits = []
            for ((v, init), is_initialized) in zip(initializers, var_is_initialized):
                with ops.init_scope():
                    if is_initialized:
                        continue
                inits.append(init)
            if inits:
                op_map = lift_to_graph.lift_to_graph(inits, ops.get_default_graph(), op_map=op_map)
            for ((v, init), is_initialized) in zip(initializers, var_is_initialized):
                with ops.init_scope():
                    if is_initialized:
                        continue
                v.assign(op_map[init], read_value=False)
        with ops.init_scope():
            options = tracing_compilation.TracingOptions(initialize_variables, 'initialize_variables', autograph=False)
            return tracing_compilation.call_function(tracing_options=options)

    def get_initialization_function(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        "Returns a `ConcreteFunction` which initializes this function's variables.\n\n    Requires that this function hasn't been accessed yet through either calling\n    it or calling get_concrete_function. Fails if we cannot build an initializer\n    function which does not depend on the concrete values of the inputs to this\n    function.\n\n    Note that running this function will overwrite any values currently assigned\n    to variables, for example restores from a checkpoint.\n\n    Args:\n      *args: arguments to the underlying python callable.\n      **kwargs: keyword arguments to the python callable.\n\n    Returns:\n      A `ConcreteFunction` object which initializes the variables of this\n      function.\n\n    Raises:\n      RuntimeError: if called after the variables have been initialized.\n    "
        with self._lock:
            if self._variable_creation_config is not None:
                raise RuntimeError('get_initialization_function cannot be called after the function has been used')
            initializers = []
            self._initialize(args, kwargs, add_initializers_to=initializers)

        def initialize_variables():
            if False:
                while True:
                    i = 10
            for (v, init) in initializers:
                v.assign(lift_to_graph.lift_to_graph([init], ops.get_default_graph())[init], read_value=False)
        options = tracing_compilation.TracingOptions(initialize_variables, 'initialize_variables')
        return tracing_compilation.trace_function(tracing_options=options)

    def _list_all_concrete_functions(self):
        if False:
            while True:
                i = 10
        'Returns all concrete functions.'
        if self.input_signature is not None:
            self.get_concrete_function()
        return self._function_cache.values()

    def _list_all_concrete_functions_for_serialization(self):
        if False:
            i = 10
            return i + 15
        'Returns all concrete functions for serialization.\n\n    Returns:\n      A list of instances of `ConcreteFunction`.\n    '
        seen_signatures = []
        if self.input_signature is not None:
            seen_signatures.append((self.input_signature, {}))
        else:
            concrete_functions = self._list_all_concrete_functions()
            for concrete_function in concrete_functions:
                signature = concrete_function.structured_input_signature
                flattened = nest.flatten(signature)
                if any((isinstance(arg, func_graph_module.UnknownArgument) for arg in flattened)):
                    logging.info('Unsupported signature for serialization: %s.', signature)
                    continue
                equal_to_signature = functools.partial(function_type_utils.is_same_structure, signature, check_values=True)
                if not any((equal_to_signature(s) for s in seen_signatures)):
                    seen_signatures.append(signature)
        concrete_functions = []
        for (args, kwargs) in seen_signatures:
            concrete_functions.append(self.get_concrete_function(*args, **kwargs))
        return concrete_functions

    def _trackable_children(self, save_type='checkpoint', **kwargs):
        if False:
            return 10
        'For implementing `Trackable`.'
        if save_type == 'checkpoint':
            return {}
        return {f'trace_{n}': fn for (n, fn) in enumerate(self._list_all_concrete_functions_for_serialization())}

    def _deserialization_dependencies(self, children):
        if False:
            while True:
                i = 10
        'Returns concrete functions which must be loaded before this object.'
        return children

    def _get_concrete_function_garbage_collected(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        "Returns a `ConcreteFunction` specialized to inputs and execution context.\n\n    Unlike `get_concrete_function(...)`, the graph will be deleted when the\n    returned function is deleted.  It's useful to avoid creating a reference\n    cycle when you know for sure that the graph will be no longer used without\n    the returned function.\n\n    Args:\n      *args: inputs to specialize on.\n      **kwargs: inputs to specialize on.\n\n    Returns:\n      A TensorFlow function which takes exactly one `tf.Tensor` per argument.\n\n    Raises:\n      ValueError: if this object has not yet been called on concrete values.\n    "
        with self._lock:
            if self._variable_creation_config is None:
                initializers = []
                self._initialize(args, kwargs, add_initializers_to=initializers)
                self._initialize_uninitialized_variables(initializers)
        if self._created_variables:
            return tracing_compilation.trace_function(args, kwargs, dataclasses.replace(self._no_variable_creation_config, bind_graph_to_function=True))
        elif self._variable_creation_config is not None:
            concrete = tracing_compilation.trace_function(args, kwargs, dataclasses.replace(self._variable_creation_config, bind_graph_to_function=True))
            if self._created_variables:
                raise ValueError('Creating variables on a non-first call to a function decorated with tf.function.')
            return concrete

    def get_concrete_function(self, *args, **kwargs):
        if False:
            return 10
        concrete = self._get_concrete_function_garbage_collected(*args, **kwargs)
        concrete._garbage_collector.release()
        return concrete

    def __tf_tracing_type__(self, _):
        if False:
            for i in range(10):
                print('nop')
        return trace_type.Weakref(weakref.ref(self))

    def __get__(self, instance, owner):
        if False:
            for i in range(10):
                print('nop')
        'Makes it possible to decorate instance methods.'
        del owner
        if isinstance(instance, composite_tensor.CompositeTensor) and instance._type_spec is not None:
            return types_lib.MethodType(self, instance)
        if instance not in self._descriptor_cache:
            if instance is None:
                return self
            self._descriptor_cache[instance] = class_method_to_instance_method(self, instance)
        return self._descriptor_cache[instance]

@tf_export('function')
@deprecation.deprecated_args(None, 'experimental_compile is deprecated, use jit_compile instead', 'experimental_compile')
@deprecation.deprecated_args(None, 'experimental_relax_shapes is deprecated, use reduce_retracing instead', 'experimental_relax_shapes')
@deprecation.deprecated_args(None, 'experimental_follow_type_hints is deprecated', 'experimental_follow_type_hints')
def function(func=None, input_signature=None, autograph=True, jit_compile=None, reduce_retracing=False, experimental_implements=None, experimental_autograph_options=None, experimental_attributes=None, experimental_relax_shapes=None, experimental_compile=None, experimental_follow_type_hints=None) -> core.PolymorphicFunction:
    if False:
        print('Hello World!')
    'Compiles a function into a callable TensorFlow graph.\n\n  `tf.function` constructs a `tf.types.experimental.PolymorphicFunction` that\n  executes a TensorFlow graph (`tf.Graph`) created by trace-compiling the\n  TensorFlow operations in `func`. More information on the topic can be found\n  in [Introduction to Graphs and tf.function]\n  (https://www.tensorflow.org/guide/intro_to_graphs).\n\n  See [Better Performance with tf.function]\n  (https://www.tensorflow.org/guide/function) for tips on performance and\n  known limitations.\n\n  Example usage:\n\n  >>> @tf.function\n  ... def f(x, y):\n  ...   return x ** 2 + y\n  >>> x = tf.constant([2, 3])\n  >>> y = tf.constant([3, -2])\n  >>> f(x, y)\n  <tf.Tensor: ... numpy=array([7, 7], ...)>\n\n  The trace-compilation allows non-TensorFlow operations to execute, but under\n  special conditions. In general, only TensorFlow operations are guaranteed to\n  run and create fresh results whenever the `PolymorphicFunction` is called.\n\n  ## Features\n\n  `func` may use data-dependent Python control flow statements, including `if`,\n  `for`, `while` `break`, `continue` and `return`:\n\n  >>> @tf.function\n  ... def f(x):\n  ...   if tf.reduce_sum(x) > 0:\n  ...     return x * x\n  ...   else:\n  ...     return -x // 2\n  >>> f(tf.constant(-2))\n  <tf.Tensor: ... numpy=1>\n\n  `func`\'s closure may include `tf.Tensor` and `tf.Variable` objects:\n\n  >>> @tf.function\n  ... def f():\n  ...   return x ** 2 + y\n  >>> x = tf.constant([-2, -3])\n  >>> y = tf.Variable([3, -2])\n  >>> f()\n  <tf.Tensor: ... numpy=array([7, 7], ...)>\n\n  `func` may also use ops with side effects, such as `tf.print`, `tf.Variable`\n  and others:\n\n  >>> v = tf.Variable(1)\n  >>> @tf.function\n  ... def f(x):\n  ...   for i in tf.range(x):\n  ...     v.assign_add(i)\n  >>> f(3)\n  >>> v\n  <tf.Variable ... numpy=4>\n\n  Important: Any Python side-effects (appending to a list, printing with\n  `print`, etc) will only happen once, when `func` is traced. To have\n  side-effects executed into your `tf.function` they need to be written\n  as TF ops:\n\n  >>> l = []\n  >>> @tf.function\n  ... def f(x):\n  ...   for i in x:\n  ...     l.append(i + 1)    # Caution! Will only happen once when tracing\n  >>> f(tf.constant([1, 2, 3]))\n  >>> l\n  [<tf.Tensor ...>]\n\n  Instead, use TensorFlow collections like `tf.TensorArray`:\n\n  >>> @tf.function\n  ... def f(x):\n  ...   ta = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)\n  ...   for i in range(len(x)):\n  ...     ta = ta.write(i, x[i] + 1)\n  ...   return ta.stack()\n  >>> f(tf.constant([1, 2, 3]))\n  <tf.Tensor: ..., numpy=array([2, 3, 4], ...)>\n\n  ## `tf.function` creates polymorphic callables\n\n  Internally, `tf.types.experimental.PolymorphicFunction` may contain multiple\n  `tf.types.experimental.ConcreteFunction`s, each specialized to arguments with\n  different data types or shapes, since TensorFlow can perform more\n  optimizations on graphs of specific shapes, dtypes and values of constant\n  arguments. `tf.function` treats any pure Python values as opaque objects (best\n  thought of as compile-time constants), and builds a separate `tf.Graph` for\n  each set of Python arguments that it encounters.\n  For more information, see the\n  [tf.function guide](https://www.tensorflow.org/guide/function#rules_of_tracing)\n\n  Executing a `PolymorphicFunction` will select and execute the appropriate\n  `ConcreteFunction` based on the argument types and values.\n\n  To obtain an individual `ConcreteFunction`, use the\n  `PolymorphicFunction.get_concrete_function` method. It can be called with the\n  same arguments as `func` and returns a\n  `tf.types.experimental.ConcreteFunction`. `ConcreteFunction`s are backed by a\n  single `tf.Graph`:\n\n  >>> @tf.function\n  ... def f(x):\n  ...   return x + 1\n  >>> isinstance(f.get_concrete_function(1).graph, tf.Graph)\n  True\n\n  `ConcreteFunction`s can be executed just like `PolymorphicFunction`s, but their\n  input is resticted to the types to which they\'re specialized.\n\n  ## Retracing\n\n  `ConcreteFunctions` are built (traced) on the fly, as the `PolymorphicFunction` is\n  called with new TensorFlow types or shapes, or with new Python values as\n  arguments. When `PolymorphicFunction` builds a new trace, it is said that `func`\n  is retraced. Retracing is a frequent performance concern for `tf.function` as\n  it can be considerably slower than executing a graph that\'s already been\n  traced. It is ideal to minimize the amount of retracing in your code.\n\n  Caution: Passing python scalars or lists as arguments to `tf.function` will\n  usually retrace. To avoid this, pass numeric arguments as Tensors whenever\n  possible:\n\n  >>> @tf.function\n  ... def f(x):\n  ...   return tf.abs(x)\n  >>> f1 = f.get_concrete_function(1)\n  >>> f2 = f.get_concrete_function(2)  # Slow - compiles new graph\n  >>> f1 is f2\n  False\n  >>> f1 = f.get_concrete_function(tf.constant(1))\n  >>> f2 = f.get_concrete_function(tf.constant(2))  # Fast - reuses f1\n  >>> f1 is f2\n  True\n\n  Python numerical arguments should only be used when they take few distinct\n  values, such as hyperparameters like the number of layers in a neural network.\n\n  ## Input signatures\n\n  For Tensor arguments, `PolymorphicFunction`creates a new `ConcreteFunction` for\n  every unique set of input shapes and datatypes. The example below creates two\n  separate `ConcreteFunction`s, each specialized to a different shape:\n\n  >>> @tf.function\n  ... def f(x):\n  ...   return x + 1\n  >>> vector = tf.constant([1.0, 1.0])\n  >>> matrix = tf.constant([[3.0]])\n  >>> f.get_concrete_function(vector) is f.get_concrete_function(matrix)\n  False\n\n  An "input signature" can be optionally provided to `tf.function` to control\n  this process. The input signature specifies the shape and type of each\n  Tensor argument to the function using a `tf.TensorSpec` object. More general\n  shapes can be used. This ensures only one `ConcreteFunction` is created, and\n  restricts the `PolymorphicFunction` to the specified shapes and types. It is\n  an effective way to limit retracing when Tensors have dynamic shapes.\n\n  >>> @tf.function(\n  ...     input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])\n  ... def f(x):\n  ...   return x + 1\n  >>> vector = tf.constant([1.0, 1.0])\n  >>> matrix = tf.constant([[3.0]])\n  >>> f.get_concrete_function(vector) is f.get_concrete_function(matrix)\n  True\n\n  ## Variables may only be created once\n\n  `tf.function` only allows creating new `tf.Variable` objects when it is called\n  for the first time:\n\n  >>> class MyModule(tf.Module):\n  ...   def __init__(self):\n  ...     self.v = None\n  ...\n  ...   @tf.function\n  ...   def __call__(self, x):\n  ...     if self.v is None:\n  ...       self.v = tf.Variable(tf.ones_like(x))\n  ...     return self.v * x\n\n  In general, it is recommended to create `tf.Variable`s outside of\n  `tf.function`.\n  In simple cases, persisting state across `tf.function` boundaries may be\n  implemented using a pure functional style in which state is represented by\n  `tf.Tensor`s passed as arguments and returned as return values.\n\n  Contrast the two styles below:\n\n  >>> state = tf.Variable(1)\n  >>> @tf.function\n  ... def f(x):\n  ...   state.assign_add(x)\n  >>> f(tf.constant(2))  # Non-pure functional style\n  >>> state\n  <tf.Variable ... numpy=3>\n\n  >>> state = tf.constant(1)\n  >>> @tf.function\n  ... def f(state, x):\n  ...   state += x\n  ...   return state\n  >>> state = f(state, tf.constant(2))  # Pure functional style\n  >>> state\n  <tf.Tensor: ... numpy=3>\n\n  ## Python operations execute only once per trace\n\n  `func` may contain TensorFlow operations mixed with pure Python operations.\n  However, when the function is executed, only the TensorFlow operations will\n  run. The Python operations run only once, at trace time. If TensorFlow\n  operations depend on results from Python operations, those results will be\n  frozen into the graph.\n\n  >>> @tf.function\n  ... def f(a, b):\n  ...   print(\'this runs at trace time; a is\', a, \'and b is\', b)\n  ...   return b\n  >>> f(1, tf.constant(1))\n  this runs at trace time; a is 1 and b is Tensor("...", shape=(), dtype=int32)\n  <tf.Tensor: shape=(), dtype=int32, numpy=1>\n\n  >>> f(1, tf.constant(2))\n  <tf.Tensor: shape=(), dtype=int32, numpy=2>\n\n  >>> f(2, tf.constant(1))\n  this runs at trace time; a is 2 and b is Tensor("...", shape=(), dtype=int32)\n  <tf.Tensor: shape=(), dtype=int32, numpy=1>\n\n  >>> f(2, tf.constant(2))\n  <tf.Tensor: shape=(), dtype=int32, numpy=2>\n\n  Args:\n    func: The function to be compiled. If `func` is None, `tf.function` returns\n      a decorator that can be invoked with a single argument - `func`. In other\n      words, `tf.function(input_signature=...)(func)` is equivalent to\n      `tf.function(func, input_signature=...)`. The former can be used as\n      decorator.\n    input_signature: A possibly nested sequence of `tf.TensorSpec` objects\n      specifying the shapes and dtypes of the Tensors that will be supplied to\n      this function. If `None`, a separate function is instantiated for each\n      inferred input signature.  If input_signature is specified, every input to\n      `func` must be a `Tensor`, and `func` cannot accept `**kwargs`.\n    autograph: Whether autograph should be applied on `func` before tracing a\n      graph. Data-dependent Python control flow statements require\n      `autograph=True`. For more information, see the\n      [tf.function and AutoGraph guide](\n      https://www.tensorflow.org/guide/function#autograph_transformations).\n    jit_compile: If `True`, compiles the function using\n      [XLA](https://tensorflow.org/xla). XLA performs compiler optimizations,\n      such as fusion, and attempts to emit more efficient code. This may\n      drastically improve the performance. If set to `True`,\n      the whole function needs to be compilable by XLA, or an\n      `errors.InvalidArgumentError` is thrown.\n      If `None` (default), compiles the function with XLA when running on TPU\n      and goes through the regular function execution path when running on\n      other devices.\n      If `False`, executes the function without XLA compilation.  Set this value\n      to `False` when directly running a multi-device function on TPUs (e.g. two\n      TPU cores, one TPU core and its host CPU).\n      Not all functions are compilable, see a list of\n      [sharp corners](https://tensorflow.org/xla/known_issues).\n    reduce_retracing: When True, `tf.function` attempts to reduce the\n      amount of retracing, for example by using more generic shapes. This\n      can be controlled for user objects by customizing their associated\n      `tf.types.experimental.TraceType`.\n    experimental_implements: If provided, contains a name of a "known" function\n      this implements. For example "mycompany.my_recurrent_cell".\n      This is stored as an attribute in inference function,\n      which can then be detected when processing serialized function.\n      See [standardizing composite ops](https://github.com/tensorflow/community/blob/master/rfcs/20190610-standardizing-composite_ops.md)  # pylint: disable=line-too-long\n      for details.  For an example of utilizing this attribute see this\n      [example](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc)\n      The code above automatically detects and substitutes function that\n      implements "embedded_matmul" and allows TFLite to substitute its own\n      implementations. For instance, a tensorflow user can use this\n       attribute to mark that their function also implements\n      `embedded_matmul` (perhaps more efficiently!)\n      by specifying it using this parameter:\n      `@tf.function(experimental_implements="embedded_matmul")`\n      This can either be specified as just the string name of the function or\n      a NameAttrList corresponding to a list of key-value attributes associated\n      with the function name. The name of the function will be in the \'name\'\n      field of the NameAttrList. To define a formal TF op for this function\n      implements, try the experimental [composite TF](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/tfr)\n      project.\n    experimental_autograph_options: Optional tuple of\n      `tf.autograph.experimental.Feature` values.\n    experimental_attributes: Optional dictionary of attributes to include in the\n      generated FunctionDefs.\n    experimental_relax_shapes: Deprecated. Use `reduce_retracing`\n      instead.\n    experimental_compile: Deprecated alias to \'jit_compile\'.\n    experimental_follow_type_hints: Deprecated. Please use input_signature or\n      reduce_retracing instead.\n\n  Returns:\n     If `func` is not None, returns a `tf.types.experimental.PolymorphicFunction`.\n     If `func` is None, returns a decorator that, when invoked with a single\n     `func` argument, returns a `tf.types.experimental.PolymorphicFunction`.\n\n  Raises:\n     `ValueError` when attempting to use `jit_compile=True`, but XLA support is\n     not available.\n  '
    if jit_compile is None and JIT_COMPILE_FUNCTIONS:
        jit_compile = True
    if experimental_relax_shapes:
        reduce_retracing = True

    def decorated(inner_function):
        if False:
            return 10
        try:
            name = inner_function.__name__
        except AttributeError:
            name = 'function'
        return tf_decorator.make_decorator(inner_function, decorator_name='tf.function', decorator_func=Function(inner_function, name, input_signature=input_signature, autograph=autograph, experimental_autograph_options=experimental_autograph_options, reduce_retracing=reduce_retracing, jit_compile=deprecation.deprecated_argument_lookup('jit_compile', jit_compile, 'experimental_compile', experimental_compile), experimental_implements=experimental_implements, experimental_attributes=experimental_attributes))
    if func is not None:
        return decorated(func)
    return decorated

def class_method_to_instance_method(original_function, instance):
    if False:
        for i in range(10):
            print('nop')
    'Constructs a new `Function` with `self` bound.'
    weak_instance = weakref.ref(instance)
    bound_method = types_lib.MethodType(original_function.python_function, tf_method_target.TfMethodTarget(weak_instance, original_function.python_function))
    assert hasattr(original_function, '_name')
    assert hasattr(original_function, '_autograph')
    assert hasattr(original_function, '_function_type')
    assert hasattr(original_function, 'python_function')
    weak_bound_method_wrapper = None

    def bound_method_wrapper(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Wraps either a dummy MethodType or a converted AutoGraph function.'
        strong_bound_method_wrapper = weak_bound_method_wrapper()
        wrapped_fn = strong_bound_method_wrapper.__wrapped__
        if wrapped_fn is strong_bound_method_wrapper.__original_wrapped__:
            wrapped_fn = original_function.python_function
            return wrapped_fn(weak_instance(), *args, **kwargs)
        return wrapped_fn(*args, **kwargs)
    weak_bound_method_wrapper = weakref.ref(bound_method_wrapper)
    instance_func = type(original_function)(tf_decorator.make_decorator(bound_method, bound_method_wrapper), name=original_function._name, autograph=original_function._autograph, input_signature=original_function.input_signature, reduce_retracing=original_function._reduce_retracing, jit_compile=original_function._jit_compile, experimental_attributes=original_function._attributes)
    wrapped_instance_func = tf_decorator.make_decorator(bound_method, instance_func)
    return wrapped_instance_func