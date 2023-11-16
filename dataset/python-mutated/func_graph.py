"""FuncGraph and related functionality."""
import traceback
from typing import Any, Callable, Hashable
import weakref
from tensorflow.core.function import trace_type
from tensorflow.core.function.capture import capture_container
from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.eager.polymorphic_function import composite_tensor_utils
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.saved_model import save_context
from tensorflow.python.types import core
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
ALLOWLIST_COLLECTIONS = [ops.GraphKeys.GLOBAL_VARIABLES, ops.GraphKeys.LOCAL_VARIABLES, ops.GraphKeys.TRAINABLE_VARIABLES, variable_scope._VARSTORE_KEY, variable_scope._VARSCOPESTORE_KEY]

class UnknownArgument(object):
    """Signifies an argument which is not currently handled."""

def convert_structure_to_signature(structure, arg_names=None, signature_context=None):
    if False:
        i = 10
        return i + 15
    'Convert a potentially nested structure to a signature.\n\n  Args:\n    structure: Structure to convert, where top level collection is a list or a\n      tuple.\n    arg_names: Optional list of arguments that has equal number of elements as\n      `structure` and is used for naming corresponding TensorSpecs.\n    signature_context: TraceType InternalTracingContext to generate alias_ids\n      for mutable objects, like ResourceVariables.\n\n  Returns:\n    Identical structure that has TensorSpec objects instead of Tensors and\n    UnknownArgument instead of any unsupported types.\n  '

    def encode_arg(arg, path):
        if False:
            i = 10
            return i + 15
        'A representation for this argument, for converting into signatures.'
        if isinstance(arg, tensor_lib.Tensor):
            user_specified_name = None
            try:
                user_specified_name = compat.as_str(arg.op.get_attr('_user_specified_name'))
            except (ValueError, AttributeError):
                pass
            if path and user_specified_name and (user_specified_name != path[0]):
                name = user_specified_name
            else:
                name = tensor_lib.sanitize_spec_name('_'.join((str(p) for p in path)))
            return tensor_lib.TensorSpec(arg.shape, arg.dtype, name)
        if isinstance(arg, resource_variable_ops.ResourceVariable):
            return trace_type.from_value(arg, signature_context)
        if isinstance(arg, composite_tensor.CompositeTensor):
            return arg._type_spec
        if isinstance(arg, (int, float, bool, str, type(None), dtypes.DType, tensor_lib.TensorSpec, type_spec.TypeSpec)):
            return arg
        return UnknownArgument()
    flattened = nest.flatten_with_tuple_paths(structure)
    if arg_names:
        if len(arg_names) != len(structure):
            raise ValueError("Passed in arg_names don't match actual signature (%s)." % arg_names)
        flattened = [((arg_names[path[0]],) + path[1:], arg) for (path, arg) in flattened]
    mapped = [encode_arg(arg, path) for (path, arg) in flattened]
    return nest.pack_sequence_as(structure, mapped)

@tf_export('__internal__.FuncGraph', v1=[])
class FuncGraph(ops.Graph):
    """Graph representing a function body.

  Attributes:
    name: The name of the function.
    inputs: Placeholder tensors representing the inputs to this function. The
      tensors are in this FuncGraph. This represents "regular" inputs as well as
      captured inputs (i.e. the values of self.captures), with the regular
      inputs coming first.
    outputs: Tensors that will be returned by this function. The tensors are in
      this FuncGraph.
    control_outputs: Operations that must be executed before the function
      represented by this graph can be said to have been executed.
    structured_input_signature: A tuple of (args, kwargs), which are both
      possibly-nested python objects that were received by this function. Note
      that these structures might contain Python `None`s.
    structured_outputs: A possibly-nested python object which will be returned
      by this function. The Tensors in this structure are the same as those of
      self.outputs. Note that this structure might contain Python `None`s.
    variables: Variables that should be watched during function execution.
    outer_graph: The graph this function is defined in. May be another FuncGraph
      or the global default Graph.
    captures: Maps external tensor -> internal tensor (i.e. input placeholder).
      The entries are in the order they were captured.
    seed: The graph-level random seed.
    capture_by_value: If True, the func graph will capture Variables by value
      instead of reference.
  """

    def __init__(self, name, collections=None, capture_by_value=None, structured_input_signature=None, structured_outputs=None):
        if False:
            print('Hello World!')
        "Construct a new FuncGraph.\n\n    The graph will inherit its graph key, collections, seed, and distribution\n    strategy stack from the current context or graph.\n\n    Args:\n      name: the name of the function.\n      collections: a dictionary of collections this FuncGraph should start with.\n        If not specified (None), the FuncGraph will read (but not write to) the\n        outer graph's collections that are not allowlisted, and both read and\n        write to the outer graph's collections that are allowlisted. The current\n        allowlisted collections are the global variables, the local variables,\n        and the trainable variables. Defaults to None.\n      capture_by_value: An optional boolean. If True, the func graph will\n        capture Variables by value instead of reference. By default inherit from\n        outer graphs, and failing that will default to False.\n      structured_input_signature: Optional. The structured input signature to\n        use for initializing the FuncGraph. See the docstring for FuncGraph for\n        more information.\n      structured_outputs: Optional. The structured outputs to use for\n        initializing the FuncGraph. See the docstring for FuncGraph for more\n        information.\n    "
        super().__init__()
        self.name = name
        self.inputs = []
        self.outputs = []
        self.control_outputs = []
        self.structured_input_signature = structured_input_signature
        self.structured_outputs = structured_outputs
        self._resource_tensor_inputs = object_identity.ObjectIdentitySet()
        self._weak_variables = []
        self._watched_variables = object_identity.ObjectIdentityWeakSet()
        self.is_control_flow_graph = False
        self._function_captures = capture_container.FunctionCaptures()
        outer_graph = ops.get_default_graph()
        self._weak_outer_graph = weakref.ref(outer_graph)
        while outer_graph.building_function:
            outer_graph = outer_graph.outer_graph
        self._fallback_outer_graph = outer_graph
        self._output_names = None
        if capture_by_value is not None:
            self.capture_by_value = capture_by_value
        elif self.outer_graph is not None and isinstance(self.outer_graph, FuncGraph):
            self.capture_by_value = self.outer_graph.capture_by_value
        else:
            self.capture_by_value = False
        self._building_function = True
        graph = self.outer_graph
        if context.executing_eagerly():
            self.seed = context.global_seed()
            self._seed_used = False
        else:
            self.seed = graph.seed
            self._seed_used = False
            self._colocation_stack = graph._colocation_stack.copy()
        if collections is None:
            for collection_name in graph.get_all_collection_keys():
                if collection_name not in ALLOWLIST_COLLECTIONS:
                    self._collections[collection_name] = graph.get_collection(collection_name)
            for collection_name in ALLOWLIST_COLLECTIONS:
                self._collections[collection_name] = graph.get_collection_ref(collection_name)
        else:
            self._collections = collections
        self._saveable = True
        self._saving_errors = set()
        self._scope_exit_callbacks = None

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return 'FuncGraph(name=%s, id=%s)' % (self.name, id(self))

    def watch_variable(self, v):
        if False:
            while True:
                i = 10
        'Marks the variable v as accessed while building this graph.'
        if isinstance(v, resource_variable_ops.ResourceVariable) and v.handle in self._resource_tensor_inputs:
            return
        while self is not None and isinstance(self, FuncGraph):
            self._watched_variables.add(v)
            self = self.outer_graph

    def capture_call_time_value(self, closure, spec, key=None, default_value=None, placeholder=None):
        if False:
            while True:
                i = 10
        'Returns a placeholder which at call time has the value closure().\n\n    The `tf.function` supports the notion of captures, that is, it allows Python\n    functions to have closure variables, which bind over some value outside the\n    function. However, this name binding is "early binding" performed before the\n    program is run, i.e.,\n    ```\n    @tf.function\n    def f():\n      return x\n\n    x = tf.constant(1)\n    f()  # returns 1\n\n    x = tf.constant(2)\n    f()  # still returns 1!\n    ```\n    while in Python, name binding is performed as the program is running.\n    ```\n    def f():\n      return x\n\n    x = 1\n    f()  # returns 1\n\n    x = 2\n    f()  # returns 2\n    ```\n    `capture_call_time_value` allows tf.function to mimic late binding as a\n    Python function does, by passing in a `closure` callable argument to be\n    executed when the tf.function is invoked eagerly.  E.g.\n    ```\n    @tf.function\n    def f():\n      return ops.get_default_graph.capture_call_time_value(lambda: x)\n\n    x = tf.constant(1)\n    f()  # returns 1\n\n    x = tf.constant(2)\n    f()  # returns 2\n    ```\n    Note that a `capture_call_time_value` function itself does not work well in\n    the saving process (since the tf.function in which it\'s called is not\n    invoked eagerly) unless passed a `default_value` argument. At saving time,\n    the `default_value` argument is returned instead.\n\n    Args:\n      closure: function which takes no arguments, to be evaluated at function\n        call time, returning a nest of tensors compatible with `spec`.\n      spec: nest of TypeSpec for the value to capture.\n      key: optional. If not None, multiple calls to lazy_capture with the same\n        key in the same graph will return the same placeholder, and the first\n        closure will be used at function call time.\n      default_value: optional value to return in environments that cannot safely\n        evaluate closure.\n      placeholder: optional. If not None, the graph will take the passed-in\n        `placeholder` as the internal capture instead of creating a new one.\n        This is useful when loading from a SavedModel.\n\n    Returns:\n      Nest of placeholders which, at function call time, will be fed with the\n      result of calling closure().\n\n    Raises:\n      ValueError: at function call time, if the return value of closure() is\n       not compatible with `spec`.\n    '
        if key is None:
            key = object()
        if key not in self._function_captures.by_ref_internal:
            trace_ctx = trace_type.InternalTracingContext(True)
            spec = trace_type.from_value(spec, trace_ctx)
            if placeholder is None:
                placeholder_ctx = trace_type.InternalPlaceholderContext(self)
                placeholder = spec.placeholder_value(placeholder_ctx)

            def wrapped_closure():
                if False:
                    return 10
                if save_context.in_save_context() and default_value is not None:
                    return default_value
                if not context.executing_eagerly():
                    graph = ops.get_default_graph()
                    assert isinstance(graph, FuncGraph), 'This API should only be used in TF2 enviroment.'
                    with graph.as_default():
                        ret_nest = graph.capture_call_time_value(closure, spec, key=key, default_value=default_value)
                else:
                    ret_nest = closure()
                ret_nest = spec.cast(ret_nest, trace_type.InternalCastContext)
                return spec.to_tensors(ret_nest)
            wrapped_closure.output_spec = spec
            self._function_captures.add_or_replace(key=key, external=wrapped_closure, internal=placeholder, tracetype=spec, is_by_ref=True)
        return self._function_captures.by_ref_internal[key]

    def control_dependencies(self, control_inputs):
        if False:
            return 10
        "Handles control dependencies.\n\n    FuncGraph wraps Graph's control_dependencies logic by first filtering out\n    any external tensors / operations and storing them in the graph's\n    control_captures member. Any consumers of this function graph must then\n    decide how to handle the control captures.\n\n    Args:\n      control_inputs: A list of `Operation` or `Tensor` objects which must be\n        executed or computed before running the operations defined in the\n        context.  Can also be `None` to clear the control dependencies.\n\n    Returns:\n     A context manager that specifies control dependencies for all\n     operations constructed within the context.\n\n    Raises:\n      TypeError: If `control_inputs` is not a list of `Operation` or\n        `Tensor` objects.\n    "
        if control_inputs is None:
            return super().control_dependencies(control_inputs)
        filtered_control_inputs = []
        for c in control_inputs:
            if isinstance(c, indexed_slices.IndexedSlices) or (hasattr(c, '_handle') and hasattr(c, 'op')):
                c = c.op
            graph_element = ops._as_graph_element(c)
            if graph_element is None:
                graph_element = c
            if graph_element is not None and getattr(graph_element, 'graph', None) is not self:
                self._function_captures.control.add(graph_element)
            else:
                filtered_control_inputs.append(graph_element)
        return super().control_dependencies(filtered_control_inputs)

    def as_default(self):
        if False:
            while True:
                i = 10
        outer_cm = super().as_default()

        @tf_contextlib.contextmanager
        def inner_cm():
            if False:
                for i in range(10):
                    print('nop')
            'Context manager for copying distribute.Strategy scope information.'
            graph = ops.get_default_graph()
            old_strategy_stack = self._distribution_strategy_stack
            self._distribution_strategy_stack = list(graph._distribution_strategy_stack)
            old_device_stack = self._device_function_stack
            if not context.executing_eagerly() and (device_stack_has_callable(graph._device_function_stack) or (self._distribution_strategy_stack and (not ops.executing_eagerly_outside_functions()))):
                self._device_function_stack = graph._device_function_stack.copy()
            old_creator_stack = self._variable_creator_stack
            self._variable_creator_stack = graph._variable_creator_stack
            old_graph_key = self._graph_key
            self._graph_key = graph._graph_key
            old_scope_exit_callbacks = self._scope_exit_callbacks
            self._scope_exit_callbacks = []
            with outer_cm as g:
                try:
                    yield g
                finally:
                    try:
                        for fn in self._scope_exit_callbacks:
                            fn()
                    finally:
                        self._scope_exit_callbacks = old_scope_exit_callbacks
                        self._distribution_strategy_stack = old_strategy_stack
                        self._device_function_stack = old_device_stack
                        self._variable_creator_stack = old_creator_stack
                        self._graph_key = old_graph_key
        return inner_cm()

    @property
    def outer_graph(self):
        if False:
            print('Hello World!')
        "The Graph this FuncGraph is nested in.\n\n    Functions may capture Tensors from graphs they are nested in (transitive).\n\n    Returns:\n      A Graph object. Initially set to the current default graph when the\n      FuncGraph was created. If the previous `outer_graph` was deleted because\n      the function that owns it was deleted, `outer_graph` is reset to the\n      outermost default graph active when the FuncGraph was created. This\n      FuncGraph won't have captured anything from the new `outer_graph` (and\n      likely not from the previous setting, since that would have created a\n      strong reference), but it is returned so that FuncGraphs always have a\n      parent.\n    "
        current = self._weak_outer_graph()
        if current is None:
            return self._fallback_outer_graph
        return current

    @outer_graph.setter
    def outer_graph(self, new_outer_graph):
        if False:
            for i in range(10):
                print('nop')
        'Sets `outer_graph` to `new_outer_graph`.'
        self._weak_outer_graph = weakref.ref(new_outer_graph)

    @property
    def output_types(self):
        if False:
            for i in range(10):
                print('nop')
        return [t.dtype for t in self.outputs]

    @property
    def output_shapes(self):
        if False:
            print('Hello World!')
        return [t.shape for t in self.outputs]

    @property
    def trainable_variables(self):
        if False:
            i = 10
            return i + 15
        'A sequence of trainable variables accessed by this FuncGraph.\n\n    Note that functions keep only weak references to variables. Calling the\n    function after a variable it accesses has been deleted is an error.\n\n    Returns:\n      Sequence of trainable variables for this func graph.\n    '
        return tuple((v for v in self.variables if v.trainable))

    @property
    def variables(self):
        if False:
            return 10
        'A sequence of variables accessed by this FuncGraph.\n\n    Note that functions keep only weak references to variables. Calling the\n    function after a variable it accesses has been deleted is an error.\n\n    Returns:\n      Sequence of variables for this func graph.\n    '

        def deref(weak_v):
            if False:
                for i in range(10):
                    print('nop')
            v = weak_v()
            if v is None:
                raise AssertionError('Called a function referencing variables which have been deleted. This likely means that function-local variables were created and not referenced elsewhere in the program. This is generally a mistake; consider storing variables in an object attribute on first call.')
            return v
        return tuple((deref(v) for v in self._weak_variables))

    @variables.setter
    def variables(self, var_list):
        if False:
            for i in range(10):
                print('nop')
        self._weak_variables = [weakref.ref(v) for v in var_list]

    def _capture_by_value(self, op_type, inputs, dtypes, input_types=None, name=None, attrs=None, op_def=None, compute_device=True):
        if False:
            for i in range(10):
                print('nop')
        reverse_captures = dict(((id(v), k) for (k, v) in self.captures))
        uncaptured_inputs = [reverse_captures.get(id(t), t) for t in inputs]
        with ops.init_scope():
            if context.executing_eagerly():
                attr_list = ('dtype', int(attrs['dtype'].type))
                (value,) = execute.execute(compat.as_bytes(op_type), 1, uncaptured_inputs, attr_list, context.context())
            else:
                op = ops.get_default_graph()._create_op_internal(op_type, uncaptured_inputs, dtypes, input_types, name, attrs, op_def, compute_device)
                value = op.outputs[0]
        captured_value = self.capture(value)
        return captured_value.op

    def _create_op_internal(self, op_type, inputs, dtypes=None, input_types=None, name=None, attrs=None, op_def=None, compute_device=True):
        if False:
            return 10
        'Like Graph.create_op, except handles external input tensors.\n\n    This overload adds functionality to create_op to "capture" any external\n    input tensors, i.e. tensors from the eager context or outer function graphs\n    if this is a nested function. See `capture` for more information.\n\n    Args:\n      op_type: The `Operation` type to create. This corresponds to the\n        `OpDef.name` field for the proto that defines the operation.\n      inputs: A list of `Tensor` objects that will be inputs to the `Operation`.\n      dtypes: (Optional) A list of `DType` objects that will be the types of the\n        tensors that the operation produces.\n      input_types: (Optional.) A list of `DType`s that will be the types of the\n        tensors that the operation consumes. By default, uses the base `DType`\n        of each input in `inputs`. Operations that expect reference-typed inputs\n        must specify `input_types` explicitly.\n      name: (Optional.) A string name for the operation. If not specified, a\n        name is generated based on `op_type`.\n      attrs: (Optional.) A dictionary where the key is the attribute name (a\n        string) and the value is the respective `attr` attribute of the\n        `NodeDef` proto that will represent the operation (an `AttrValue`\n        proto).\n      op_def: (Optional.) The `OpDef` proto that describes the `op_type` that\n        the operation will have.\n      compute_device: (Optional.) If True, device functions will be executed to\n        compute the device property of the Operation.\n\n    Returns:\n      An `Operation` object.\n    '
        if self.capture_by_value and op_type in ['ReadVariableOp', 'ResourceGather']:
            return self._capture_by_value(op_type, inputs, dtypes, input_types, name, attrs, op_def, compute_device)
        if op_type == 'Enter' and inputs[0].op.type == 'Enter':
            if inputs[0].op.get_attr('frame_name') == attrs['frame_name'].s:
                return inputs[0].op
        ctxt = ops.get_default_graph()._control_flow_context
        captured_inputs = []
        for inp in inputs:
            if ctxt is not None and hasattr(ctxt, 'AddValue'):
                inp = ctxt.AddValue(inp)
            inp = self.capture(inp)
            captured_inputs.append(inp)
        return super()._create_op_internal(op_type, captured_inputs, dtypes, input_types, name, attrs, op_def, compute_device)

    def capture(self, tensor, name=None, shape=None):
        if False:
            print('Hello World!')
        return self._function_captures.capture_by_value(self, tensor, name)

    def _validate_in_scope(self, tensor):
        if False:
            for i in range(10):
                print('nop')
        inner_graph = tensor.graph
        while inner_graph is not None and isinstance(inner_graph, FuncGraph):
            if inner_graph is self:
                try:
                    tb = tensor.op.traceback
                except AttributeError:
                    tensor_traceback = '<unknown>'
                else:
                    tensor_traceback_list = []
                    for frame in traceback.format_list(tb.get_user_frames()):
                        tensor_traceback_list.extend([f'  {line}' for line in frame.split('\n') if line.strip()])
                    tensor_traceback = '\n'.join(tensor_traceback_list)
                raise errors.InaccessibleTensorError(f'{tensor!r} is out of scope and cannot be used here. Use return values, explicit Python locals or TensorFlow collections to access it.\nPlease see https://www.tensorflow.org/guide/function#all_outputs_of_a_tffunction_must_be_return_values for more information.\n\n{tensor!r} was defined here:\n{tensor_traceback}\n\nThe tensor {tensor!r} cannot be accessed from {self}, because it was defined in {tensor.graph}, which is out of scope.')
            inner_graph = inner_graph.outer_graph

    def _capture_helper(self, tensor, name):
        if False:
            i = 10
            return i + 15
        return self._function_captures._create_placeholder_helper(self, tensor, name)

    def _experimental_capture_side_input_by_ref(self, identifier: Hashable, func: Callable[[], Any]) -> ...:
        if False:
            while True:
                i = 10
        'Implement capturing side input by reference for tf.function.\n\n    Note that this API will only register the capture in the func_graph where\n    it is called. In the case of nested graph, like nested tf.function or\n    tf.while, the outer graph is not aware of this capture in the inner graph.\n    Thus, the outer tf.function will not retrace when the by-ref capture\n    changes. It\'s the user\'s responsibility to call this API in the outer\n    func_graph as well if proper retracing is needed.\n\n    For example:\n\n    ```\n    x = 1\n\n    # Correct usage\n    @tf.function\n    def f_1():\n      graph = tf.compat.v1.get_default_graph()\n      # Capture the same x for the outer tf.function\n      graph._experimental_capture_side_input_by_ref("x", lambda: x)\n\n      @tf.function\n      def g():\n        graph = tf.compat.v1.get_default_graph()\n        cap_x = graph._experimental_capture_side_input_by_ref("x", lambda: x)\n        return cap_x + 1\n\n      return g()\n\n    # Incorrect usage\n    @tf.function\n    def f_2():\n\n      @tf.function\n      def g():\n        graph = tf.compat.v1.get_default_graph()\n        cap_x = graph._experimental_capture_side_input_by_ref("x", lambda: x)\n        return cap_x + 1\n\n      return g()\n\n    assert f_1() == 2\n    assert f_2() == 2\n    x = 2\n    assert f_1() == 3\n    assert f_2() == 2  # This is incorrect\n    ```\n\n    Args:\n      identifier: A hashable object as the key for the capture.\n      func: A Python function that takes no arguments and returns the value of\n        side input. The function is evaluated at function call time.\n\n    Returns:\n      A nested structure with the same structure as the side input. Tensors\n        are replaced with placehoders, and non-tensors remain the same.\n\n    '
        if context.executing_eagerly():
            return func()

        def maybe_convert_to_tensor():
            if False:
                while True:
                    i = 10
            value = func()
            if not (isinstance(value, core.Value) or isinstance(value, core.Symbol)):
                value = constant_op.constant(value)
            return value
        placeholder = self._function_captures._capture_by_ref(self, maybe_convert_to_tensor, identifier)
        return placeholder

    @property
    def captures(self):
        if False:
            print('Hello World!')
        'Order list of tuples containing external and internal captures.'
        return self._function_captures.by_val_capture_tuples

    def add_capture(self, tensor, placeholder):
        if False:
            i = 10
            return i + 15
        'Capture a specific tensor and utilize the provided placeholder.\n\n    Args:\n      tensor: Tensor to captures.\n      placeholder: Provided placeholder for the tensor.\n    '
        self._function_captures.add_or_replace(key=id(tensor), external=tensor, internal=placeholder, is_by_ref=False)
        self.inputs.append(placeholder)

    def replace_capture(self, tensor, placeholder):
        if False:
            for i in range(10):
                print('nop')
        'Replace already existing capture.'
        self._function_captures.add_or_replace(key=id(tensor), external=tensor, internal=placeholder, is_by_ref=False)

    def replace_capture_with_deferred_capture(self, tensor, closure, spec, placeholder, default_value=None):
        if False:
            for i in range(10):
                print('nop')
        "Replaces existing capture `tensor` with a deferred capture `closure`.\n\n    Caution: It is the caller's responsibility to make sure that, after calling\n    this function, the TypeSpec of the `inputs` (i.e. internal placeholders) and\n    the `_captured_inputs` (i.e. external captures) of a concrete function that\n    wraps this function graph are still compatible. Thus user should pairing\n    usage of this function with `ConcreteFunction.set_external_captures` to make\n    sure the order still matches. For example,\n    ```\n    # concrete_fn._captured_inputs == [tensor1, tensor2, tensor3]\n    # concrete_fn.inputs == [placeholder1, placeholder2, placeholder3]\n    # replace external capture `tensor2` with a deferred_capture, i.e., a\n    # closure, `closure2`\n    concrete_fn.graph.replace_capture_with_deferred_capture(tensor2,\n                                                            closure2,\n                                                            placeholder2,\n                                                            some_spec,\n                                                            some_default)\n    concrete_fn.set_external_captures([tensor1, closure2, tensor3])\n    ```\n\n    Args:\n      tensor: Tensor already captured.\n      closure: function which takes no arguments, to be evaluated at function\n        call time, returning a nest of tensors compatible with `spec`.\n      spec: nest of TypeSpec for the value to capture.\n      placeholder: the internal placeholder corresponding to the captured\n        `tensor`.\n      default_value: optional value to use in environments that cannot safely\n        evaluate closure.\n    "
        self._function_captures.pop(id(tensor), is_by_ref=False)
        self.capture_call_time_value(closure, spec, key=id(tensor), default_value=default_value, placeholder=placeholder)

    @property
    def external_captures(self):
        if False:
            return 10
        'External tensors captured by this function.'
        return list(self._function_captures.by_val_external.values())

    @property
    def internal_captures(self):
        if False:
            print('Hello World!')
        'Placeholders in this function corresponding captured tensors.'
        return list(self._function_captures.by_val_internal.values())

    @property
    def deferred_external_captures(self):
        if False:
            i = 10
            return i + 15
        'Ordered nest of tensors whose placeholders will be fed at call time.'
        return list(self._function_captures.by_ref_external.values())

    @property
    def deferred_internal_captures(self):
        if False:
            for i in range(10):
                print('nop')
        'List of nest of placeholders which at call time will be fed.'
        return list(self._function_captures.by_ref_internal.values())

    @property
    def variable_captures(self):
        if False:
            i = 10
            return i + 15
        'Map of python object ids of variables to variables which are captured.'
        return self.variables

    @property
    def function_captures(self):
        if False:
            return 10
        return self._function_captures

    def mark_as_unsaveable(self, error_message):
        if False:
            i = 10
            return i + 15
        'Marks this FuncGraph as unsaveable.\n\n    Any attempts to export this FuncGraph will raise an error with the specified\n    message.\n\n    Args:\n      error_message: List or string containing the error message to be raised\n        when saving this FuncGraph to SavedModel.\n    '
        self._saveable = False
        if isinstance(error_message, str):
            error_message = [error_message]
        self._saving_errors.update(error_message)

    @property
    def saveable(self):
        if False:
            return 10
        'Returns whether this FuncGraph is saveable.'
        return self._saveable

    @property
    def saving_errors(self):
        if False:
            i = 10
            return i + 15
        'Returns set of errors preventing this FuncGraph from being saved.'
        return self._saving_errors

    def _add_scope_exit_callback(self, fn):
        if False:
            while True:
                i = 10
        'Add a function to call when this graph exits the default scope.'
        if not callable(fn):
            raise TypeError('fn is not callable: {}'.format(fn))
        if self._scope_exit_callbacks is None:
            raise RuntimeError("Attempting to add a scope exit callback, but the default graph is not the context scope graph.  Did you forget to call 'with graph.as_default(): ...'?")
        self._scope_exit_callbacks.append(fn)

def func_graph_from_py_func(name, python_func, args, kwargs, signature=None, func_graph=None, add_control_dependencies=True, arg_names=None, op_return_value=None, collections=None, capture_by_value=None, create_placeholders=True):
    if False:
        print('Hello World!')
    "Returns a `FuncGraph` generated from `python_func`.\n\n  Args:\n    name: an identifier for the function.\n    python_func: the Python function to trace.\n    args: the positional args with which the Python function should be called;\n      ignored if a signature is provided.\n    kwargs: the keyword args with which the Python function should be called;\n      ignored if a signature is provided.\n    signature: a possibly nested sequence of `TensorSpecs` specifying the shapes\n      and dtypes of the arguments. When a signature is provided, `args` and\n      `kwargs` are ignored, and `python_func` is traced with Tensors conforming\n      to `signature`. If `None`, the shapes and dtypes are inferred from the\n      inputs.\n    func_graph: Optional. An instance of FuncGraph. If provided, we will use\n      this graph else a new one is built and returned.\n    add_control_dependencies: If True, automatically adds control dependencies\n      to ensure program order matches execution order and stateful ops always\n      execute.\n    arg_names: Optional list of argument names, used to give input placeholders\n      recognizable names.\n    op_return_value: Optional. A Tensor. If set and `python_func` returns\n      Operations, those return values will be replaced with this value. If not\n      set, returning an Operation triggers an error.\n    collections: a dictionary of collections this FuncGraph should start with.\n      If not specified (None), the FuncGraph will read (but not write to) the\n      outer graph's collections that are not allowlisted, and both read and\n      write to the outer graph's collections that are allowlisted. The current\n      allowlisted collections are the global variables, the local variables, and\n      the trainable variables. Defaults to None.\n    capture_by_value: An optional boolean. If True, the func graph will capture\n      Variables by value instead of reference. By default inherit from outer\n      graphs, and failing that will default to False.\n    create_placeholders: An optional boolean. If True, then func graph will\n      create placeholders for the inputs as graph ops. If False, the input args\n      and kwargs will be treated as the input placeholders.\n\n  Returns:\n    A FuncGraph.\n\n  Raises:\n    TypeError: If any of `python_func`'s return values is neither `None`, a\n      `Tensor` or a `tf.experimental.ExtensionType`.\n  "
    if op_return_value is not None:
        assert isinstance(op_return_value, tensor_lib.Tensor), op_return_value
    if func_graph is None:
        func_graph = FuncGraph(name, collections=collections, capture_by_value=capture_by_value)
    assert isinstance(func_graph, FuncGraph)
    if add_control_dependencies:
        deps_control_manager = auto_control_deps.AutomaticControlDependencies()
    else:
        deps_control_manager = ops.NullContextmanager()
    with func_graph.as_default(), deps_control_manager as deps_ctx:
        current_scope = variable_scope.get_variable_scope()
        default_use_resource = current_scope.use_resource
        current_scope.set_use_resource(True)
        if signature is not None:
            args = signature
            kwargs = {}
        if create_placeholders:
            (func_args, func_kwargs) = _create_placeholders(args, kwargs, arg_names)
        else:
            (func_args, func_kwargs) = (args, kwargs)
        input_trace_types = trace_type.from_value([func_args, func_kwargs])
        func_graph.inputs = input_trace_types.to_tensors([func_args, func_kwargs])
        func_graph._watched_variables = object_identity.ObjectIdentityWeakSet()
        for arg in func_graph.inputs:
            if arg.dtype == dtypes.resource:
                func_graph._resource_tensor_inputs.add(arg)
        signature_context = trace_type.InternalTracingContext()
        func_graph.structured_input_signature = (convert_structure_to_signature(func_args, arg_names, signature_context=signature_context), convert_structure_to_signature(func_kwargs, signature_context=signature_context))
        func_args_before = nest.pack_sequence_as(func_args, nest.flatten(func_args, expand_composites=True), expand_composites=True)
        func_kwargs_before = nest.pack_sequence_as(func_kwargs, nest.flatten(func_kwargs, expand_composites=True), expand_composites=True)

        def convert(x):
            if False:
                i = 10
                return i + 15
            'Converts a function output to a Tensor.'
            if x is None:
                return None
            if op_return_value is not None and isinstance(x, ops.Operation):
                with ops.control_dependencies([x]):
                    x = array_ops.identity(op_return_value)
            elif not isinstance(x, tensor_array_ops.TensorArray):
                try:
                    x = ops.convert_to_tensor_or_composite(x)
                except (ValueError, TypeError):
                    raise TypeError(f'To be compatible with tf.function, Python functions must return zero or more Tensors or ExtensionTypes or None values; in compilation of {str(python_func)}, found return value of type {type(x).__name__}, which is not a Tensor or ExtensionType.')
            if add_control_dependencies:
                x = deps_ctx.mark_as_return(x)
            return x
        (_, original_func) = tf_decorator.unwrap(python_func)
        func_outputs = python_func(*func_args, **func_kwargs)
        func_outputs = variable_utils.convert_variables_to_tensors(func_outputs)
        func_outputs = nest.map_structure(convert, func_outputs, expand_composites=True)
        func_args = nest.pack_sequence_as(func_args, nest.flatten(func_args, expand_composites=True), expand_composites=True)
        func_kwargs = nest.pack_sequence_as(func_kwargs, nest.flatten(func_kwargs, expand_composites=True), expand_composites=True)
        check_func_mutation(func_args_before, func_kwargs_before, func_args, func_kwargs, original_func)
        current_scope.set_use_resource(default_use_resource)
        inputs = []
        for arg in composite_tensor_utils.flatten_with_variables([func_args, func_kwargs]):
            if isinstance(arg, resource_variable_ops.BaseResourceVariable):
                capture = func_graph._function_captures.pop(id(arg.handle), False)
                assert len(capture) >= 2
                resource_placeholder = capture[1]
                if resource_placeholder is None:
                    continue
                inputs.append(resource_placeholder)
            elif isinstance(arg, tensor_lib.Tensor):
                inputs.append(arg)
        func_graph.inputs = inputs + func_graph.internal_captures + nest.flatten(func_graph.deferred_internal_captures, expand_composites=True)
        func_graph.structured_outputs = func_outputs
        func_graph.outputs.extend((func_graph.capture(x) for x in flatten(func_graph.structured_outputs) if x is not None))
        func_graph.variables = func_graph._watched_variables
    if add_control_dependencies:
        func_graph.control_outputs.extend(deps_control_manager.ops_which_must_run)
        func_graph.collective_manager_ids_used = deps_control_manager.collective_manager_ids_used
    return func_graph

def maybe_captured(tensor):
    if False:
        for i in range(10):
            print('nop')
    'If t is a captured value placeholder, returns the original captured value.\n\n  Args:\n    tensor: Tensor.\n\n  Returns:\n    A tensor, potentially from a different Graph/FuncGraph.\n  '
    if not isinstance(tensor, ops.EagerTensor) and tensor.op.graph.building_function and (tensor.op.type == 'Placeholder'):
        for (input_t, placeholder_t) in tensor.op.graph.captures:
            if tensor == placeholder_t:
                return maybe_captured(input_t)
    return tensor

def device_stack_has_callable(device_stack):
    if False:
        i = 10
        return i + 15
    'Checks whether a device stack contains a callable.'
    return any((callable(spec._device_name_or_function) for spec in device_stack.peek_objs()))

def has_mutation(n1, n2):
    if False:
        for i in range(10):
            print('nop')
    'Returns true if n1 and n2 are different (using `is` to compare leaves).'
    try:
        nest.assert_same_structure(n1, n2, expand_composites=True)
    except ValueError:
        return True
    for (arg1, arg2) in zip(nest.flatten(n1, expand_composites=True), nest.flatten(n2, expand_composites=True)):
        if arg1 is not arg2:
            return True
    return False

def check_func_mutation(old_args, old_kwargs, new_args, new_kwargs, func):
    if False:
        for i in range(10):
            print('nop')
    'Checks that the arguments to a function are not modified.'
    if not has_mutation((old_args, old_kwargs), (new_args, new_kwargs)):
        return
    func_name = getattr(func, '__qualname__', getattr(func, '__name__', func))
    signature = tf_inspect.signature(func)
    try:
        old_bound = signature.bind(*old_args, **old_kwargs).arguments
        new_bound = signature.bind(*new_args, **new_kwargs).arguments
    except TypeError as e:
        raise ValueError(f'{func_name}{signature} should not modify its Python input arguments. Check if it modifies any lists or dicts passed as arguments. Modifying a copy is allowed.') from e
    assert set(old_bound) == set(new_bound)
    modified_args = [arg_name for arg_name in new_bound if has_mutation(old_bound[arg_name], new_bound[arg_name])]
    changes = ', '.join(modified_args)
    raise ValueError(f'{func_name}{signature} should not modify its Python input arguments. Modifying a copy is allowed. The following parameter(s) were modified: {changes}')

def flatten(sequence):
    if False:
        print('Hello World!')
    'Like nest.flatten w/ expand_composites, but returns flow for TensorArrays.\n\n  Args:\n    sequence: A nested structure of Tensors, CompositeTensors, and TensorArrays.\n\n  Returns:\n    A list of tensors.\n  '
    flat_sequence = nest.flatten(sequence, expand_composites=True)
    return [item.flow if isinstance(item, tensor_array_ops.TensorArray) else item for item in flat_sequence]

def pack_sequence_as(structure, flat_sequence):
    if False:
        print('Hello World!')
    'Like `nest.pack_sequence_as` but also builds TensorArrays from flows.\n\n  Args:\n    structure: The structure to pack into. May contain Tensors,\n      CompositeTensors, or TensorArrays.\n    flat_sequence: An iterable containing tensors.\n\n  Returns:\n    A nested structure.\n\n  Raises:\n    AssertionError if `structure` and `flat_sequence` are not compatible.\n  '
    flat_sequence = list(flat_sequence)
    flattened_structure = nest.flatten(structure, expand_composites=True)
    if len(flattened_structure) != len(flat_sequence):
        raise ValueError('Mismatch in element count')
    for i in range(len(flat_sequence)):
        if isinstance(flattened_structure[i], tensor_array_ops.TensorArray):
            flat_sequence[i] = tensor_array_ops.build_ta_with_new_flow(old_ta=flattened_structure[i], flow=flat_sequence[i])
    return nest.pack_sequence_as(structure, flat_sequence, expand_composites=True)

def _create_placeholders(args, kwargs, arg_names=None):
    if False:
        print('Hello World!')
    'Create placeholders given positional args and keyword args.'
    signature_context = trace_type.InternalTracingContext(is_legacy_signature=True)
    arg_trace_types = trace_type.from_value(tuple(args), signature_context)
    kwarg_trace_types = trace_type.from_value(kwargs, signature_context)
    placeholder_mapping = signature_context.get_placeholder_mapping()
    placeholder_context = trace_type.InternalPlaceholderContext(ops.get_default_graph(), placeholder_mapping)
    if arg_names is None:
        arg_names = [None] * len(arg_trace_types.components)
    func_args = []
    for (name, trace_type_arg) in zip(arg_names, arg_trace_types.components):
        placeholder_context.update_naming_scope(name)
        placeholder = trace_type_arg.placeholder_value(placeholder_context)
        func_args.append(placeholder)
    func_kwargs = {}
    for (name, trace_type_kwarg) in zip(*sorted(kwarg_trace_types.mapping.items())):
        placeholder_context.update_naming_scope(name)
        placeholder = trace_type_kwarg.placeholder_value(placeholder_context)
        func_kwargs[name] = placeholder
    return (tuple(func_args), func_kwargs)

def dismantle_func_graph(func_graph):
    if False:
        for i in range(10):
            print('nop')
    "Removes reference cycles in `func_graph` FuncGraph.\n\n  Helpful for making sure the garbage collector doesn't need to run when\n  the FuncGraph goes out of scope, e.g. in tests using defun with\n  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True).\n\n  Args:\n    func_graph: A `FuncGraph` object to destroy. `func_graph` is unusable after\n      this function.\n  "
    func_graph._function_captures.clear()
    ops.dismantle_graph(func_graph)

def override_func_graph_name_scope(func_graph, name_scope):
    if False:
        i = 10
        return i + 15
    func_graph._name_stack = name_scope