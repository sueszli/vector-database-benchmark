"""OutsideCompilation, TPUReplicateContext, and supporting functions."""
from typing import Any, Callable, List, Optional, Text, Tuple, Union
from absl import logging
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables
from tensorflow.python.tpu import device_assignment as device_assignment_lib
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.types import core as core_types
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
_MAX_WARNING_LINES = 5
_TPU_REPLICATE_ATTR = '_tpu_replicate'
_OUTSIDE_COMPILATION_ATTR = '_xla_outside_compilation'
_MAP_OUTSIDE_COMPILATION_ATTR = '_xla_map_outside_compilation'
_DENYLISTED_OPS = frozenset(['Placeholder'])
_UNSUPPORTED_OPS = frozenset(['AudioSummary', 'AudioSummaryV2', 'HistogramSummary', 'ImageSummary', 'MergeSummary', 'Print', 'ScalarSummary', 'TensorSummary', 'TensorSummaryV2'])

def is_tpu_strategy(strategy: Any) -> bool:
    if False:
        i = 10
        return i + 15
    is_tpu_strat = lambda k: k.__name__.startswith('TPUStrategy')
    clz = strategy.__class__
    return is_tpu_strat(clz) or any(map(is_tpu_strat, clz.__bases__))

def _enclosing_tpu_device_assignment() -> Optional[device_assignment_lib.DeviceAssignment]:
    if False:
        i = 10
        return i + 15
    if not distribute_lib.has_strategy():
        return None
    strategy = distribute_lib.get_strategy()
    if not is_tpu_strategy(strategy):
        return None
    return strategy.extended._device_assignment

class TPUReplicateContext(control_flow_ops.XLAControlFlowContext):
    """A `ControlFlowContext` for nodes inside a TPU computation.

  The primary role of `TPUReplicateContext` is to mark operators inside a
  tpu.replicate() computation with the attribute "_tpu_replicate=XYZ", where XYZ
  is a unique name.

  We use a `ControlFlowContext` to perform the annotation since it integrates
  with Tensorflow constructs like ResourceVariables. For example, if a
  `ResourceVariable` is constructed inside a tpu.replicate() block, the
  `ResourceVariable` implementation can use
  `with ops.control_dependencies(None)` to build the variable's definition
  outside the replicated computation.
  """

    def __init__(self, name: Text, num_replicas: int, pivot: ops.Operation):
        if False:
            for i in range(10):
                print('nop')
        'Builds a new TPUReplicateContext.\n\n    Args:\n      name: a unique name for the context, used to populate the `_tpu_replicate`\n        attribute.\n      num_replicas: an integer that gives the number of replicas for the\n        computation.\n      pivot: a pivot node. Nodes in the TPUReplicateContext that do not have any\n        inputs will have a control dependency on the pivot node. This ensures\n        that nodes are correctly included in any enclosing control flow\n        contexts.\n    '
        super(TPUReplicateContext, self).__init__()
        self._num_replicas = num_replicas
        self._outer_device_function_stack = None
        self._oc_dev_fn_stack = None
        self._outside_compilation_cluster = None
        self._is_map_outside_compilation = False
        self._outside_compilation_v2_context = None
        self._outside_compilation_counter = 0
        self._in_gradient_colocation = None
        self._gradient_colocation_stack = []
        self._host_compute_core = []
        self._name = name
        self._tpu_replicate_attr = attr_value_pb2.AttrValue(s=compat.as_bytes(self._name))
        self._unsupported_ops = []
        self._pivot = pivot
        self._replicated_vars = {}

    def get_replicated_var_handle(self, name: Text, handle_id: Text, vars_: Union[List[core_types.Tensor], List[variables.Variable]], is_mirrored: bool=False, is_packed: bool=False) -> core_types.Tensor:
        if False:
            while True:
                i = 10
        "Returns a variable handle for replicated TPU variable 'var'.\n\n    This is a method used by an experimental replicated variable implementation\n    and is not intended as a public API.\n\n    Args:\n      name: The common name of the variable.\n      handle_id: Unique ID of the variable handle, used as the cache key.\n      vars_: The replicated TPU variables or handles.\n      is_mirrored: Whether the variables are mirrored, which guarantees the\n        values in each replica are always the same.\n      is_packed: Whether the replicated variables are packed into one variable.\n\n    Returns:\n      The handle of the TPU replicated input node.\n    "
        device_assignment = _enclosing_tpu_device_assignment()
        handle = self._replicated_vars.get(handle_id)
        if handle is not None:
            return handle
        if device_assignment is not None and (not is_packed):
            job_name = pydev.DeviceSpec.from_string(vars_[0].device).job
            devices_to_vars = {device_util.canonicalize(v.device): v for v in vars_}
            replicated_vars = []
            for replica_id in range(device_assignment.num_replicas):
                for logical_core in range(device_assignment.num_cores_per_replica):
                    device = device_util.canonicalize(device_assignment.tpu_device(replica=replica_id, logical_core=logical_core, job=job_name))
                    if device in devices_to_vars:
                        replicated_vars.append(devices_to_vars[device])
                        break
                else:
                    raise ValueError('Failed to find a variable on any device in replica {} for current device assignment'.format(replica_id))
        else:
            replicated_vars = vars_
        (_, graph) = _enclosing_tpu_context_and_graph()
        with graph.as_default():
            if isinstance(replicated_vars[0], variables.Variable):
                replicated_vars = [v.handle for v in replicated_vars]
            saved_context = graph._get_control_flow_context()
            graph._set_control_flow_context(self.outer_context)
            handle = tpu_ops.tpu_replicated_input(replicated_vars, name=name + '/handle', is_mirrored_variable=is_mirrored, is_packed=is_packed)
            graph._set_control_flow_context(saved_context)
        self._replicated_vars[handle_id] = handle
        return handle

    def report_unsupported_operations(self) -> None:
        if False:
            i = 10
            return i + 15
        if self._unsupported_ops:
            op_str = '\n'.join(('  %s (%s)' % (op.type, op.name) for op in self._unsupported_ops[:_MAX_WARNING_LINES]))
            logging.warning('%d unsupported operations found: \n%s', len(self._unsupported_ops), op_str)
            if len(self._unsupported_ops) > _MAX_WARNING_LINES:
                logging.warning('... and %d more', len(self._unsupported_ops) - _MAX_WARNING_LINES)

    def EnterGradientColocation(self, op: ops.Operation, gradient_uid: Text):
        if False:
            while True:
                i = 10
        if op is not None:
            if ops.get_default_graph()._control_flow_context is None:
                try:
                    outside_attr = op.get_attr(_OUTSIDE_COMPILATION_ATTR).decode('ascii')
                except ValueError:
                    return
                parts = outside_attr.split('.')
                cluster = parts[0] + '.' + gradient_uid
                self._outside_compilation_v2_context = OutsideCompilationV2Context(cluster)
                self._outside_compilation_v2_context.Enter()
                return
            self._gradient_colocation_stack.append(op)
            if not self._outside_compilation_cluster:
                try:
                    outside_attr = op.get_attr(_OUTSIDE_COMPILATION_ATTR).decode('ascii')
                    if self._in_gradient_colocation:
                        raise NotImplementedError('Cannot nest gradient colocation operations outside compilation')
                    if gradient_uid == '__unsupported__':
                        raise NotImplementedError('No gradient_uid calling gradient within outside_compilation')
                    self._in_gradient_colocation = op
                    parts = outside_attr.split('.')
                    cluster = parts[0] + '.' + gradient_uid
                    self._EnterOutsideCompilationScope(cluster=cluster)
                except ValueError:
                    pass

    def ExitGradientColocation(self, op: ops.Operation, gradient_uid: Text):
        if False:
            while True:
                i = 10
        if op is not None:
            if ops.get_default_graph()._control_flow_context is None:
                assert self._outside_compilation_v2_context is None
                return
            if self._outside_compilation_v2_context is not None:
                self._outside_compilation_v2_context.Exit()
                self._outside_compilation_v2_context = None
                return
            if not self._gradient_colocation_stack:
                raise errors.InternalError(op.node_def, op, 'Badly nested gradient colocation: ' + f'empty stack when popping Op {op.name}')
            last_op = self._gradient_colocation_stack.pop()
            if op is last_op:
                if op is self._in_gradient_colocation:
                    self._in_gradient_colocation = None
                    self._ExitOutsideCompilationScope()
            else:
                raise errors.InternalError(op.node_def, op, 'Badly nested gradient colocation, ' + f'expected {last_op}, got {op.name}')

    def _EnterOutsideCompilationScope(self, cluster: Optional[Text]=None, is_map_outside_compilation=False):
        if False:
            print('Hello World!')

        class FakeOp(object):
            """A helper class to determine the current device.

      Supports only the type and device set/get methods needed to run the
      graph's _apply_device_function method.
      """

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                self._device = ''

            @property
            def type(self):
                if False:
                    while True:
                        i = 10
                return 'FakeOp'

            @property
            def device(self):
                if False:
                    return 10
                return self._device

            def _set_device(self, device):
                if False:
                    i = 10
                    return i + 15
                if isinstance(device, pydev.DeviceSpec):
                    self._device = device.to_string()
                else:
                    self._device = device

            def _set_device_from_string(self, device_str):
                if False:
                    i = 10
                    return i + 15
                self._device = device_str
        if self._outside_compilation_cluster:
            raise NotImplementedError('Cannot nest outside_compilation clusters')
        if cluster:
            self._outside_compilation_cluster = cluster
        else:
            self._outside_compilation_cluster = str(self._outside_compilation_counter)
            self._outside_compilation_counter += 1
        if is_map_outside_compilation:
            self._is_map_outside_compilation = True
        graph = ops.get_default_graph()
        fake_op = FakeOp()
        graph._apply_device_functions(fake_op)
        device = pydev.DeviceSpec.from_string(fake_op.device)
        if device.device_type == 'TPU_REPLICATED_CORE' and device.device_index is not None:
            self._host_compute_core.append(self._outside_compilation_cluster + ':' + str(device.device_index))
        self._oc_dev_fn_stack = graph._device_function_stack
        graph._device_function_stack = self._outer_device_function_stack

    def _ExitOutsideCompilationScope(self):
        if False:
            while True:
                i = 10
        if not self._outside_compilation_cluster:
            raise ValueError('Attempted to exit outside_compilation scope when not in scope')
        self._outside_compilation_cluster = None
        self._is_map_outside_compilation = False
        graph = ops.get_default_graph()
        graph._device_function_stack = self._oc_dev_fn_stack

    def Enter(self) -> None:
        if False:
            return 10
        if not self._outer_device_function_stack:
            graph = ops.get_default_graph()
            self._outer_device_function_stack = graph._device_function_stack.copy()
        super(TPUReplicateContext, self).Enter()

    def HostComputeCore(self) -> List[Text]:
        if False:
            while True:
                i = 10
        return self._host_compute_core

    def _RemoveExternalControlEdges(self, op: ops.Operation) -> Tuple[List[ops.Operation], List[ops.Operation]]:
        if False:
            return 10
        'Remove any external control dependency on this op.'
        internal_control_inputs = []
        external_control_inputs = []
        for x in op.control_inputs:
            is_internal_op = False
            ctxt = x._get_control_flow_context()
            while ctxt is not None:
                if ctxt == self:
                    is_internal_op = True
                    break
                ctxt = ctxt._outer_context
            if is_internal_op:
                internal_control_inputs.append(x)
            else:
                external_control_inputs.append(x)
        op._remove_all_control_inputs()
        op._add_control_inputs(internal_control_inputs)
        return (internal_control_inputs, external_control_inputs)

    def AddOp(self, op: ops.Operation) -> None:
        if False:
            return 10
        if op.type in _DENYLISTED_OPS:
            logging.error('Operation of type %s (%s) is not supported on the TPU. Execution will fail if this op is used in the graph. ', op.type, op.name)
        if op.type in _UNSUPPORTED_OPS:
            self._unsupported_ops.append(op)
        if any((x.dtype._is_ref_dtype for x in op.inputs)):
            raise NotImplementedError(f'Non-resource Variables are not supported inside TPU computations (operator name: {op.name})')
        if _TPU_REPLICATE_ATTR in op.node_def.attr and '_cloned' not in op.node_def.attr:
            raise ValueError(f'TPU computations cannot be nested on op ({op})')
        op._set_attr(_TPU_REPLICATE_ATTR, self._tpu_replicate_attr)
        if self._outside_compilation_cluster:
            op._set_attr(_OUTSIDE_COMPILATION_ATTR, attr_value_pb2.AttrValue(s=compat.as_bytes(self._outside_compilation_cluster)))
        if self._is_map_outside_compilation:
            op._set_attr(_MAP_OUTSIDE_COMPILATION_ATTR, attr_value_pb2.AttrValue(b=True))
        if self._num_replicas > 1 or not self._outside_compilation_cluster:
            op.graph.prevent_feeding(op)
            op.graph.prevent_fetching(op)
        (internal_control_inputs, external_control_inputs) = self._RemoveExternalControlEdges(op)
        if not op.inputs:
            if not internal_control_inputs:
                op._add_control_input(self.GetControlPivot())
        else:
            for index in range(len(op.inputs)):
                x = op.inputs[index]
                real_x = self.AddValue(x)
                if real_x is not x:
                    op._update_input(index, real_x)
        if external_control_inputs:
            with ops.control_dependencies(None):
                self.Enter()
                external_control_inputs = [array_ops.identity(x.outputs[0]).op for x in external_control_inputs if x.outputs]
                self.Exit()
            op._add_control_inputs(external_control_inputs)
        output_names = [x.name for x in op.outputs]
        context = self
        while context is not None:
            context._values.update(output_names)
            context = context._outer_context
        if self._outer_context:
            self._outer_context.AddInnerOp(op)

    def AddValue(self, val: core_types.Tensor) -> core_types.Tensor:
        if False:
            return 10
        'Add `val` to the current context and its outer context recursively.'
        if not self._outer_context:
            return val
        if val.name in self._values:
            result = self._external_values.get(val.name)
            return val if result is None else result
        result = val
        self._values.add(val.name)
        if self._outer_context:
            result = self._outer_context.AddValue(val)
            self._values.add(result.name)
        self._external_values[val.name] = result
        return result

    def AddInnerOp(self, op: ops.Operation):
        if False:
            for i in range(10):
                print('nop')
        self.AddOp(op)
        if self._outer_context:
            self._outer_context.AddInnerOp(op)

    @property
    def grad_state(self):
        if False:
            print('Hello World!')
        return None

    @property
    def back_prop(self):
        if False:
            i = 10
            return i + 15
        'Forwards to the enclosing while context, if any.'
        if self.GetWhileContext():
            return self.GetWhileContext().back_prop
        return False

    def GetControlPivot(self) -> ops.Operation:
        if False:
            print('Hello World!')
        return self._pivot

    def RequiresUniqueFunctionRetracing(self):
        if False:
            i = 10
            return i + 15
        return True

def _enclosing_tpu_context_and_graph() -> Tuple[Any, Any]:
    if False:
        print('Hello World!')
    'Returns the TPUReplicateContext and its associated graph.'
    graph = ops.get_default_graph()
    while graph is not None:
        context_ = graph._get_control_flow_context()
        while context_ is not None:
            if isinstance(context_, TPUReplicateContext):
                return (context_, graph)
            context_ = context_.outer_context
        graph = getattr(graph, 'outer_graph', None)
    raise ValueError("get_replicated_var_handle() called without TPUReplicateContext. This shouldn't happen. Please file a bug.")

class OutsideCompilationV2Context(control_flow_ops.ControlFlowContext):
    """The context for outside compilation in Tensorflow 2.0.

  Every op added in this context will be assigned an _xla_outside_compilation
  attribute.
  """

    def __init__(self, name: Text, is_map_outside_compilation=False):
        if False:
            while True:
                i = 10
        control_flow_ops.ControlFlowContext.__init__(self)
        self._name = name
        self._is_map_outside_compilation = is_map_outside_compilation

    def AddOp(self, op: ops.Operation) -> None:
        if False:
            while True:
                i = 10
        if self._outer_context:
            self._outer_context.AddOp(op)
        self._set_outside_compilation_attributes(op)

    def AddInnerOp(self, op: ops.Operation) -> None:
        if False:
            i = 10
            return i + 15
        if self._outer_context:
            self._outer_context.AddInnerOp(op)
        self._set_outside_compilation_attributes(op)

    def to_control_flow_context_def(self, context_def, export_scope=None):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def _set_outside_compilation_attributes(self, op: ops.Operation) -> None:
        if False:
            for i in range(10):
                print('nop')
        op._set_attr(_OUTSIDE_COMPILATION_ATTR, attr_value_pb2.AttrValue(s=compat.as_bytes(self._name)))
        if self._is_map_outside_compilation:
            op._set_attr(_MAP_OUTSIDE_COMPILATION_ATTR, attr_value_pb2.AttrValue(b=True))

def outside_compilation_impl(is_map, computation: Callable[..., Any], *args, **kwargs) -> Any:
    if False:
        while True:
            i = 10
    'Tags ops in `computation` with outside compilation attributes for ordinary `outside_compilation` or `map_outside_compilation`.'
    args = [] if args is None else args
    graph = ops.get_default_graph()
    if isinstance(graph, func_graph.FuncGraph):
        try:
            (tpu_context, _) = _enclosing_tpu_context_and_graph()
        except ValueError:
            logging.warning('Outside compilation attempted outside TPUReplicateContext scope. As no enclosing TPUReplicateContext can be found, returning the result of `computation` as is.')
            return computation(*args, **kwargs)
        outside_compilation_name = str(tpu_context._outside_compilation_counter)
        tpu_context._outside_compilation_counter = tpu_context._outside_compilation_counter + 1
        outside_compilation_context = OutsideCompilationV2Context(outside_compilation_name, is_map_outside_compilation=is_map)
        outside_compilation_context.Enter()
        args = [] if args is None else args
        retval = computation(*args, **kwargs)
        outside_compilation_context.Exit()
        return retval
    initial_context = graph._get_control_flow_context()
    context = initial_context
    while context:
        if isinstance(context, TPUReplicateContext):
            context._EnterOutsideCompilationScope(is_map_outside_compilation=is_map)
        context = context.outer_context
    retval = computation(*args, **kwargs)
    final_context = graph._get_control_flow_context()
    if initial_context is not final_context:
        raise NotImplementedError('Control-flow context cannot be different at start and end of an outside_compilation scope')
    context = initial_context
    while context:
        if isinstance(context, TPUReplicateContext):
            context._ExitOutsideCompilationScope()
        context = context.outer_context
    return retval

@tf_export(v1=['tpu.outside_compilation'])
def outside_compilation(computation: Callable[..., Any], *args, **kwargs) -> Any:
    if False:
        i = 10
        return i + 15
    "Builds part of a computation outside any current TPU replicate scope.\n\n  `tf.tpu.outside_compilation()` is used to run ops in `computation` on CPU\n  instead of running on TPU. For example, users can run ops that are not\n  supported on TPU's (e.g. tf.summary.write()) by explicitly placing those\n  ops on CPU's. Below usage of outside compilation will place ops in\n  `computation_with_string_ops` on CPU.\n\n  Example usage:\n\n  ```python\n  def computation_with_string_ops(x):\n    # strings types are not supported on TPU's and below ops must\n    # run on CPU instead.\n    output = tf.strings.format('1{}', x)\n    return tf.strings.to_number(output)\n\n  def tpu_computation():\n    # Expected output is 11.\n    output = tf.tpu.outside_compilation(computation_with_string_ops, 1)\n  ```\n\n  Outside compilation should be called inside TPUReplicateContext. That is,\n  `tf.tpu.outside_compilation()` should be called inside a function that is\n  passed to `tpu.split_compile_and_replicate()` -- this is implied when\n  outside compilation is invoked inside a function passed to TPUStrategy\n  `run()`. If invoked outside of TPUReplicateContext,\n  then this simply returns the result of `computation`, and therefore,\n  would be a no-op. Note that outside compilation is different from\n  `tf.distribute.experimental.TPUStrategy.merge_call()` as logic in\n  outside compilation is replicated and executed separately for each\n  replica. On the other hand, `merge_call()` requires a `merge_fn`\n  to aggregate the inputs from different replicas and is executed only\n  once.\n\n  For variables placed in TPU device, which includes variables created inside\n  TPUStrategy scope, outside compilation logic must not include variable\n  read/write. For variables placed on host, which is the case when variables\n  created via TPUEstimator, variable read/write is only allowed if the variable\n  is not accessed by any other ops in the TPU computation. Variable read/write\n  from outside compilation cluster is not visible from TPU computation and\n  vice versa. Therefore, if outside compilation logic contains such host\n  variables read/write ops and if the variables are accessed by TPU\n  computation as well, then this may lead to deadlock.\n\n  Internally, `tf.tpu.outside_compilation()` adds outside compilation\n  attributes to all ops in `computation`. During a later passes ops with outside\n  compilation attributes are moved to a host-side graph. Inputs to this extract\n  host-side graph are sent from TPU computation graph to host graph via a pair\n  of XlaSendToHost and XlaRecvFromHost ops. Note that using\n  `tf.tpu.outside_compilation()` may result in tensor transfer between TPU and\n  CPU, leading to non-trivial performance impact.\n\n  Args:\n    computation: A Python function that builds the computation to place on the\n      host.\n    *args: the positional arguments for the computation.\n    **kwargs: the keyword arguments for the computation.\n\n  Returns:\n    The Tensors returned by computation.\n  "
    return outside_compilation_impl(False, computation, *args, **kwargs)

def experimental_map_outside_compilation(computation: Callable[..., Any], *args, **kwargs) -> Any:
    if False:
        while True:
            i = 10
    "Maps `computation` onto shards and puts it outside any current TPU replicate scope.\n\n  `experimental_map_outside_compilation(f, x)` maps `f` onto the shards\n  of `x`, where `x` is split-sharded. Each invocation of `f` on a split occurs\n  on the CPU that's associated with the TPU that owns the split.\n\n  Example usage:\n\n  ```python\n  def normalize_each_split(split):\n    return split - tf.math.reduce_mean(split)\n\n  def tpu_computation(x):\n    x_split = strategy.experimental_split_to_logical_devices(\n                x, [num_cores_per_replica, 1])\n    y = experimental_map_outside_compilation(\n          normalize_each_split, x_split)\n    y_split = strategy.experimental_split_to_logical_devices(\n                x, [num_cores_per_replica, 1])\n    return y_split\n  ```\n\n  `experimental_map_outside_compilation` should be called inside\n  TPUReplicateContext. That is, `outside_compilation()` should be called\n  inside a function that is passed to `tpu.split_compile_and_replicate()` --\n  this is implied when outside compilation is invoked inside a function passed\n  to TPUStrategy `run()`. It is invalid to invoke outside of\n  TPUReplicateContext.\n\n  `experimental_map_outside_compilation` should input and output tensors that\n  are located on the TPU.\n\n  Internally, `experimental_map_outside_compilation()` adds outside\n  compilation attributes to all ops in `computation` and moves outside-compiled\n  ops to a host-side graph. This is similar to `tf.tpu.outside_compilation()`.\n  Send/recv ops from/to the TPU send each split directly to the TPU's host.\n\n  Args:\n    computation: A Python function that builds the computation to place on the\n      host.\n    *args: the positional arguments for the computation.\n    **kwargs: the keyword arguments for the computation.\n\n  Returns:\n    The Tensors returned by computation.\n  "
    return outside_compilation_impl(True, computation, *args, **kwargs)