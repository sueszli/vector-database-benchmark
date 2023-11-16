"""Utilities for cross_device_ops."""
import copy
import threading
from typing import Callable, List, Optional, Union
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import values as value_lib
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nccl_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import core
INSTANCE_KEY_START_NUMBER = 100

def aggregate_gradients_using_nccl(replica_grads):
    if False:
        while True:
            i = 10
    'Aggregate gradients using nccl allreduce.'
    agg_all_g_and_v = []
    for single_g_and_v in zip(*replica_grads):
        single_grads = [g for (g, _) in single_g_and_v]
        agg_grads = nccl_ops.all_sum(single_grads)
        agg_all_g_and_v.append([(g, v) for (g, (_, v)) in zip(agg_grads, single_g_and_v)])
    agg_all_g_and_v = list(zip(*agg_all_g_and_v))
    return agg_all_g_and_v

def aggregate_gradients_using_hierarchical_copy(avail_devices, replica_grads):
    if False:
        return 10
    'Aggregate gradients using hierarchical copies.\n\n  Args:\n    avail_devices: available GPU devices.\n    replica_grads: List of lists of (gradient, variable) tuples. The outer list\n      is over replicas. The inner list is over individual gradients.\n\n  Returns:\n    The list of (aggregated_gradient, variable), where the gradient has been\n      summed across all replicas and the variable is chosen from the first\n      replica.\n  '
    agg_grads = []
    num_devices = len(avail_devices)
    group_size = num_devices // 2
    for (i, single_grads) in enumerate(zip(*replica_grads)):
        group_0_main_device = i % num_devices
        group_1_main_device = (group_0_main_device + group_size) % num_devices
        if group_0_main_device < group_size:
            group_0_begin = 0
            group_1_begin = group_size
        else:
            group_0_begin = group_size
            group_1_begin = 0
        group_0_device_grads = single_grads[group_0_begin:group_0_begin + group_size]
        with ops.device(avail_devices[group_0_main_device]):
            (group_0_agg_grads, _) = aggregate_single_gradient_using_copy(group_0_device_grads, False, False)
        group_1_device_grads = single_grads[group_1_begin:group_1_begin + group_size]
        with ops.device(avail_devices[group_1_main_device]):
            (group_1_agg_grads, _) = aggregate_single_gradient_using_copy(group_1_device_grads, False, False)
        with ops.device(avail_devices[group_0_main_device]):
            ((agg_total_grads, _), _) = aggregate_single_gradient_using_copy([group_0_agg_grads, group_1_agg_grads], False, False)
        with ops.device(avail_devices[group_0_main_device]):
            group_0_agg_grads_bcast = array_ops.identity(agg_total_grads)
        with ops.device(avail_devices[group_1_main_device]):
            group_1_agg_grads_bcast = array_ops.identity(agg_total_grads)
        agg_grads_bcast = []
        for j in range(len(single_grads)):
            with ops.device(avail_devices[j]):
                if (group_0_main_device < group_size) == (j < group_size):
                    src_device_grad = group_0_agg_grads_bcast
                else:
                    src_device_grad = group_1_agg_grads_bcast
                agg_grads_bcast.append(array_ops.identity(src_device_grad))
        agg_grads.append([(g, v) for (g, (_, v)) in zip(agg_grads_bcast, single_grads)])
    agg_grads = list(zip(*agg_grads))
    return agg_grads

def aggregate_single_gradient_using_copy(grad_and_vars, use_mean, check_inf_nan):
    if False:
        return 10
    'Calculate the average gradient for a shared variable across all replicas.\n\n  Note that this function provides a synchronization point across all replicas.\n\n  Args:\n    grad_and_vars: A list or tuple of (gradient, variable) tuples. Each\n      (gradient, variable) pair within the outer list represents the gradient\n      of the variable calculated for a single replica, and the number of pairs\n      equals the number of replicas.\n    use_mean: if True, mean is taken, else sum of gradients is taken.\n    check_inf_nan: check grads for nans and infs.\n\n  Returns:\n    The tuple ([(average_gradient, variable),], has_nan_or_inf) where the\n      gradient has been averaged across all replicas. The variable is chosen\n      from the first replica. The has_nan_or_inf indicates the grads has nan or\n      inf.\n  '
    grads = [g for (g, _) in grad_and_vars]
    grad = math_ops.add_n(grads)
    if use_mean and len(grads) > 1:
        grad = array_ops.multiply(grad, 1.0 / len(grads))
    v = grad_and_vars[0][1]
    if check_inf_nan:
        has_nan_or_inf = array_ops.logical_not(array_ops.reduce_all(array_ops.is_finite(grads)))
        return ((grad, v), has_nan_or_inf)
    else:
        return ((grad, v), None)

class CollectiveKeys(object):
    """Class that manages collective keys.

  We need to manage three different keys for collective:

  *Group key*: an integer key to identify the set of cooperative devices.
  Collective ops work under the same set of devices must using the same group
  key.

  *Instance key*: an integer key to identify the set of same counterpart of
  tensors on different devices in a device group that need to be all-reduced.

  This class is thread safe.
  """

    def __init__(self, group_key_start=1):
        if False:
            while True:
                i = 10
        'Initializes the object.\n\n    Args:\n      group_key_start: the starting integer of group key.\n    '
        self._group_key = group_key_start
        self._instance_key_table = {}
        self._lock = threading.Lock()
        self._known_groups = {}

    def get_group_key(self, devices):
        if False:
            while True:
                i = 10
        'Returns a group key for the list of local devices.\n\n    The same group key is returned if the list of local devices is the same.\n\n    Args:\n      devices: a list of local canonical device strings in a collective group.\n\n    Returns:\n      a group key.\n    '
        with self._lock:
            devices_key = ','.join(devices)
            if devices_key not in self._known_groups:
                self._known_groups[devices_key] = self._get_new_group_key(devices)
            return self._known_groups[devices_key]

    def _get_new_group_key(self, devices):
        if False:
            for i in range(10):
                print('nop')
        'Returns a new group key.\n\n    The caller should store and reuse the same group key for the same set of\n    devices. Calling this method always returns a new group key.\n\n    This method is not thread-safe.\n\n    Args:\n      devices: a list of canonical device strings in a collective group.\n\n    Returns:\n      a new group key.\n    '
        new_key = self._group_key
        self._group_key += 1
        self._instance_key_table[new_key] = {}
        for device in devices:
            self._instance_key_table[new_key][device] = INSTANCE_KEY_START_NUMBER
        return new_key

    def get_instance_key(self, group_key, device):
        if False:
            while True:
                i = 10
        'Returns a new instance key for use in defining a collective op.\n\n    You should call this once per each collective op of a collective instance.\n\n    Args:\n      group_key: the group key returned by get_group_key(). You should not\n        assign the group key yourself.\n      device: a canonical device string. It should be the device this collective\n        op is on.\n\n    Returns:\n      a new instance key.\n\n    Raises:\n      ValueError: when the group key is invalid or the device is not in the\n      group.\n    '
        with self._lock:
            group = self._instance_key_table.get(group_key, None)
            if group is None:
                raise ValueError(f'Group {group_key} is not found.')
            if device not in group:
                raise ValueError(f'Device {device} is not present in group {group_key}')
            v = group[device]
            group[device] += 1
            return v

    def __deepcopy__(self, memo):
        if False:
            while True:
                i = 10
        copied = CollectiveKeys()
        copied._group_key = self._group_key
        copied._instance_key_table = copy.deepcopy(self._instance_key_table, memo)
        return copied

class CollectiveReplicaLauncher(object):
    """Launch collectives on one replica."""
    _prefer_unique_instance_key = True
    _prefer_ordering_token = True

    def __init__(self, group_key: int, group_size: int, collective_keys: CollectiveKeys, device: str, options: collective_util.Options):
        if False:
            i = 10
            return i + 15
        self._group_key = group_key
        self._group_size = group_size
        self._collective_keys = collective_keys
        self._device = device
        self._options = options
        if self._use_ordering_token():
            with ops.init_scope(), ops.device(device):
                self._ordering_token = resource_variable_ops.ResourceVariable(0.0)
        else:
            self._ordering_token = None

    def _control_input(self, control_input: Union[core.TensorLike, ops.Operation]):
        if False:
            while True:
                i = 10
        if control_input is not None and (not self._use_ordering_token()):
            return ops.control_dependencies([control_input])
        return ops.NullContextmanager()

    def _use_unique_instance_key(self):
        if False:
            for i in range(10):
                print('nop')
        if not ops.executing_eagerly_outside_functions():
            return False
        return CollectiveReplicaLauncher._prefer_unique_instance_key

    def _use_ordering_token(self):
        if False:
            print('Hello World!')
        if not ops.executing_eagerly_outside_functions():
            return False
        return CollectiveReplicaLauncher._prefer_ordering_token

    def _next_instance_key(self):
        if False:
            i = 10
            return i + 15
        'Returns the next instance key.'
        if self._use_unique_instance_key():
            graph = ops.get_default_graph()
            while getattr(graph, 'is_control_flow_graph', False):
                graph = graph.outer_graph
            if not context.executing_eagerly() and graph.building_function:
                with graph.as_default():
                    return graph.capture_call_time_value(self._next_instance_key, tensor_spec.TensorSpec([], dtypes.int32))
            else:
                instance_key = self._collective_keys.get_instance_key(self._group_key, self._device)
                with ops.device('CPU:0'):
                    return ops.convert_to_tensor(instance_key, dtype=dtypes.int32)
        else:
            return self._collective_keys.get_instance_key(self._group_key, self._device)

    def _get_ordering_token(self):
        if False:
            while True:
                i = 10
        if self._use_ordering_token():
            return self._ordering_token.handle

    def can_order_nccl(self):
        if False:
            print('Hello World!')
        'Whether this launcher can order NCCL operations.'
        return self._use_ordering_token()

    def all_reduce(self, input_tensor: core.TensorLike, control_input: Optional[Union[core.TensorLike, ops.Operation]]=None, options: Optional[collective_util.Options]=None) -> core.Tensor:
        if False:
            for i in range(10):
                print('nop')
        'All-reduce a dense tensor.\n\n    Args:\n      input_tensor: a dense tensor. It must have the same shape on all replicas.\n      control_input: if not None, add control edges between control_input and\n        the all-reduce.\n      options: an optional tf.distribute.experimental.CommunicationOptions. If\n        provided, it overrides the default options.\n\n    Returns:\n      The reduced tensor.\n    '
        instance_key = self._next_instance_key()
        options = self._options.merge(options)
        ordering_token = self._get_ordering_token()
        with ops.device(self._device), self._control_input(control_input):
            return collective_ops.all_reduce_v2(input_tensor, self._group_size, self._group_key, instance_key, communication_hint=options.implementation.value, timeout=options.timeout_seconds, ordering_token=ordering_token)

    def _all_gather(self, input_tensor: core.TensorLike, options: Optional[collective_util.Options]) -> core.Tensor:
        if False:
            return 10
        'All-gather a dense tensor.\n\n    Args:\n      input_tensor: a dense tensor. It must have the same shape on all replicas.\n      options: an optional tf.distribute.experimental.CommunicationOptions. If\n        provided, it overrides the default options.\n\n    Returns:\n      The reduced tensor.\n    '
        instance_key = self._next_instance_key()
        options = self._options.merge(options)
        ordering_token = self._get_ordering_token()
        with ops.device(self._device):
            return collective_ops.all_gather_v2(input_tensor, self._group_size, self._group_key, instance_key, communication_hint=options.implementation.value, timeout=options.timeout_seconds, ordering_token=ordering_token)

    def batch_all_reduce(self, input_tensor_packs: List[List[core.TensorLike]], options: Optional[collective_util.Options]=None) -> core.Tensor:
        if False:
            while True:
                i = 10
        "Batch all-reduce dense tensors.\n\n    This takes a list of batches of tensors. Using multiple batches have the\n    benefit that it doesn't need to wait for all inputs to be ready to start the\n    all-reduce.\n\n    Args:\n      input_tensor_packs: a list of lists of dense tensors.\n      options: an optional tf.distribute.experimental.CommunicationOptions. If\n        provided, it overrides the default options.\n\n    Returns:\n      A flat list of reduced tensors.\n    "
        options = self._options.merge(options)
        outputs = []
        for pack in input_tensor_packs:
            if context.executing_eagerly():
                for input_tensor in pack:
                    outputs.append(self.all_reduce(input_tensor, None, options))
            else:
                with ops.device(self._device):
                    flat_tensors = [array_ops.reshape(t, [-1]) for t in pack]
                    shapes = [array_ops.shape(t) for t in pack]
                    if options.implementation == collective_util.CommunicationImplementation.NCCL and outputs:
                        control_input = outputs[-1]
                    else:
                        control_input = None
                    reduced = self.all_reduce(array_ops.concat(flat_tensors, axis=0), control_input, options)
                    num_elements = [math_ops.reduce_prod(s) for s in shapes]
                    flat_outputs = array_ops.split(reduced, num_elements, axis=0)
                    for (shape, flat_output) in zip(shapes, flat_outputs):
                        outputs.append(array_ops.reshape(flat_output, shape))
        return outputs

    def all_gather(self, input_tensor: core.TensorLike, axis: core.TensorLike, options: Optional[collective_util.Options]=None) -> core.Tensor:
        if False:
            print('Hello World!')
        'All-gather a dense tensor.\n\n    This method must be called inside a tf.function.\n\n    Args:\n      input_tensor: a dense tensor. It must have the same rank on all replicas,\n        and dimensions other than `axis` need to be the same as well.\n      axis: 0-D int32 Tensor. Dimension along which to gather. Must be in the\n        range [0, rank(value)).\n      options: an optional tf.distribute.experimental.CommunicationOptions. If\n        provided, it overrides the default options.\n\n    Returns:\n      The gathered Tensor.\n\n    Raises:\n      RuntimeError: if called in eager mode.\n    '
        if context.executing_eagerly():
            raise RuntimeError('all_gather is not supported in eager mode.')
        with ops.device(self._device), ops.control_dependencies([array_ops.identity(input_tensor)]):
            perm_pre = array_ops.concat(([axis], math_ops.range(axis), math_ops.range(axis + 1, array_ops.rank(input_tensor))), axis=0)
            input_tensor_t = array_ops.transpose(input_tensor, perm=perm_pre)
            gathered_shape = self._all_gather(array_ops.expand_dims_v2(array_ops.shape_v2(input_tensor_t), axis=0), options)
            first_dims = gathered_shape[:, 0]
            full_axis_dim = math_ops.reduce_max(first_dims)
            padded_input_tensor = _pad_util(input_tensor_t, full_axis_dim)
            gather_padded_out_tensor = self._all_gather(padded_input_tensor, options)
            split_tensors = []
            for i in range(self._group_size):
                start_pos = i * full_axis_dim
                split_tensors.append(gather_padded_out_tensor[start_pos:start_pos + first_dims[i]])
            out_tensor_t = array_ops.concat(split_tensors, 0)
            perm_after = array_ops.concat((math_ops.range(1, axis + 1), [0], math_ops.range(axis + 1, array_ops.rank(input_tensor_t))), axis=0)
            return array_ops.transpose(out_tensor_t, perm=perm_after)

    def all_reduce_indexed_slices(self, input_slices: indexed_slices.IndexedSlices, options: Optional[collective_util.Options]=None) -> indexed_slices.IndexedSlices:
        if False:
            for i in range(10):
                print('nop')
        'All-reduce an IndexedSlices.\n\n    This method can be called outside  tf.function.\n\n    Args:\n      input_slices: an IndexedSlices.\n      options: an optional tf.distribute.experimental.CommunicationOptions. If\n        provided, it overrides the default options.\n\n    Returns:\n      The reduced IndexedSlices.\n    '
        options = self._options.merge(options)
        with ops.device(self._device):

            def all_gather_indexed_slices(all_gather_fn: Callable[[core.TensorLike, Optional[collective_util.Options]], core.Tensor]) -> indexed_slices.IndexedSlices:
                if False:
                    while True:
                        i = 10
                'Use all_gather_fn to aggregate `IndexedSlices`.'
                all_values = all_gather_fn(input_slices.values, options)
                if options.implementation == collective_util.CommunicationImplementation.NCCL:
                    control = [all_values]
                else:
                    control = []
                with ops.control_dependencies(control):
                    all_indices = all_gather_fn(input_slices.indices, options)
                return indexed_slices.IndexedSlices(values=all_values, indices=all_indices, dense_shape=input_slices.dense_shape)
            length = array_ops.shape(input_slices.indices)
            all_lengths = self._all_gather(length, options)

            def all_gather_with_padding(input_tensor: core.TensorLike, options: Optional[collective_util.Options]) -> core.Tensor:
                if False:
                    return 10
                'all_gather tensors of different sizes using padding.'
                max_length = math_ops.reduce_max(all_lengths)
                padded_tensor = _pad_util(input_tensor, max_length)
                all_padded_tensors = self._all_gather(padded_tensor, options)
                split_tensors = []
                for i in range(self._group_size):
                    start_pos = i * max_length
                    split_tensors.append(all_padded_tensors[start_pos:start_pos + all_lengths[i]])
                return array_ops.concat(split_tensors, 0)
            return cond.cond(math_ops.equal(math_ops.reduce_max(all_lengths), math_ops.reduce_min(all_lengths)), lambda : all_gather_indexed_slices(self._all_gather), lambda : all_gather_indexed_slices(all_gather_with_padding))

def aggregate_tensors_or_indexed_slices(values, accumulation_fn=math_ops.add_n):
    if False:
        for i in range(10):
            print('nop')
    'Aggregate tensors using `accumulation_fn` and IndexedSlices via concat.'
    if any((isinstance(v, indexed_slices.IndexedSlices) for v in values)):
        return backprop_util.AggregateIndexedSlicesGradients(values)
    else:
        return accumulation_fn(values)

def divide_by_n_tensors_or_indexed_slices(value, n):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(value, indexed_slices.IndexedSlices):
        value = backprop_util.FlattenNestedIndexedSlices(value)
        return indexed_slices.IndexedSlices(value.values / n, value.indices, value.dense_shape)
    else:
        return value / n

def copy_tensor_or_indexed_slices_to_device(value, device):
    if False:
        while True:
            i = 10
    'Copies a tensor or IndexedSlices to a device.'
    with ops.device(device):
        if isinstance(value, indexed_slices.IndexedSlices):
            copied_values = array_ops.identity(value.values)
            copied_indices = array_ops.identity(value.indices)
            if value.dense_shape is not None:
                copied_shape = array_ops.identity(value.dense_shape)
            else:
                copied_shape = None
            result = indexed_slices.IndexedSlices(copied_values, copied_indices, copied_shape)
        else:
            result = array_ops.identity(value)
    return result

def is_indexed_slices(value):
    if False:
        while True:
            i = 10
    if isinstance(value, indexed_slices.IndexedSlices):
        return True
    if isinstance(value, value_lib.DistributedValues):
        return all((isinstance(v, indexed_slices.IndexedSlices) for v in value.values))
    return False

def split_by_sparsity(values):
    if False:
        i = 10
        return i + 15
    'Split values into dense and sparse values.\n\n  Args:\n    values: a list of tensors or `PerReplica`s.\n\n  Returns:\n    Four lists:\n      a list of dense values, a list of their indices in `values` and\n      a list of sparse values, a list of their indices in `values`.\n  '
    dense_values = []
    dense_indices = []
    sparse_values = []
    sparse_indices = []
    for (i, v) in enumerate(values):
        if is_indexed_slices(v):
            sparse_values.append(v)
            sparse_indices.append(i)
        else:
            dense_values.append(v)
            dense_indices.append(i)
    return (dense_values, dense_indices, sparse_values, sparse_indices)

def stitch_values(values_and_indices_list):
    if False:
        print('Hello World!')
    'Stitch values together according to their indices.\n\n  Args:\n    values_and_indices_list: a list of tuples of values and indices indicating\n      the values and positions in the returned list.\n\n  Returns:\n    a stitched list of values.\n  '
    length = 0
    for values_and_indices in values_and_indices_list:
        length += len(values_and_indices[0])
    result = [None] * length
    for values_and_indices in values_and_indices_list:
        if values_and_indices and values_and_indices[0]:
            for (v, i) in zip(*values_and_indices):
                assert result[i] is None
                result[i] = v
    return result

def group_by_size(input_tensors, bytes_per_pack):
    if False:
        print('Hello World!')
    'Groups `input_tensors` into chunks of `bytes_per_pack`.\n\n  The method preserves the original order of `input_tensors`. The grouping is\n  best effort, each pack could have more or less bytes than `bytes_per_pack`.\n  It only groups values with known shape.\n\n  Args:\n    input_tensors: a list of Tensor.\n    bytes_per_pack: an integer.\n\n  Returns:\n    A list of packs of Tensor. All values are grouped into one pack if\n    `bytes_per_pack` is zero or any of the value has unknown shape.\n  '
    if bytes_per_pack == 0:
        return [input_tensors]
    packs = []
    last_pack_size = 0
    for value in input_tensors:
        num_elements = value.shape.num_elements()
        if num_elements is None:
            logging.warning('not packing values due to the unknown or inconsistent shape of %s', value)
            return [input_tensors]
        size = num_elements * value.dtype.size
        if not packs or last_pack_size > bytes_per_pack:
            packs.append([])
            last_pack_size = 0
        packs[-1].append(value)
        last_pack_size += size
    return packs

def _pad_util(input_tensor, full_axis_dim):
    if False:
        for i in range(10):
            print('nop')
    "Pad the `input_tensor`'s first dimension to be `full_axis_dim`."
    missing_axis_dim = full_axis_dim - array_ops.shape_v2(input_tensor)[0]
    tensor_rank = array_ops.rank(input_tensor)
    paddings_axis = [[0, missing_axis_dim]]
    paddings = array_ops.concat([paddings_axis, array_ops.zeros(shape=(tensor_rank - 1, 2), dtype=dtypes.int32)], axis=0)
    padded_input_tensor = array_ops.pad(input_tensor, paddings)
    return padded_input_tensor