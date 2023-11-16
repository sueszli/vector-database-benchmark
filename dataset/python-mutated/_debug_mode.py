import inspect
import math
import traceback
import warnings
from queue import Queue
import nvidia.dali.backend as _b
import nvidia.dali.ops as _ops
import nvidia.dali.pipeline as _pipeline
import nvidia.dali.tensors as _tensors
import nvidia.dali.types as _types
from nvidia.dali import _conditionals
from nvidia.dali._utils.eager_utils import _Classification, _transform_data_to_tensorlist
from nvidia.dali.data_node import DataNode as _DataNode, _check
from nvidia.dali.fn import _to_snake_case
from nvidia.dali._utils.external_source_impl import get_callback_from_source as _get_callback_from_source, accepted_arg_count as _accepted_arg_count
from nvidia.dali.ops._operator_utils import _build_input_sets, _repack_output_sets

class DataNodeDebug(_DataNode):
    """Wrapper class around Tensor, implementing all of the DataNode attributes."""

    def __init__(self, data, name, device, source):
        if False:
            i = 10
            return i + 15
        super().__init__(name, device, source)
        self._data = data

    def __str__(self):
        if False:
            while True:
                i = 10
        indent = ' ' * 4
        return f'DataNodeDebug(\n{indent}name="{self.name}",\n{indent}data=' + f"{_tensors._tensorlist_to_string(self._data, indent + ' ' * 5)})"
    __repr__ = __str__

    def gpu(self):
        if False:
            i = 10
            return i + 15
        if _conditionals.conditionals_enabled():
            ([self_split], _) = _conditionals.apply_conditional_split_to_args([self], {})
            data = self_split._data._as_gpu() if self.device == 'cpu' else self_split._data
            gpu_node = DataNodeDebug(data, self_split.name, 'gpu', self_split.source)
            _conditionals.register_data_nodes(gpu_node, [self])
            return gpu_node
        if self.device == 'gpu':
            return self
        return DataNodeDebug(self._data._as_gpu(), self.name, 'gpu', self.source)

    def get(self):
        if False:
            print('Hello World!')
        return self._data

    def shape(self):
        if False:
            while True:
                i = 10
        return self._data.shape()

    @staticmethod
    def _arithm_op(*inputs, name=None):
        if False:
            for i in range(10):
                print('nop')
        return _PipelineDebug.current()._wrap_op_call(_ops.ArithmeticGenericOp, DataNodeDebug._aritm_op_name, *inputs, name=name)

    def __add__(self, other):
        if False:
            print('Hello World!')
        return self._arithm_op(self, other, name='add')

    def __radd__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self._arithm_op(other, self, name='add')

    def __sub__(self, other):
        if False:
            print('Hello World!')
        return self._arithm_op(self, other, name='sub')

    def __rsub__(self, other):
        if False:
            return 10
        return self._arithm_op(other, self, name='sub')

    def __mul__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self._arithm_op(self, other, name='mul')

    def __rmul__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self._arithm_op(other, self, name='mul')

    def __pow__(self, other):
        if False:
            i = 10
            return i + 15
        return self._arithm_op(self, other, name='pow')

    def __rpow__(self, other):
        if False:
            return 10
        return self._arithm_op(other, self, name='pow')

    def __truediv__(self, other):
        if False:
            i = 10
            return i + 15
        return self._arithm_op(self, other, name='fdiv')

    def __rtruediv__(self, other):
        if False:
            i = 10
            return i + 15
        return self._arithm_op(other, self, name='fdiv')

    def __floordiv__(self, other):
        if False:
            i = 10
            return i + 15
        return self._arithm_op(self, other, name='div')

    def __rfloordiv__(self, other):
        if False:
            i = 10
            return i + 15
        return self._arithm_op(other, self, name='div')

    def __neg__(self):
        if False:
            while True:
                i = 10
        return self._arithm_op(self, name='minus')

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return self._arithm_op(self, other, name='eq')

    def __ne__(self, other):
        if False:
            return 10
        return self._arithm_op(self, other, name='neq')

    def __lt__(self, other):
        if False:
            return 10
        return self._arithm_op(self, other, name='lt')

    def __le__(self, other):
        if False:
            while True:
                i = 10
        return self._arithm_op(self, other, name='leq')

    def __gt__(self, other):
        if False:
            return 10
        return self._arithm_op(self, other, name='gt')

    def __ge__(self, other):
        if False:
            while True:
                i = 10
        return self._arithm_op(self, other, name='geq')

    def __and__(self, other):
        if False:
            while True:
                i = 10
        return self._arithm_op(self, other, name='bitand')

    def __rand__(self, other):
        if False:
            i = 10
            return i + 15
        return self._arithm_op(other, self, name='bitand')

    def __or__(self, other):
        if False:
            return 10
        return self._arithm_op(self, other, name='bitor')

    def __ror__(self, other):
        if False:
            while True:
                i = 10
        return self._arithm_op(other, self, name='bitor')

    def __xor__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self._arithm_op(self, other, name='bitxor')

    def __rxor__(self, other):
        if False:
            return 10
        return self._arithm_op(other, self, name='bitxor')
    _aritm_op_name = _to_snake_case(_ops.ArithmeticGenericOp.__name__)

class _ExternalSourceDebug:
    """Debug mode version of ExternalSource operator."""

    def __init__(self, source=None, num_outputs=None, batch_size=-1, cycle=None, name=None, device='cpu', device_id=-1, layout=None, batch=None, batch_info=None, **kwargs):
        if False:
            return 10
        if name is not None and num_outputs is not None:
            raise ValueError('`num_outputs` is not compatible with named `ExternalSource`')
        (callback, source_desc) = _get_callback_from_source(source, cycle, batch_info or False)
        self._name = name
        self._layout = layout
        self._num_outputs = num_outputs
        self._batch = batch
        self._batch_size = batch_size
        self._callback = callback
        self._device = device
        self._device_id = device_id
        self._source_desc = source_desc
        self._batch_info = batch_info
        self._current_iter = 0
        self._current_sample = 0
        self._feed_inputs = Queue()
        if callback is not None:
            arg_count = _accepted_arg_count(callback)
            if arg_count not in [0, 1]:
                raise TypeError('External source callback must be a callable with 0 or 1 argument')
            self.accepts_arg = arg_count > 0

    def _callback_args(self, idx_in_batch, epoch_idx):
        if False:
            while True:
                i = 10
        if not self.accepts_arg:
            return ()
        if idx_in_batch is not None:
            arg = _types.SampleInfo(self._current_sample + idx_in_batch, idx_in_batch, self._current_iter, epoch_idx)
        elif self._batch_info:
            arg = _types.BatchInfo(self._current_iter, epoch_idx)
        else:
            arg = self._current_iter
        return (arg,)

    def _get_batch(self, epoch_idx):
        if False:
            for i in range(10):
                print('nop')
        try:
            if self._batch:
                callback_out = self._callback(*self._callback_args(None, epoch_idx))
            else:
                callback_out = [self._callback(*self._callback_args(i, epoch_idx)) for i in range(self._batch_size)]
            self._current_sample += self._batch_size
            self._current_iter += 1
        except StopIteration:
            self._current_iter = 0
            self._current_sample = 0
            raise
        return callback_out

    def _feed_input(self, data, kwargs):
        if False:
            for i in range(10):
                print('nop')
        if self._callback is not None:
            raise RuntimeError(f"Cannot use `feed_input` on the external source '{self._name}' with a `source` argument specified.")
        self._feed_inputs.put((data, kwargs))

    def _fetch(self, epoch_idx):
        if False:
            print('Hello World!')
        'Fetches data from callback or provided with feed_input.'

        def to_data_node_debug(data):
            if False:
                for i in range(10):
                    print('nop')
            data = _transform_data_to_tensorlist(data, self._batch_size, layout, self._device_id)
            if self._device == 'gpu' and isinstance(data, _tensors.TensorListCPU):
                data = data._as_gpu()
            elif self._device == 'cpu' and isinstance(data, _tensors.TensorListGPU):
                data = data.as_cpu()
                warnings.warn('Loading GPU-originated data into CPU ExternalSource operator is discouraged and might be inefficient', Warning)
            return DataNodeDebug(data, self._name, self._device, self._source_desc)
        if self._callback is not None:
            callback_out = self._get_batch(epoch_idx)
            layout = self._layout
            if self._num_outputs is not None:
                raw_data = []
                for idx in range(self._num_outputs):
                    if self._batch:
                        raw_data.append(callback_out[idx])
                    else:
                        raw_data.append([callback_out[i][idx] for i in range(self._batch_size)])
            else:
                raw_data = callback_out
        else:
            (raw_data, feed_input_params) = self._feed_inputs.get()
            layout = feed_input_params.get('layout', self._layout)
        if self._num_outputs is not None:
            return [to_data_node_debug(data) for data in raw_data]
        return to_data_node_debug(raw_data)

class _IterBatchInfo:

    def __init__(self, size, source_context, non_uniform_batch=False):
        if False:
            while True:
                i = 10
        "Track information about the batch size within the iteration.\n\n        Parameters\n        ----------\n        size : int\n            The size of the detected batch size to be maintained in this iteration\n        source_context\n            Source information about where the batch size was detected for better error reporting.\n        non_uniform_batch : bool, optional\n            Should be set to True if pure Split/Merge are used by hand and not as an implementation\n            of conditional operation. In that case there can be varying batch sizes within\n            the pipeline and we don't know how to track them, by default False\n        "
        self._size = size
        self._source_context = source_context
        self._non_uniform_batch = non_uniform_batch

    @property
    def size(self):
        if False:
            return 10
        if self._non_uniform_batch:
            return -1
        if _conditionals.conditionals_enabled():
            cs = _conditionals.this_condition_stack()
            batch = cs.scope_batch_size_tracker()
            assert batch is None or isinstance(batch, DataNodeDebug), 'Conditionals in debug mode work only with DataNodeDebug'
            if batch is not None:
                return len(batch.get())
        return self._size

    def reset(self):
        if False:
            return 10
        self._size = -1
        self._source_context = None

    def mark_non_uniform_batch(self):
        if False:
            return 10
        'Mark the usage of Split operator that is not a part of the conditional statement\n        (the _if_stmt=True was not passed). It indicates that the batch tracking is no longer\n        possible as we cannot detect the conditional scopes.\n        '
        self._non_uniform_batch = True

    def set_if_empty(self, size, context):
        if False:
            return 10
        if self.size == -1:
            self.__init__(size, context, self._non_uniform_batch)
            return True
        return False

    def check_input(self, other_size, other_context, op_name, input_idx):
        if False:
            return 10
        if not self.set_if_empty(other_size, other_context) and self.size != other_size:
            raise RuntimeError(f"Batch size must be uniform across an iteration. Input {input_idx} for operator '{op_name}' has batch size = {other_size}. Expected batch size = {self.size}from:\n{self._source_context}")

    def check_external_source(self, other_size, other_context, output_idx=-1):
        if False:
            for i in range(10):
                print('nop')
        if not self.set_if_empty(other_size, other_context) and self.size != other_size:
            if self._source_context == other_context and output_idx > 0:
                raise RuntimeError(f'External source must return outputs with consistent batch size. Output {output_idx} has batch size = {other_size}, previous batch size = {self.size}')
            else:
                raise RuntimeError(f'Batch size must be uniform across an iteration. External Source operator returned batch size: {other_size}, expected: {self.size}.\nIf you want to use variable batch size (that is different batch size in each iteration) you must call all the external source operators at the beginning of your debug pipeline, before other DALI operators. All the external source operators are expected to return the same batch size in a given iteration, but it can change between the iterations. Other operators will use that batch size for processing.')

class _OperatorManager:
    """Utility class to manage single operator in the debug mode.

    Uses :class:`ops.Operator` to create OpSpec and handle input sets.
    """

    def __init__(self, op_class, op_name, pipe, source_context, next_logical_id, batch_size, device_id, seed, inputs, kwargs):
        if False:
            return 10
        'Creates direct operator.'
        self._batch_size = batch_size
        self._separate_kwargs(kwargs)
        if op_name == 'arithmetic_generic_op':
            inputs = self._init_arithm_op(kwargs['name'], inputs)
        self._inputs_classification = []
        input_set_len = -1
        for (i, input) in enumerate(inputs):
            classification = _Classification(input, f'Input {i}')
            if isinstance(classification.is_batch, list):
                if input_set_len == -1:
                    input_set_len = len(classification.is_batch)
                elif input_set_len != len(classification.is_batch):
                    raise ValueError(f"All argument lists for Multiple Input Sets used with operator '{op_name}' must have the same length.")
            self._inputs_classification.append(classification)
        if _conditionals.conditionals_enabled():
            if input_set_len != -1:
                raise ValueError('Multiple input sets are not supported with conditional execution (when `enable_conditionals=True`).')
        self.expected_inputs_size = len(inputs)
        if 'device' not in self._init_args and len(inputs) > 0:
            self._init_args['device'] = self._inputs_classification[0].device
        if 'seed' not in self._init_args:
            self._init_args['seed'] = seed
        self._device = self._init_args.get('device', 'cpu')
        self._device_id = device_id
        self._expected_inputs_size = len(inputs)
        self.op_helper = op_class(**self._init_args)
        self._op_name = op_name
        self.op_spec = self.op_helper._spec
        self._pipe = pipe
        self._source_context = source_context
        self.logical_ids = [id for id in range(next_logical_id, next_logical_id + abs(input_set_len))]
        for i in range(len(inputs)):
            self.op_spec.AddInput(op_name + f'[{i}]', self._inputs_classification[i].device)
        for arg_name in self._call_args.keys():
            self.op_spec.AddArgumentInput(arg_name, '')
        if self.op_helper.schema_name == '_conditional__Split':
            if '_if_stmt: True' not in repr(self.op_spec):
                self._pipe._cur_iter_batch_info.mark_non_uniform_batch()

    def _separate_kwargs(self, kwargs):
        if False:
            print('Hello World!')
        self._init_args = {}
        self._call_args = {}
        self._kwargs_classification = {}
        for (key, value) in kwargs.items():
            classification = _Classification(value, f'Argument {key}', arg_constant_len=self._batch_size)
            if classification.is_batch:
                self._call_args[key] = classification.data
            else:
                self._init_args[key] = classification.data
            self._kwargs_classification[key] = classification

    def _init_arithm_op(self, name, inputs):
        if False:
            while True:
                i = 10
        'Fills arithmetic operator init arguments and returns inputs that are DataNodes.'
        (categories_idxs, data_nodes, integers, reals) = _ops._group_inputs(inputs)
        input_desc = _ops._generate_input_desc(categories_idxs, integers, reals)
        self._init_args['device'] = _ops._choose_device(data_nodes)
        self._init_args['expression_desc'] = f'{name}({input_desc})'
        self._init_args['integer_constants'] = integers
        self._init_args['real_constants'] = reals
        return data_nodes

    def _pack_to_data_node_debug(self, data, position=None):
        if False:
            i = 10
            return i + 15
        if isinstance(data, (list, tuple)):
            return [self._pack_to_data_node_debug(elem, pos) for (pos, elem) in enumerate(data)]

        def position_to_suffix(position):
            if False:
                return 10
            if position is None:
                return ''
            else:
                return f'[{position}]'
        return DataNodeDebug(data, self._op_name + position_to_suffix(position), 'gpu' if isinstance(data, _tensors.TensorListGPU) else 'cpu', self)

    def _check_arg_len(self, expected_len, actual_len, args_type):
        if False:
            return 10
        if expected_len != actual_len:
            raise RuntimeError(f"Trying to use operator '{self._op_name}' with different number of {args_type} than when it was built. Expected: {expected_len} {args_type}, got {actual_len}.")

    def _check_device_classification(self, expected, actual, arg_type, value):
        if False:
            print('Hello World!')
        if expected != actual:
            raise RuntimeError(f"{arg_type} {value} for operator '{self._op_name}' is on '{actual}' but was on '{expected}' when created.")

    def _check_batch_classification(self, expected_is_batch, actual_is_batch, arg_type, value):
        if False:
            while True:
                i = 10

        def classification_to_str(is_batch):
            if False:
                for i in range(10):
                    print('nop')
            return 'batch' if is_batch else 'constant'
        if expected_is_batch != actual_is_batch:
            expected_str = classification_to_str(expected_is_batch)
            actual_str = classification_to_str(actual_is_batch)
            raise RuntimeError(f"{arg_type} {value} for operator '{self._op_name}' is a {actual_str} but was a {expected_str} when created.")

    def _check_batch_size(self, classification, input_idx):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(classification.is_batch, list):
            for input in classification.data:
                self._pipe._cur_iter_batch_info.check_input(len(input), self._source_context, self._op_name, input_idx)
        else:
            self._pipe._cur_iter_batch_info.check_input(len(classification.data), self._source_context, self._op_name, input_idx)

    def _check_call_arg_meta_data(self, expected_data, actual_data, arg_type, value):
        if False:
            i = 10
            return i + 15
        " Check for changes in layout, ndim and dtype.\n\n        Args:\n            expected_data: Expected value of the data.\n            actual_data: Actual value of the data.\n            arg_type (str): String representation of the argument type, e.g. 'Input', 'Argument'.\n            value (str): Argument name for keyword arguments and a number for inputs.\n        "

        def raise_err(meta_name, actual_value, expected_value):
            if False:
                for i in range(10):
                    print('nop')
            raise RuntimeError(f"{arg_type} {value} for operator '{self._op_name}' has {meta_name} = {actual_value}, expected: {expected_value}.")
        expected_input_set = isinstance(expected_data, list)
        if expected_input_set != isinstance(actual_data, list):
            raise RuntimeError(f"{arg_type} {value} expected {('' if expected_input_set else 'not ')}to be an input set.")
        if isinstance(actual_data, list):
            if len(actual_data) != len(expected_data):
                raise RuntimeError(f'{arg_type} {value} expected to be used as Multiple Input Set with length = {len(expected_data)}, but has length = {len(actual_data)}.')
            for (expected_elem, actual_elem) in zip(expected_data, actual_data):
                self._check_call_arg_meta_data(expected_elem, actual_elem, arg_type, value)
        else:
            if len(actual_data) == 0:
                return
            if expected_data.layout() != actual_data.layout():
                raise_err('layout', actual_data.layout(), expected_data.layout())
            if expected_data.dtype != actual_data.dtype:
                raise_err('dtype', actual_data.dtype, expected_data.dtype)
            (expected_ndim, actual_ndim) = (len(expected_data[0].shape()), len(actual_data[0].shape()))
            if expected_ndim != actual_ndim:
                raise_err('ndim', actual_ndim, expected_ndim)

    def _prep_input_sets(self, inputs):
        if False:
            while True:
                i = 10
        inputs = list(inputs)
        for (i, input) in enumerate(inputs):
            if not isinstance(input, (_tensors.TensorListCPU, _tensors.TensorListGPU)) and (not (isinstance(input, list) and all([isinstance(elem, (_tensors.TensorListCPU, _tensors.TensorListGPU)) for elem in input]))):
                inputs[i] = _transform_data_to_tensorlist(input, len(input), device_id=self._device_id)
        return _build_input_sets(inputs, self._op_name)

    def _update_classification(self, old_collection, position, new_classification):
        if False:
            return 10
        'Keeps the data classification up to date in case of running the conditional mode or\n        split and merge operations producing empty batches.\n        Otherwise it is no-op.\n\n        Parameters\n        ----------\n        old_collection : list or dict\n            The old classification - list of input classification or dictionary of kwarg\n            classification\n        position : int or str\n            The lookup to the currently examinet element in the `old_collection`\n        new_classification : _Classification\n            New classification of the input/kwarg\n        '
        if old_collection[position].is_batch and len(old_collection[position].data) == 0:
            old_collection[position] = new_classification
            return new_classification
        return old_collection[position]

    def run(self, inputs, kwargs):
        if False:
            while True:
                i = 10
        'Checks correctness of inputs and kwargs and runs the backend operator.'
        self._check_arg_len(self._expected_inputs_size, len(inputs), 'inputs')
        self._check_arg_len(len(self._kwargs_classification), len(kwargs), 'keyword arguments')
        if _conditionals.conditionals_enabled():
            (inputs, kwargs) = _conditionals.apply_conditional_split_to_args(inputs, kwargs)
            input_data_nodes_bkp = inputs
        call_args = {}
        inputs = list(inputs)
        for (i, (input, expected_classification)) in enumerate(zip(inputs, self._inputs_classification)):
            classification = _Classification(input, f'Input {i}')
            expected_classification = self._update_classification(self._inputs_classification, i, classification)
            self._check_batch_classification(expected_classification.is_batch, classification.is_batch, 'Input', i)
            self._check_device_classification(expected_classification.device, classification.device, 'Input', i)
            if classification.is_batch:
                if self.op_helper.schema_name != '_conditional__Merge':
                    self._check_batch_size(classification, i)
                self._check_call_arg_meta_data(expected_classification.data, classification.data, 'Input', i)
            if classification.device != ('gpu' if self._device == 'gpu' else 'cpu'):
                raise RuntimeError(f"Cannot call {self._device.upper()} operator '{self._op_name}' with {classification.device.upper()} input {i}.")
            inputs[i] = classification.data
        input_sets = self._prep_input_sets(inputs)
        for (key, value) in kwargs.items():
            classification = _Classification(value, f'Argument {key}', arg_constant_len=self._batch_size)
            self._update_classification(self._kwargs_classification, key, classification)
            self._check_batch_classification(self._kwargs_classification[key].is_batch, classification.is_batch, 'Argument', key)
            self._check_device_classification(self._kwargs_classification[key].device, classification.device, 'Argument', key)
            if not classification.is_batch and classification.data != self._init_args[key] and (not (math.isnan(classification.data) and math.isnan(self._init_args[key]))):
                raise RuntimeError(f"Argument '{key}' for operator '{self._op_name}' unexpectedly changed value from '{self._init_args[key]}' to '{classification.data}'")
            if classification.is_batch:
                self._check_call_arg_meta_data(self._kwargs_classification[key].data, classification.data, 'Argument', key)
                call_args[key] = classification.data
        if _conditionals.conditionals_enabled():
            for (i, classification) in enumerate(self._inputs_classification):
                if classification.is_batch and (not classification.was_data_node):
                    raise ValueError(f"Debug mode with conditional execution (when `enable_conditionals=True`) doesn't allow for modification of operator outputs by libraries other than DALI or using the TensorLists extracted via `.get()` as inputs. Expected `DataNodeDebug` as an input, got {type(classification.original)} at input {i}.")
            for (key, classification) in self._kwargs_classification.items():
                if classification.is_batch and (not classification.was_data_node):
                    raise ValueError(f"Debug mode with conditional execution (when `enable_conditionals=True`) doesn't allow for modification of operator outputs by libraries other than DALI or using the TensorLists extracted via `.get()` as inputs. Expected `DataNodeDebug` as an input, got {type(classification.original)} for argument '{key}'.")
        res = [self._pipe._run_op_on_device(self._op_name, logical_id, self._device, input, call_args) for (input, logical_id) in zip(input_sets, self.logical_ids)]
        self._pipe._cur_iter_batch_info.set_if_empty(len(res[0][0]), self._source_context)
        if len(res) == 1:
            res = self._pack_to_data_node_debug(res[0])
        else:
            res = _repack_output_sets(res)
            res = self._pack_to_data_node_debug(res)
        if _conditionals.conditionals_enabled():
            _conditionals.register_data_nodes(res, input_data_nodes_bkp, kwargs)
            if self.op_helper.schema_name != '_conditional__Split':
                res = list(_conditionals.apply_conditional_split_to_branch_outputs(res))
        if len(res) == 1:
            return res[0]
        return res

class _PipelineDebug(_pipeline.Pipeline):
    """Debug mode for pipeline. Allows access to data inside the pipeline execution."""

    def __init__(self, exec_func, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self._debug_on = False
        self._external_sources = {}
        self._feed_input_data = {}
        self._exec_func = exec_func
        self._cur_operator_id = -1
        self._next_logical_id = 0
        self._seed_upper_bound = (1 << 31) - 1
        self._operators = {}
        self._operators_built = False
        self._cur_iter_batch_info = _IterBatchInfo(-1, None)
        device_id = self._device_id if self._device_id is not None else _types.CPU_ONLY_DEVICE_ID
        self._pipe = _b.PipelineDebug(self._max_batch_size, self._num_threads, device_id, self._set_affinity)
        import numpy as np
        seed = kwargs.get('seed', -1)
        if seed < 0:
            seed = np.random.randint(self._seed_upper_bound)
        self._seed_generator = np.random.default_rng(seed)

    def __enter__(self):
        if False:
            return 10
        raise RuntimeError('Currently pipeline in debug mode works only with `pipeline_def` decorator.Using `with` statement is not supported.')

    def build(self):
        if False:
            for i in range(10):
                print('nop')
        'Build the pipeline.\n\n        Symbolic version of build from the standard pipeline.\n        In debug mode operators are built during\n        the first run of the pipeline.\n\n        Refer to :meth:`Pipeline.build() <nvidia.dali.Pipeline.build>` for details.'
        self._built = True

    def run(self):
        if False:
            while True:
                i = 10
        'Run the pipeline and return the result.'
        import numpy as np
        if not self._built:
            raise RuntimeError('Pipeline must be built first.')
        self._debug_on = True
        self._cur_operator_id = -1
        self._cur_iter_batch_info.reset()
        _pipeline.Pipeline.push_current(self)
        res = self._exec_func()
        if res is None:
            res = ()
        elif not isinstance(res, tuple):
            res = (res,)
        self._debug_on = False
        if not self._operators_built:
            self._operators_built = True
        _pipeline.Pipeline.pop_current()
        outputs = []
        for (i, val) in enumerate(res):
            if isinstance(val, DataNodeDebug):
                outputs.append(val.get())
            elif isinstance(val, (list, tuple)):
                raise TypeError(f'Illegal pipeline output type.The output {i} contains a nested `DataNodeDebug`')
            else:
                outputs.append(_tensors.TensorListCPU(np.tile(val, (self._max_batch_size, *[1] * np.array(val).ndim))))
        self._condition_stack = _conditionals._ConditionStack()
        return tuple(outputs)

    def feed_input(self, data_node, data, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Pass data to an ExternalSource operator inside the pipeline.\n\n        Refer to :meth:`Pipeline.feed_input() <nvidia.dali.Pipeline.feed_input>` for details.'
        if not self._built:
            raise RuntimeError('Pipeline must be built first.')
        if isinstance(data_node, str):
            name = data_node
        else:
            _check(data_node)
            name = data_node.name
        if name not in self._external_sources:
            if name not in self._feed_input_data:
                self._feed_input_data[name] = []
            self._feed_input_data[name].append((data, kwargs))
        else:
            self._external_sources[name]._feed_input(name, data, kwargs)

    def _create_op(self, op_class, op_name, key, cur_context, inputs, kwargs):
        if False:
            return 10
        'Creates direct operator.'
        self._operators[key] = _OperatorManager(op_class, op_name, self, cur_context, self._next_logical_id, self._max_batch_size, self._device_id, self._seed_generator.integers(self._seed_upper_bound), inputs, kwargs)
        self._pipe.AddMultipleOperators(self._operators[key].op_spec, self._operators[key].logical_ids)
        self._next_logical_id = self._operators[key].logical_ids[-1] + 1

    def _check_external_source_batch_size(self, data, cur_context):
        if False:
            while True:
                i = 10
        if isinstance(data, list):
            for (i, output) in enumerate(data):
                self._cur_iter_batch_info.check_external_source(len(output.get()), cur_context, i)
        else:
            self._cur_iter_batch_info.check_external_source(len(data.get()), cur_context)

    def _external_source(self, name=None, **kwargs):
        if False:
            return 10
        self._cur_operator_id += 1
        cur_frame = inspect.currentframe().f_back.f_back
        key = inspect.getframeinfo(cur_frame)[:3] + (self._cur_operator_id,)
        if not self._operators_built:
            es = _ExternalSourceDebug(batch_size=self._max_batch_size, device_id=self._device_id, name=name, **kwargs)
            for (data, fi_kwargs) in self._feed_input_data.pop(name, []):
                es._feed_input(data, fi_kwargs)
            self._external_sources[key] = es
        if key in self._external_sources:
            data = self._external_sources[key]._fetch(self._epoch_idx)
            self._check_external_source_batch_size(data, ''.join(traceback.format_stack(cur_frame, limit=1)))
            return data
        else:
            raise RuntimeError("Unexpected operator 'ExternalSource'. Debug mode does not support changing the order of operators executed within the pipeline.")

    def _run_op_on_device(self, op_name, logical_id, device, inputs, kwargs):
        if False:
            return 10

        def is_converted_to_batch(elem):
            if False:
                for i in range(10):
                    print('nop')
            return isinstance(elem, (_tensors.TensorListCPU, _tensors.TensorGPU))
        batch_input = any((is_converted_to_batch(input) for input in inputs))
        batch_input = batch_input or any((is_converted_to_batch(arg) for (_, arg) in kwargs.items()))
        if batch_input:
            requested_size = self._cur_iter_batch_info.size
        else:
            requested_size = -1
        if device == 'gpu':
            return self._pipe.RunOperatorGPU(logical_id, inputs, kwargs, requested_size)
        if device == 'cpu':
            return self._pipe.RunOperatorCPU(logical_id, inputs, kwargs, requested_size)
        if device == 'mixed':
            return self._pipe.RunOperatorMixed(logical_id, inputs, kwargs, requested_size)
        raise ValueError(f"Unknown device: '{device}' in operator '{op_name}'.")

    def _run_op(self, op_helper, inputs, kwargs):
        if False:
            while True:
                i = 10
        return op_helper.run(inputs, kwargs)

    @staticmethod
    def _extract_data_node_inputs(inputs):
        if False:
            while True:
                i = 10
        '\n        Extracts DataNodeDebugs from inputs for\n        arithmetic operator and transforms data to GPU if needed.\n        '
        data_nodes = []
        to_gpu = any([input.device == 'gpu' for input in inputs if isinstance(input, DataNodeDebug)])
        for input in inputs:
            if isinstance(input, DataNodeDebug):
                if to_gpu and input.device != 'gpu':
                    data_nodes.append(input.gpu())
                else:
                    data_nodes.append(input)
        return data_nodes

    def _wrap_op_call(self, op_class, op_name, *inputs, **kwargs):
        if False:
            print('Hello World!')
        self._cur_operator_id += 1
        cur_frame = inspect.currentframe().f_back.f_back
        cur_context = ''.join(traceback.format_stack(cur_frame, limit=1))
        key = inspect.getframeinfo(cur_frame)[:3] + (self._cur_operator_id,)
        if not self._operators_built:
            self._create_op(op_class, op_name, key, cur_context, inputs, kwargs)
        if key in self._operators:
            if op_name == 'arithmetic_generic_op':
                inputs = _PipelineDebug._extract_data_node_inputs(inputs)
            return self._run_op(self._operators[key], inputs, kwargs)
        else:
            raise RuntimeError(f"Unexpected operator '{op_name}'. Debug mode does not support changing the order of operators executed within the pipeline.")