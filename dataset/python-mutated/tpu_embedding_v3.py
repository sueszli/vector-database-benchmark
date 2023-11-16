"""Mid level API for TPU Embeddings With V2 Embedding Accelerator."""
import collections
import copy
import functools
import operator
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from absl import logging
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.distribute import tpu_util
from tensorflow.python.distribute import tpu_values
from tensorflow.python.distribute import values
from tensorflow.python.distribute import values_util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_collective_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.tpu import _pywrap_tpu_embedding
from tensorflow.python.tpu import tpu_embedding_base
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu.ops import gen_xla_ops as xla_ops
from tensorflow.python.trackable import base
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
_PIPELINE_ATTRIBUTE = '_embedding_pipelining'
_PIPELINE_MODE_FORWARD = 'forward'
_PIPELINE_MODE_BACKWARD = 'backward'

class EmbeddingPipeliningContext(control_flow_ops.ControlFlowContext):
    """Sets the _embedding_pipelining attribute on all ops created in the scope."""

    def __init__(self, mode: str, enable: bool):
        if False:
            while True:
                i = 10
        super().__init__()
        self._name = 'EmbeddingPipelinigContext'
        self._mode = attr_value_pb2.AttrValue(s=compat.as_bytes(mode))
        self._enable = enable

    def to_control_flow_context_def(self, context_def: Any, export_scope: Any=None):
        if False:
            return 10
        super().to_control_flow_context_def(context_def, export_scope)

    def AddOp(self, op: ops.Operation):
        if False:
            return 10
        if self._enable:
            op._set_attr(_PIPELINE_ATTRIBUTE, self._mode)
        if self._outer_context:
            self._outer_context.AddOp(op)

class TPUEmbeddingShardedSaveable(saveable_object.SaveableObject):
    """Defines how to save and restore a shard of TPUEmbedding sharded variable."""

    def __init__(self, variable: tf_variables.Variable, shard_id: int, num_shards: int, shard_dim: int, name: str):
        if False:
            print('Hello World!')
        'Init TPUEmbeddingShardedSaveable.'
        self._shard_id = shard_id
        self._variable = variable
        var_offset = [0] * len(variable.shape)
        var_offset[shard_dim] = shard_id * variable.shape[shard_dim]
        fullshape = variable.shape.as_list()
        fullshape[shard_dim] = num_shards * fullshape[shard_dim]
        save_slice_info = tf_variables.Variable.SaveSliceInfo(full_name=name, full_shape=fullshape, var_offset=var_offset, var_shape=variable.shape.as_list())
        spec = saveable_object.SaveSpec(tensor=variable.read_value, slice_spec=save_slice_info.spec, name=name, dtype=variable.dtype, device=variable.device)
        super().__init__(variable.read_value, [spec], name)

    def restore(self, restored_tensors: List[tensor.Tensor], restored_shapes: List[tensor_shape.TensorShape]) -> Any:
        if False:
            while True:
                i = 10
        del restored_shapes
        restored_tensor = restored_tensors[0]
        return values_util.assign_on_device(self._variable.device, self._variable, restored_tensor)

@saveable_compat.legacy_saveable_name('')
class TPUEmbeddingShardedVariable(tpu_values.TPUVariableMixin, values.DistributedVariable):
    """A ShardedVariable class for Embedding tables on TPU."""

    def _is_mirrored(self) -> bool:
        if False:
            i = 10
            return i + 15
        return False

    @property
    def shard_dim(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return 0

    @property
    def shape(self) -> tensor_shape.TensorShape:
        if False:
            print('Hello World!')
        'Returns the shape of the embedding variable for the current context.'
        local_shape = self._values[0].shape
        global_shape = local_shape.as_list()
        global_shape[self.shard_dim] = global_shape[self.shard_dim] * len(self.values)
        return tensor_shape.TensorShape(global_shape)

    def _write_object_proto(self, proto, options):
        if False:
            print('Hello World!')
        super()._write_object_proto(proto, options)
        proto.variable.shape.CopyFrom(self._values[0].shape.as_proto())

    def _gather_saveables_for_checkpoint(self) -> Dict[str, Callable[..., Any]]:
        if False:
            i = 10
            return i + 15
        'Overrides Trackable method.\n\n    Returns:\n      A dictionary mapping attribute names to `SaveableObject` factories.\n    '

        def _saveable_factory(name=self._common_name):
            if False:
                return 10
            saveables = []
            num_shards = len(self.values)
            for shard_id in range(num_shards):
                saveables.append(TPUEmbeddingShardedSaveable(self.values[shard_id], shard_id, num_shards, self.shard_dim, name))
            return saveables
        return {base.VARIABLE_VALUE_KEY: _saveable_factory}

    def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
        if False:
            print('Hello World!')
        'Converts a variable to a tensor.'
        if tpu_util.enclosing_tpu_context() is None:
            return self._values[0].read_value()
        else:
            return self._read_variable_op()

    def read_value(self) -> Any:
        if False:
            print('Hello World!')
        if tpu_util.enclosing_tpu_context() is None:
            raise NotImplementedError('Reading in cross replica mode is not yet supportedfor TPUEmbeddingShardedVariable.')
        else:
            return self._read_variable_op()

    def assign(self, value: Any, use_locking: bool=False, name: Optional[Any]=None, read_value: bool=True) -> Any:
        if False:
            for i in range(10):
                print('nop')
        if tpu_util.enclosing_tpu_context() is None:
            for device in self.distribute_strategy.extended.worker_devices:
                with ops.device(device):
                    self.assign_on_device(device, value)
        return tpu_util.make_raw_assign_fn(gen_resource_variable_ops.assign_variable_op)(self, value=value, use_locking=use_locking, name=name, read_value=read_value)

    def assign_on_device(self, device, value):
        if False:
            print('Hello World!')
        if self._packed_var is None:
            raise NotImplementedError('Required packed variable support')
        with ops.device(device):
            gen_resource_variable_ops.assign_variable_op(resource=self._packed_var.handle, value=value)

    def read_from_device(self, device):
        if False:
            while True:
                i = 10
        if self._packed_var is None:
            raise NotImplementedError('Required packed variable support')
        with ops.device(device):
            return gen_resource_variable_ops.read_variable_op(resource=self._packed_var.handle, dtype=self.dtype)
PartitionedCsrFormatTensor = collections.namedtuple('PartitionedCsrFormatTensor', ['row_pointers', 'sorted_sample_ids', 'sorted_token_ids', 'sorted_gains', 'sample_count', 'num_minibatches_per_physical_sparse_core'])

@tf_export('tpu.experimental.embedding.TPUEmbeddingV2')
class TPUEmbeddingV2(tpu_embedding_base.TPUEmbeddingBase):
    """The TPUEmbedding mid level API running on TPU with sparse core accelerator."""

    def __init__(self, feature_config: Union[tpu_embedding_v2_utils.FeatureConfig, Iterable], optimizer: Optional[tpu_embedding_v2_utils._Optimizer]=None, pipeline_execution_with_tensor_core: bool=False):
        if False:
            print('Hello World!')
        "Creates the TPUEmbeddingV2 mid level API object.\n\n    Args:\n      feature_config: A nested structure of\n        `tf.tpu.experimental.embedding.FeatureConfig` configs.\n      optimizer: An instance of one of `tf.tpu.experimental.embedding.SGD`,\n        `tf.tpu.experimental.embedding.Adagrad` or\n        `tf.tpu.experimental.embedding.Adam`. When not created under TPUStrategy\n        may be set to None to avoid the creation of the optimizer slot\n        variables, useful for optimizing memory consumption when exporting the\n        model for serving where slot variables aren't needed.\n      pipeline_execution_with_tensor_core: If True, the TPU embedding\n        computations will overlap with the TensorCore computations (and hence\n        will be one step old). Set to True for improved performance.\n\n    Raises:\n      ValueError: If optimizer is not one of tf.tpu.experimental.embedding.(SGD,\n      Adam or Adagrad) or None when created under a TPUStrategy.\n      RuntimeError: If not created under TPUStrategy.\n    "
        super().__init__(self._clone_feature_config(feature_config), optimizer)
        self._strategy = distribute_lib.get_strategy()
        if not isinstance(self._strategy, (tpu_strategy.TPUStrategy, tpu_strategy.TPUStrategyV2)):
            raise RuntimeError('TPUEmbeddingV2 should be created under TPUStrategy but found {}.'.format(self._strategy))
        self._num_sc_per_chip = self._strategy.extended.tpu_hardware_feature.num_embedding_devices_per_chip
        if self._num_sc_per_chip == 0:
            logging.warning('No embedding devices per chip info is found. Using 4 as the default value for SparseCore.')
            self._num_sc_per_chip = 4
        self._num_sc_shards = self._strategy.num_replicas_in_sync * self._num_sc_per_chip
        self._flat_features = nest.flatten_with_joined_string_paths(self._feature_config)
        self._round_table_sizes()
        self._stack_tables_with_same_table_dim_and_optimizer()
        self.max_ids_per_chip_per_sample = 64
        self.max_minibatches_per_sc = 64
        self._pipelining = pipeline_execution_with_tensor_core

    def _clone_feature_config(self, feature_config):
        if False:
            while True:
                i = 10
        old_to_new_table = {}
        new_features = []
        for old_feature in nest.flatten(feature_config):
            feature = copy.copy(old_feature)
            if feature.table not in old_to_new_table:
                old_to_new_table[feature.table] = copy.copy(feature.table)
            feature.table = old_to_new_table[feature.table]
            new_features.append(feature)
        return nest.pack_sequence_as(feature_config, new_features)

    def _round_table_sizes(self):
        if False:
            while True:
                i = 10
        num_shards = self._num_sc_shards * 8
        self._table_to_padding_columns = {}
        self._table_to_padding_rows = {}
        for table in self._table_config:
            extra_rows = (num_shards - table.vocabulary_size % num_shards) % num_shards
            extra_cols = (8 - table.dim % 8) % 8
            if extra_rows != 0:
                if table.vocabulary_size < num_shards:
                    logging.warning('!!! Adding %d extra rows to a small table %s!!! Table had %d rows before padding and %d rows after padding.', extra_rows, table.name, table.vocabulary_size, table.vocabulary_size + extra_rows)
                else:
                    logging.warning('Adding %d extra rows to table %s to get %d rows.', extra_rows, table.name, table.vocabulary_size + extra_rows)
            if extra_cols != 0:
                logging.warning('Adding %d extra columns to table %s to get %d columns.', extra_cols, table.name, table.dim + extra_cols)
            self._table_to_padding_columns[table.name] = extra_cols
            self._table_to_padding_rows[table.name] = extra_rows
            table.vocabulary_size += extra_rows
            table.dim += extra_cols
        return

    @property
    def embedding_tables(self) -> Dict[tpu_embedding_v2_utils.TableConfig, tf_variables.Variable]:
        if False:
            return 10
        'Returns a dict of embedding tables, keyed by `TableConfig`.'
        self._maybe_build()
        return {stacked_table_name: self._variables[stacked_table_name]['parameters'] for stacked_table_name in self._stacked_table_to_tables}

    @property
    def embedding_table_shards(self) -> Dict[tpu_embedding_v2_utils.TableConfig, List[tf_variables.Variable]]:
        if False:
            for i in range(10):
                print('nop')
        'Returns a dict of embedding tables, keyed by `TableConfig`.'
        self._maybe_build()
        ordered_devices = []
        for devices in self._strategy.extended._tpu_devices:
            ordered_devices.extend(devices)
        table_shards = {name: [(device, var.read_from_device(device)) for device in ordered_devices] for (name, var) in self.embedding_tables.items()}
        return table_shards

    @property
    def variables(self) -> Dict[tpu_embedding_v2_utils.TableConfig, Dict[str, tf_variables.Variable]]:
        if False:
            print('Hello World!')
        'Returns a dict of variables, keyed by `TableConfig`, then by slot name.'
        self._maybe_build()
        return self._variables

    def _create_variables(self, stacked_tables: List[tpu_embedding_v2_utils.TableConfig], stacked_table_name: str) -> Dict[str, tf_variables.Variable]:
        if False:
            print('Hello World!')
        'Create all variables including table variables and slot variables.'
        total_vocab_size = sum([table.vocabulary_size for table in stacked_tables])
        table_dim = stacked_tables[0].dim
        variable_shape = (total_vocab_size, table_dim)
        optimizer = stacked_tables[0].optimizer

        def table_initialize_fn(shape, dtype, shard_info=None):
            if False:
                print('Hello World!')
            table_tensors = []
            for i in range(self._num_sc_per_chip):
                full_tables = {}
                for table in stacked_tables:
                    shift = self._table_to_stacked_table_offset[table.name][2]
                    arg_spec = tf_inspect.getfullargspec(table.initializer)
                    sharding_aware = 'shard_info' in arg_spec.args or 'shard_info' in arg_spec.kwonlyargs
                    if shard_info and (not sharding_aware):
                        if table.name not in full_tables:
                            full_tables[table.name] = table.initializer(shape=(table.vocabulary_size, table.dim), dtype=dtype)
                    if shard_info is not None:
                        partition_shape = shard_info.shape
                        partition_offset = shard_info.offset
                        sc_shard_size = table.vocabulary_size * partition_shape[0] // total_vocab_size // self._num_sc_per_chip
                        sc_shard_offset = table.vocabulary_size * partition_offset[0] // total_vocab_size + i * sc_shard_size
                        sc_shard_info = base.ShardInfo([sc_shard_size, table.dim], [sc_shard_offset, 0])
                        if sharding_aware:
                            sc_shard = table.initializer(shape=(table.vocabulary_size, table.dim), dtype=dtype, shard_info=sc_shard_info)
                        else:
                            shard_index = sc_shard_info.offset[0] // sc_shard_info.shape[0]
                            shard_index = (shard_index - shift) % self._num_sc_shards
                            tpu_devices = self._strategy.extended._tpu_devices
                            (num_replicas, num_cores_per_replica) = tpu_devices.shape
                            num_sc = num_replicas * num_cores_per_replica * self._num_sc_per_chip
                            sc_shard = full_tables[table.name][shard_index::num_sc, :]
                    else:
                        sc_shard = table.initializer(shape=(table.vocabulary_size * shape[0] // total_vocab_size // self._num_sc_per_chip, table.dim), dtype=dtype)
                    table_tensors.append(sc_shard)
            return array_ops.concat(table_tensors, axis=0)

        def getter(name, shape, dtype, initializer, trainable):
            if False:
                i = 10
                return i + 15
            del shape
            initial_value = functools.partial(initializer, shape=variable_shape, dtype=dtype)
            return tf_variables.Variable(name=name, initial_value=initial_value, shape=variable_shape, dtype=dtype, trainable=trainable)

        def variable_creator(name, initializer):
            if False:
                print('Hello World!')
            return self._add_variable_with_custom_getter(name=name, initializer=initializer, shape=variable_shape, dtype=dtypes.float32, getter=getter, trainable=False)
        with variable_scope.variable_creator_scope(make_sharded_variable_creator(self._strategy)):
            parameters = variable_creator(stacked_table_name, table_initialize_fn)

        def slot_creator(name, initializer):
            if False:
                return 10
            return variable_creator(stacked_table_name + '/' + name, initializer)
        if optimizer is not None:
            with variable_scope.variable_creator_scope(make_sharded_variable_creator(self._strategy)):
                slot_vars = optimizer._create_slots(parameters, slot_creator)
        else:
            slot_vars = {}
        slot_vars['parameters'] = parameters
        return slot_vars

    def _stack_tables_with_same_table_dim_and_optimizer(self):
        if False:
            for i in range(10):
                print('nop')
        'Stack tables with the same table dim and optimizer.'
        logging.info('Number of tables before stacking is %d', len(self._table_config))
        table_names = []
        table_widths = []
        table_heights = []
        table_num_samples = []
        table_groups = []
        table_data_to_group = {}
        table_to_num_samples = {table.name: 0 for table in self._table_config}
        table_name_to_table = {}
        for (_, feature) in self._flat_features:
            table_to_num_samples[feature.table.name] += functools.reduce(operator.mul, feature.output_shape)
        for table in self._table_config:
            table_name_to_table[table.name] = table
            key = (table.dim, table.optimizer, repr(table.quantization_config) if table.quantization_config else None)
            if key not in table_data_to_group:
                table_data_to_group[key] = len(table_data_to_group)
            table_groups.append(table_data_to_group[key])
            table_names.append(table.name)
            table_widths.append(table.dim)
            table_heights.append(table.vocabulary_size)
            table_num_samples.append(table_to_num_samples[table.name])
        table_stacks_by_name = _pywrap_tpu_embedding.stack_tables(table_heights, table_widths, table_num_samples, table_groups, table_names, self._strategy.num_replicas_in_sync)
        table_stacks = [[table_name_to_table[table_name] for table_name in stack_by_name] for stack_by_name in table_stacks_by_name]
        self._stacked_table_to_tables = {}
        self._table_to_stacked_table_offset = {}
        self._quantization_configs = {}
        for tables in table_stacks:
            stacked_table_name = '_'.join(map(lambda table: table.name, tables))
            if stacked_table_name in self._stacked_table_to_tables:
                raise ValueError(f'{stacked_table_name} already exists!')
            self._stacked_table_to_tables[stacked_table_name] = tables
            self._quantization_configs[stacked_table_name] = tables[0].quantization_config
            current_offset = 0
            current_index = 0
            for table in tables:
                self._table_to_stacked_table_offset[table.name] = (stacked_table_name, current_offset, self._num_sc_per_chip * current_index)
                current_offset += table.vocabulary_size
                current_index += 1
        logging.info('Number of tables after stacking is %d.', len(self._stacked_table_to_tables))
        self._feature_to_sample_offset = {}
        self._table_to_sample_count = {table_name: 0 for table_name in self._stacked_table_to_tables}
        for (feature_path, feature) in self._flat_features:
            stacked_table_name = self._table_to_stacked_table_offset[feature.table.name][0]
            self._feature_to_sample_offset[feature_path] = self._table_to_sample_count[stacked_table_name]
            self._table_to_sample_count[stacked_table_name] += functools.reduce(operator.mul, feature.output_shape)

    def _create_variables_and_slots(self) -> Dict[str, Dict[str, tf_variables.Variable]]:
        if False:
            for i in range(10):
                print('nop')
        "Create variables for TPU embeddings.\n\n    Returns:\n      A dict of dicts. The outer dict is keyed by the table names and the inner\n      dicts are keyed by 'parameters' and the slot variable names.\n    "
        variables = {}
        for (stacked_table_name, tables) in self._stacked_table_to_tables.items():
            variables[stacked_table_name] = self._create_variables(tables, stacked_table_name=stacked_table_name)
        return variables

    def _maybe_build(self):
        if False:
            print('Hello World!')
        if not self._built:
            with ops.init_scope():
                self.build()

    def build(self):
        if False:
            print('Hello World!')
        'Create variables and slots variables for TPU embeddings.'
        if self._built:
            return
        self._variables = self._create_variables_and_slots()
        self._built = True

    def apply_gradients(self, gradients: Any, preserved_outputs: Dict[str, PartitionedCsrFormatTensor]):
        if False:
            for i in range(10):
                print('nop')
        'Applies the gradient update to the embedding tables.\n\n    If a gradient of `None` is passed in any position of the nested structure,\n    then a gradient update with a zero gradient is applied for that feature.\n    For optimizers like SGD or Adagrad, this is the same as applying no update\n    at all. For lazy Adam and other sparsely applied optimizers with decay,\n    ensure you understand the effect of applying a zero gradient.\n\n    Args:\n      gradients: A nested structure of gradients, with structure matching the\n        `feature_config` passed to this object.\n      preserved_outputs: A dicts of PartitionedCsrFormatTensor, coming from the\n        second output of the embedding lookup call.\n\n    Raises:\n      RuntimeError: if not built.\n      ValueError: If a non-`tf.Tensor` non-`None` gradient is passed in, or a\n        `tf.Tensor` of the incorrect shape is passed in. Also if\n        the size of any sequence in `gradients` does not match corresponding\n        sequence in `feature_config`.\n      TypeError: If the type of any sequence in `gradients` does not match\n        corresponding sequence in `feature_config`.\n    '
        if not self._built:
            raise RuntimeError('apply_gradients called on unbuilt TPUEmbeddingV2 object. Please either call the embedding lookup method first or manually call the build method.')
        nest.assert_same_structure(self._feature_config, gradients)
        gradients = self._stack_gradients(gradients)
        context = EmbeddingPipeliningContext(_PIPELINE_MODE_BACKWARD, self._pipelining)
        context.Enter()

        def _wrap_param(param, dtype=dtypes.float32):
            if False:
                while True:
                    i = 10
            if callable(param):
                param = math_ops.cast(param(), dtype=dtype)
            return ops.convert_to_tensor(param, dtype=dtype)
        num_minibatches_per_physical_sparse_core = list(preserved_outputs.values())[0].num_minibatches_per_physical_sparse_core
        for table_name in self._stacked_table_to_tables:
            gradient = gradients[table_name]
            partitioned_tensor = preserved_outputs[table_name]
            table = self.variables[table_name]['parameters']
            optimizer = self._stacked_table_to_tables[table_name][0].optimizer
            if isinstance(optimizer, tpu_embedding_v2_utils.SGD):
                updated_embedding_table = xla_ops.xla_sparse_dense_matmul_grad_with_sgd_and_csr_input(row_pointers=partitioned_tensor.row_pointers, sorted_sample_ids=partitioned_tensor.sorted_sample_ids, sorted_token_ids=partitioned_tensor.sorted_token_ids, sorted_gains=partitioned_tensor.sorted_gains, activation_gradients=gradient, learning_rate=_wrap_param(optimizer.learning_rate), embedding_table=table.read_value(), num_minibatches_per_physical_sparse_core=num_minibatches_per_physical_sparse_core, table_name=table_name)
                table.assign(updated_embedding_table)
            elif isinstance(optimizer, tpu_embedding_v2_utils.Adagrad):
                accumulators = self.variables[table_name]['accumulators']
                (updated_embedding_table, updated_accumulator) = xla_ops.xla_sparse_dense_matmul_grad_with_adagrad_and_csr_input(row_pointers=partitioned_tensor.row_pointers, sorted_sample_ids=partitioned_tensor.sorted_sample_ids, sorted_token_ids=partitioned_tensor.sorted_token_ids, sorted_gains=partitioned_tensor.sorted_gains, activation_gradients=gradient, learning_rate=_wrap_param(optimizer.learning_rate), embedding_table=table.read_value(), accumulator=accumulators.read_value(), num_minibatches_per_physical_sparse_core=num_minibatches_per_physical_sparse_core, table_name=table_name)
                accumulators.assign(updated_accumulator)
                table.assign(updated_embedding_table)
            elif isinstance(optimizer, tpu_embedding_v2_utils.AdagradMomentum):
                accumulators = self.variables[table_name]['accumulators']
                momenta = self.variables[table_name]['momenta']
                (updated_embedding_table, updated_accumulator, updated_momenta) = xla_ops.xla_sparse_dense_matmul_grad_with_adagrad_momentum_and_csr_input(row_pointers=partitioned_tensor.row_pointers, sorted_sample_ids=partitioned_tensor.sorted_sample_ids, sorted_token_ids=partitioned_tensor.sorted_token_ids, sorted_gains=partitioned_tensor.sorted_gains, activation_gradients=gradient, learning_rate=_wrap_param(optimizer.learning_rate), embedding_table=table.read_value(), accumulator=accumulators.read_value(), momenta=momenta.read_value(), num_minibatches_per_physical_sparse_core=num_minibatches_per_physical_sparse_core, use_nesterov=optimizer.use_nesterov, exponent=optimizer.exponent, beta1=optimizer.momentum, beta2=optimizer.beta2, epsilon=optimizer.epsilon, table_name=table_name)
                momenta.assign(updated_momenta)
                accumulators.assign(updated_accumulator)
                table.assign(updated_embedding_table)
            elif isinstance(optimizer, tpu_embedding_v2_utils.Adam):
                momenta = self.variables[table_name]['momenta']
                velocity = self.variables[table_name]['velocities']
                (updated_embedding_table, updated_momenta, updated_velocity) = xla_ops.xla_sparse_dense_matmul_grad_with_adam_and_csr_input(row_pointers=partitioned_tensor.row_pointers, sorted_sample_ids=partitioned_tensor.sorted_sample_ids, sorted_token_ids=partitioned_tensor.sorted_token_ids, sorted_gains=partitioned_tensor.sorted_gains, activation_gradients=gradient, learning_rate=_wrap_param(optimizer.learning_rate), embedding_table=table.read_value(), momenta=momenta.read_value(), velocity=velocity.read_value(), num_minibatches_per_physical_sparse_core=num_minibatches_per_physical_sparse_core, use_sum_inside_sqrt=optimizer.sum_inside_sqrt, beta1=optimizer.beta_1, beta2=optimizer.beta_2, epsilon=optimizer.epsilon, table_name=table_name)
                velocity.assign(updated_velocity)
                momenta.assign(updated_momenta)
                table.assign(updated_embedding_table)
            elif isinstance(optimizer, tpu_embedding_v2_utils.FTRL):
                accumulators = self.variables[table_name]['accumulators']
                linears = self.variables[table_name]['linears']
                (updated_table_tensor, updated_accum_tensor, updated_linear_tensor) = xla_ops.xla_sparse_dense_matmul_grad_with_ftrl_and_csr_input(row_pointers=partitioned_tensor.row_pointers, sorted_sample_ids=partitioned_tensor.sorted_sample_ids, sorted_token_ids=partitioned_tensor.sorted_token_ids, sorted_gains=partitioned_tensor.sorted_gains, activation_gradients=gradient, learning_rate=_wrap_param(optimizer.learning_rate), embedding_table=table.read_value(), accumulator=accumulators.read_value(), linear=linears.read_value(), num_minibatches_per_physical_sparse_core=num_minibatches_per_physical_sparse_core, multiply_linear_by_learning_rate=optimizer.multiply_linear_by_learning_rate, beta=optimizer.beta, learning_rate_power=optimizer.learning_rate_power, l1_regularization_strength=optimizer.l1_regularization_strength, l2_regularization_strength=optimizer.l2_regularization_strength, table_name=table_name)
                linears.assign(updated_linear_tensor)
                accumulators.assign(updated_accum_tensor)
                table.assign(updated_table_tensor)
            else:
                raise ValueError('Unsupported optimizer in minibatching mode.')
        context.Exit()

    def _stack_gradients(self, gradients):
        if False:
            for i in range(10):
                print('nop')
        'Stack the incoming gradients to per table gradients.'
        table_to_gradient_list = {table_name: [] for table_name in self._stacked_table_to_tables}
        flattend_gradients = nest.flatten(gradients)
        for (gradient, (path, feature)) in zip(flattend_gradients, self._flat_features):
            sample_count = functools.reduce(operator.mul, feature.output_shape)
            if gradient is not None and (not isinstance(gradient, tensor.Tensor)):
                raise ValueError(f'found non-tensor type: {type(gradient)} at path {path}.')
            if gradient is None:
                logging.warning('No gradient passed for feature %s, sending zero gradient. This may not be correct behavior for certain optimizers like Adam.', path)
                gradient = array_ops.zeros((sample_count, feature.table.dim), dtype=dtypes.float32)
            table_name = self._table_to_stacked_table_offset[feature.table.name][0]
            extra_cols = self._table_to_padding_columns[feature.table.name]
            gradient = array_ops.reshape(gradient, [-1, feature.table.dim - extra_cols])
            if extra_cols != 0:
                gradient = array_ops.pad(gradient, [[0, 0], [0, extra_cols]])
                gradient.set_shape([sample_count, feature.table.dim])
            table_to_gradient_list[table_name].append(gradient)
        return {table_name: array_ops.concat(table_to_gradient_list[table_name], axis=0) for table_name in table_to_gradient_list}

    def _unstack_activations(self, activations: Dict[str, tensor.Tensor]):
        if False:
            for i in range(10):
                print('nop')
        'Untack the incoming per table activations into per feature.'
        flattened_activations = []
        table_to_current_offset = {table_name: 0 for table_name in self._stacked_table_to_tables}
        for (_, feature) in self._flat_features:
            sample_count = functools.reduce(operator.mul, feature.output_shape)
            table_name = self._table_to_stacked_table_offset[feature.table.name][0]
            extra_cols = self._table_to_padding_columns[feature.table.name]
            activation = array_ops.slice(activations[table_name], [table_to_current_offset[table_name], 0], [sample_count, feature.table.dim - extra_cols])
            activation = array_ops.reshape(activation, list(feature.output_shape) + [feature.table.dim - extra_cols])
            flattened_activations.append(activation)
            table_to_current_offset[table_name] += sample_count
        return nest.pack_sequence_as(self._feature_config, flattened_activations)

    def __call__(self, features: Any, weights: Optional[Any]=None) -> Tuple[Any, Dict[str, PartitionedCsrFormatTensor]]:
        if False:
            for i in range(10):
                print('nop')
        'Call the mid level api to do embedding lookup.'
        return self.embedding_lookup(features, weights)

    @staticmethod
    def _convert_input_feature_to_coo(input_feature: Union[tensor.Tensor, sparse_tensor.SparseTensor, ragged_tensor.RaggedTensor], weight: Optional[tensor.Tensor], feature_config: tpu_embedding_v2_utils.FeatureConfig, row_offset: int, col_offset: int, col_shift: int, vocab_size: int, num_sc_shards: int) -> Any:
        if False:
            while True:
                i = 10
        'Convert any of the expected input types to a COO format.'
        sample_count = functools.reduce(operator.mul, feature_config.output_shape)
        if isinstance(input_feature, tensor.Tensor):
            input_feature = array_ops.reshape(input_feature, [-1])
            if weight is None:
                weight = array_ops.ones_like(input_feature, dtype=dtypes.float32)
            elif isinstance(weight, tensor.Tensor):
                weight = array_ops.reshape(weight, [-1])
            else:
                raise ValueError(f'Expect weight to be Tensor type but got {type(weight)}')
            (row_ids, col_ids, gains) = xla_ops.convert_to_coo_tensor(indices_or_row_splits=array_ops.zeros((0,), dtype=dtypes.int32), values=math_ops.cast(input_feature, dtype=dtypes.int32), weights=math_ops.cast(weight, dtypes.float32), sample_count=sample_count, combiner=feature_config.table.combiner)
        elif isinstance(input_feature, sparse_tensor.SparseTensor):
            if weight is None:
                weight = array_ops.ones_like(input_feature.values, dtype=dtypes.float32)
            elif isinstance(weight, sparse_tensor.SparseTensor):
                weight = weight.values
            else:
                raise ValueError(f'Expect weight to be SparseTensor type but got {type(weight)}')
            (row_ids, col_ids, gains) = xla_ops.convert_to_coo_tensor(indices_or_row_splits=math_ops.cast(input_feature.indices, dtype=dtypes.int32), values=math_ops.cast(input_feature.values, dtype=dtypes.int32), weights=math_ops.cast(weight, dtypes.float32), sample_count=sample_count, combiner=feature_config.table.combiner)
        elif isinstance(input_feature, ragged_tensor.RaggedTensor):
            if not weight:
                weight = array_ops.ones_like(input_feature.values, dtype=dtypes.float32)
            elif isinstance(weight, ragged_tensor.RaggedTensor):
                weight = weight.values
            else:
                raise ValueError(f'Expect weight to be RaggedTensor type but got {type(weight)}')
            (row_ids, col_ids, gains) = xla_ops.convert_to_coo_tensor(indices_or_row_splits=math_ops.cast(input_feature.row_splits, dtype=dtypes.int32), values=math_ops.cast(input_feature.values, dtype=dtypes.int32), weights=math_ops.cast(weight, dtypes.float32), sample_count=sample_count, combiner=feature_config.table.combiner)
        else:
            raise ValueError(f'Input of unknown type {type(input_feature)}. Please only pass Tensor, SparseTensor or RaggedTensor as input to embedding lookup.')
        return (row_ids + row_offset, (col_ids + col_shift) % num_sc_shards + col_ids // num_sc_shards * num_sc_shards + col_offset, gains)

    @staticmethod
    def _preprocess_inputs_and_weights_to_coo_tensor(flat_inputs: Any, flat_weights: Any, flat_features: Any, stacked_table_to_tables: Dict[str, Any], table_to_stacked_table_offset: Dict[str, Tuple[str, int, int]], feature_to_sample_offset: Dict[str, int], num_sc_shards: int) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        'Convert the raw inputs into coo tensor.'
        table_to_list_of_coos = {table_name: ([], [], []) for table_name in stacked_table_to_tables}
        for (inp, weight, (feature_path, feature)) in zip(flat_inputs, flat_weights, flat_features):
            (table_name, col_offset, col_shift) = table_to_stacked_table_offset[feature.table.name]
            row_offset = feature_to_sample_offset[feature_path]
            (row_ids, col_ids, gains) = TPUEmbeddingV2._convert_input_feature_to_coo(inp, weight, feature, row_offset, col_offset, col_shift, feature.table.vocabulary_size, num_sc_shards)
            table_to_list_of_coos[table_name][0].append(row_ids)
            table_to_list_of_coos[table_name][1].append(col_ids)
            table_to_list_of_coos[table_name][2].append(gains)
        return table_to_list_of_coos

    @staticmethod
    def _get_minibatch_splits_from_coo_tensor(num_replicas_in_sync: int, table_to_list_of_coos: Dict[str, Any], stacked_table_to_tables: Dict[str, Any], table_to_sample_count: Dict[str, int], num_sc_per_chip: int) -> Tuple[Dict[str, Any], List[tensor.Tensor]]:
        if False:
            print('Hello World!')
        'Compute minibatch splits from the coo tensor.'
        table_to_sorted_coo_tensor = {}
        per_replica_table_splits = []
        for table_name in stacked_table_to_tables:
            row_ids = array_ops.concat(table_to_list_of_coos[table_name][0], axis=0)
            col_ids = array_ops.concat(table_to_list_of_coos[table_name][1], axis=0)
            gains = array_ops.concat(table_to_list_of_coos[table_name][2], axis=0)
            feature_width = stacked_table_to_tables[table_name][0].dim
            total_vocab_size = sum([table.vocabulary_size for table in stacked_table_to_tables[table_name]])
            (sorted_row_ids, sorted_col_ids, sorted_gains, splits, id_counts, unused_max_ids, unused_max_uniques) = xla_ops.get_minibatch_splits_with_physical_replica(program_key=constant_op.constant(['']), row_ids=row_ids, col_ids=col_ids, gains=gains, sample_count=table_to_sample_count[table_name], num_replica=num_replicas_in_sync, table_vocab_size=total_vocab_size, feature_width=feature_width, num_sc_per_chip=num_sc_per_chip, table_name=table_name, mini_batch_splits='')
            table_to_sorted_coo_tensor[table_name] = (sorted_row_ids, sorted_col_ids, sorted_gains, id_counts)
            per_replica_table_splits.append(splits)
        return (table_to_sorted_coo_tensor, per_replica_table_splits)

    @staticmethod
    def _get_minibatches_from_sorted_coo_tensor(num_replicas_in_sync: int, max_ids_per_chip_per_sample: int, max_minibatches_per_sc: int, table_to_sorted_coo_tensor: Dict[str, Any], cross_replica_table_splits: tensor.Tensor, stacked_table_to_tables: Dict[str, Any], table_to_sample_count: Dict[str, int], num_sc_per_chip: int) -> Any:
        if False:
            for i in range(10):
                print('nop')
        'Partition the sorted coo tensor into minibatches.'
        table_to_csr_format_tensor = {}
        for table_name in stacked_table_to_tables:
            (sorted_row_ids, sorted_col_ids, sorted_gains, id_counts) = table_to_sorted_coo_tensor[table_name]
            feature_width = stacked_table_to_tables[table_name][0].dim
            total_vocab_size = sum([table.vocabulary_size for table in stacked_table_to_tables[table_name]])
            (row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains, row_pointers_unpadded_size, ids_unpadded_size, num_minibatches_per_physical_sparse_core) = xla_ops.get_minibatches_in_csr_with_physical_replica(program_key=constant_op.constant(['']), row_ids=sorted_row_ids, col_ids=sorted_col_ids, gains=sorted_gains, splits=cross_replica_table_splits, id_counts=id_counts, sample_count=table_to_sample_count[table_name], num_replica=num_replicas_in_sync, max_minibatches_per_sc=max_minibatches_per_sc, max_ids_per_chip_per_sample=max_ids_per_chip_per_sample, table_vocab_size=total_vocab_size, feature_width=feature_width, num_sc_per_chip=num_sc_per_chip, table_name=table_name, mini_batch_in_csr='')
            table_to_csr_format_tensor[table_name] = (PartitionedCsrFormatTensor(row_pointers=row_pointers, sorted_sample_ids=sorted_sample_ids, sorted_token_ids=sorted_token_ids, sorted_gains=sorted_gains, sample_count=table_to_sample_count[table_name], num_minibatches_per_physical_sparse_core=num_minibatches_per_physical_sparse_core), row_pointers_unpadded_size, ids_unpadded_size)
        return table_to_csr_format_tensor

    def _raise_error_for_incorrect_control_flow_context(self):
        if False:
            return 10
        'Raises an error if we are not in the TPUReplicateContext.'
        graph = ops.get_default_graph()
        in_tpu_ctx = False
        while graph is not None:
            ctx = graph._get_control_flow_context()
            while ctx is not None:
                if isinstance(ctx, tpu_replication.TPUReplicateContext):
                    in_tpu_ctx = True
                    break
                ctx = ctx.outer_context
            if in_tpu_ctx:
                break
            graph = getattr(graph, 'outer_graph', None)
        if graph != ops.get_default_graph() and in_tpu_ctx:
            raise RuntimeError('Current graph {} does not match graph which contains TPUReplicateContext {}. This is most likely due to the fact that enqueueing embedding data is called inside control flow or a tf.function inside `strategy.run`. This is not supported because outside compilation fails to extract the enqueue ops as the head of a computation.'.format(ops.get_default_graph(), graph))
        return in_tpu_ctx

    @staticmethod
    def preprocess_features(num_replicas_in_sync: int, max_ids_per_chip_per_sample: int, max_minibatches_per_sc: int, num_sc_per_chip: int, num_sc_shards: int, stacked_table_to_tables: Dict[str, Any], table_to_stacked_table_offset: Dict[str, Tuple[str, int, int]], table_to_sample_count: Dict[str, int], feature_to_sample_offset: Dict[str, int], flat_features: Any, flat_inputs: Any, flat_weights: Optional[Any]=None) -> Any:
        if False:
            while True:
                i = 10
        'Function to preprocess features.'
        table_to_list_of_coos = TPUEmbeddingV2._preprocess_inputs_and_weights_to_coo_tensor(flat_inputs, flat_weights, flat_features, stacked_table_to_tables, table_to_stacked_table_offset, feature_to_sample_offset, num_sc_shards)
        (table_to_sorted_coo_tensor, per_replica_table_splits) = TPUEmbeddingV2._get_minibatch_splits_from_coo_tensor(num_replicas_in_sync, table_to_list_of_coos, stacked_table_to_tables, table_to_sample_count, num_sc_per_chip)
        cross_replica_table_splits = gen_collective_ops.collective_gather_v2(input=per_replica_table_splits, group_size=num_replicas_in_sync, group_key=0, instance_key=math_ops.cast(xla_ops.global_iter_id(), dtypes.int32), ordering_token=[])
        table_to_csr_format_tensor = TPUEmbeddingV2._get_minibatches_from_sorted_coo_tensor(num_replicas_in_sync, max_ids_per_chip_per_sample, max_minibatches_per_sc, table_to_sorted_coo_tensor, cross_replica_table_splits, stacked_table_to_tables, table_to_sample_count, num_sc_per_chip)
        return table_to_csr_format_tensor

    def enqueue(self, features: Any, weights: Optional[Any]=None, device: Optional[str]=None) -> Any:
        if False:
            while True:
                i = 10
        'Preprocessing the features on host.'
        nest.assert_same_structure(self._feature_config, features)
        flat_inputs = nest.flatten(features)
        flat_weights = [None] * len(flat_inputs)
        if weights is not None:
            nest.assert_same_structure(self._feature_config, weights)
            flat_weights = nest.flatten(weights)
        in_tpu_context = self._raise_error_for_incorrect_control_flow_context()
        if in_tpu_context:
            return tpu_replication.outside_compilation(TPUEmbeddingV2.preprocess_features, num_replicas_in_sync=self._strategy.num_replicas_in_sync, max_ids_per_chip_per_sample=self.max_ids_per_chip_per_sample, max_minibatches_per_sc=self.max_minibatches_per_sc, num_sc_per_chip=self._num_sc_per_chip, num_sc_shards=self._num_sc_shards, stacked_table_to_tables=self._stacked_table_to_tables, table_to_stacked_table_offset=self._table_to_stacked_table_offset, table_to_sample_count=self._table_to_sample_count, feature_to_sample_offset=self._feature_to_sample_offset, flat_features=self._flat_features, flat_inputs=flat_inputs, flat_weights=flat_weights)
        elif device is None:
            tpu_devices = self._strategy.extended._tpu_devices
            with ops.device(device_util.get_host_for_device(tpu_devices[0][0])):
                return TPUEmbeddingV2.preprocess_features(num_replicas_in_sync=self._strategy.num_replicas_in_sync, max_ids_per_chip_per_sample=self.max_ids_per_chip_per_sample, max_minibatches_per_sc=self.max_minibatches_per_sc, num_sc_per_chip=self._num_sc_per_chip, num_sc_shards=self._num_sc_shards, stacked_table_to_tables=self._stacked_table_to_tables, table_to_stacked_table_offset=self._table_to_stacked_table_offset, table_to_sample_count=self._table_to_sample_count, feature_to_sample_offset=self._feature_to_sample_offset, flat_features=self._flat_features, flat_inputs=flat_inputs, flat_weights=flat_weights)
        else:
            device_spec = tf_device.DeviceSpec.from_string(device)
            if device_spec.device_type != 'TPU':
                raise ValueError('Non-TPU device {} passed to enqueue.'.format(device))
            with ops.device(device_util.get_host_for_device(device)):
                return TPUEmbeddingV2.preprocess_features(num_replicas_in_sync=self._strategy.num_replicas_in_sync, max_ids_per_chip_per_sample=self.max_ids_per_chip_per_sample, max_minibatches_per_sc=self.max_minibatches_per_sc, num_sc_per_chip=self._num_sc_per_chip, num_sc_shards=self._num_sc_shards, stacked_table_to_tables=self._stacked_table_to_tables, table_to_stacked_table_offset=self._table_to_stacked_table_offset, table_to_sample_count=self._table_to_sample_count, feature_to_sample_offset=self._feature_to_sample_offset, flat_features=self._flat_features, flat_inputs=flat_inputs, flat_weights=flat_weights)

    def _copy_tensors_to_device(self, partitioned_tensors: Dict[str, Any]) -> Any:
        if False:
            while True:
                i = 10
        'Copy tensors to device.'
        partitioned_device_tensors = {}
        for table_name in partitioned_tensors:
            partitioned_tensor = partitioned_tensors[table_name][0]
            row_pointers_unpadded_size = partitioned_tensors[table_name][1]
            ids_unpadded_size = partitioned_tensors[table_name][2]
            (row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains) = xla_ops.tpu_copy_with_dynamic_shape([partitioned_tensor.row_pointers, partitioned_tensor.sorted_sample_ids, partitioned_tensor.sorted_token_ids, partitioned_tensor.sorted_gains], [row_pointers_unpadded_size, ids_unpadded_size, ids_unpadded_size, ids_unpadded_size])
            (row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains) = xla_ops.tpu_annotate_tensors_with_dynamic_shape([row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains])
            partitioned_device_tensors[table_name] = PartitionedCsrFormatTensor(row_pointers=row_pointers, sorted_sample_ids=sorted_sample_ids, sorted_token_ids=sorted_token_ids, sorted_gains=sorted_gains, sample_count=partitioned_tensor.sample_count, num_minibatches_per_physical_sparse_core=partitioned_tensor.num_minibatches_per_physical_sparse_core)
        return partitioned_device_tensors

    def dequeue(self, partitioned_tensors: Tuple[Dict[str, PartitionedCsrFormatTensor], int, int]) -> Tuple[Any, Dict[str, PartitionedCsrFormatTensor]]:
        if False:
            for i in range(10):
                print('nop')
        'Perform embedding lookup.'
        context = EmbeddingPipeliningContext(_PIPELINE_MODE_FORWARD, self._pipelining)
        context.Enter()
        partitioned_tensors = tpu_replication.outside_compilation(self._copy_tensors_to_device, partitioned_tensors=partitioned_tensors)
        activations = {}
        num_minibatches_per_physical_sparse_core = list(partitioned_tensors.values())[0].num_minibatches_per_physical_sparse_core
        for table_name in self._stacked_table_to_tables:
            partitioned_tensor = partitioned_tensors[table_name]
            table = self.variables[table_name]['parameters']
            quantization_config = self._quantization_configs[table_name]
            if not isinstance(partitioned_tensor, PartitionedCsrFormatTensor):
                raise ValueError(f'Expect PartitionedCsrFormatTensor but get {type(partitioned_tensor)}.')
            activation = xla_ops.xla_sparse_dense_matmul_with_csr_input(row_pointers=partitioned_tensor.row_pointers, sorted_sample_ids=partitioned_tensor.sorted_sample_ids, sorted_token_ids=partitioned_tensor.sorted_token_ids, sorted_gains=partitioned_tensor.sorted_gains, input_size=self._table_to_sample_count[table_name], embedding_table=table, num_minibatches_per_physical_sparse_core=num_minibatches_per_physical_sparse_core, quantization_config_low=quantization_config.lower if quantization_config else 0, quantization_config_high=quantization_config.upper if quantization_config else 0, quantization_config_num_buckets=quantization_config.num_buckets if quantization_config else 0, table_name=table_name)
            activations[table_name] = activation
        context.Exit()
        activations = self._unstack_activations(activations)
        return (activations, partitioned_tensors)

    def embedding_lookup(self, features: Any, weights: Optional[Any]=None) -> Tuple[Any, Dict[str, PartitionedCsrFormatTensor]]:
        if False:
            for i in range(10):
                print('nop')
        'Perform embedding lookup on the input feature.\n\n    Args:\n      features: A nested structure of `tf.Tensor`s, `tf.SparseTensor`s or\n        `tf.RaggedTensor`s, with the same structure as `feature_config`. Inputs\n        will be downcast to `tf.int32`. Only one type out of `tf.SparseTensor`\n        or `tf.RaggedTensor` is supported per call.\n      weights: If not `None`, a nested structure of `tf.Tensor`s,\n        `tf.SparseTensor`s or `tf.RaggedTensor`s, matching the above, except\n        that the tensors should be of float type (and they will be downcast to\n        `tf.float32`). For `tf.SparseTensor`s we assume the `indices` are the\n        same for the parallel entries from `features` and similarly for\n        `tf.RaggedTensor`s we assume the row_splits are the same.\n\n    Raises:\n      ValueError: If the input feature is not one of the Tensor, SparseTensor or\n        RaggedTensor type.\n      TypeError: If the type of any sequence in `features` does not match\n        corresponding sequence in `feature_config`. Similarly for `weights`, if\n        not `None`.\n\n    Returns:\n      packed_activations: Embedding lookup results packed as the same sequence\n        of the input feature.\n      packed_output: A dict of PartitionedCsrFormatTensors.\n    '
        if not self._built:
            self._maybe_build()
        context = EmbeddingPipeliningContext(_PIPELINE_MODE_FORWARD, self._pipelining)
        context.Enter()
        partitioned_tensors = self.enqueue(features, weights)
        context.Exit()
        result = self.dequeue(partitioned_tensors)
        return result

    @staticmethod
    def _experimental_preprocess_features(num_replicas_in_sync: int, max_ids_per_chip_per_sample: int, max_minibatches_per_sc: int, num_sc_per_chip: int, num_sc_shards: int, stacked_table_to_tables: Dict[str, Any], table_to_stacked_table_offset: Dict[str, Tuple[str, int, int]], table_to_sample_count: Dict[str, int], feature_to_sample_offset: Dict[str, int], flat_features: Any, flat_inputs: Any, flat_weights: Optional[Any]=None) -> Any:
        if False:
            i = 10
            return i + 15
        'Function to preprocess features.'
        table_to_list_of_coos = TPUEmbeddingV2._experimental_preprocess_inputs_and_weights_to_list_of_coo_tensors(flat_inputs, flat_weights, flat_features, stacked_table_to_tables, table_to_stacked_table_offset, feature_to_sample_offset, num_sc_per_chip, table_to_sample_count, num_sc_shards)
        (table_to_sorted_coo_tensor, is_minibatching_needed_per_replica) = TPUEmbeddingV2._experimental_sort_list_of_coo_tensors(num_replicas_in_sync, table_to_list_of_coos, stacked_table_to_tables, num_sc_per_chip)
        is_minibatching_needed_cross_replica = gen_collective_ops.collective_gather_v2(input=is_minibatching_needed_per_replica, group_size=num_replicas_in_sync, group_key=0, instance_key=math_ops.cast(xla_ops.global_iter_id(), dtypes.int32), ordering_token=[])
        table_to_csr_format_tensor = cond.cond(math_ops.equal(math_ops.reduce_sum(is_minibatching_needed_cross_replica), 0), lambda : TPUEmbeddingV2._experimental_get_single_minibatch_from_sorted_coo_tensor(num_replicas_in_sync, max_ids_per_chip_per_sample, max_minibatches_per_sc, table_to_sorted_coo_tensor, stacked_table_to_tables, table_to_sample_count, num_sc_per_chip), lambda : TPUEmbeddingV2._experimental_get_multiple_minibatches_from_sorted_coo_tensor(num_replicas_in_sync, max_ids_per_chip_per_sample, max_minibatches_per_sc, table_to_sorted_coo_tensor, stacked_table_to_tables, table_to_sample_count, num_sc_per_chip), strict=True)
        return table_to_csr_format_tensor

    @staticmethod
    def _experimental_convert_input_feature_to_list_of_coo_tensors(input_feature: Union[tensor.Tensor, sparse_tensor.SparseTensor, ragged_tensor.RaggedTensor], weight: Optional[tensor.Tensor], feature_config: tpu_embedding_v2_utils.FeatureConfig, row_offset: int, col_offset: int, col_shift: int, vocab_size: int, num_sc_per_chip: int, num_sc_shards: int, stacked_table_sample_count: int) -> Any:
        if False:
            i = 10
            return i + 15
        'Convert any of the expected input types to a COO format.'
        sample_count = functools.reduce(operator.mul, feature_config.output_shape)
        if isinstance(input_feature, tensor.Tensor):
            input_feature = array_ops.reshape(input_feature, [-1])
            if weight is None:
                weight = array_ops.ones_like(input_feature, dtype=dtypes.float32)
            elif isinstance(weight, tensor.Tensor):
                weight = array_ops.reshape(weight, [-1])
            else:
                raise ValueError(f'Expect weight to be Tensor type but got {type(weight)}')
            (row_ids_list, col_ids_list, gains_list) = xla_ops.convert_to_list_of_coo_tensors(indices_or_row_splits=array_ops.zeros((0,), dtype=dtypes.int32), values=math_ops.cast(input_feature, dtype=dtypes.int32), weights=math_ops.cast(weight, dtypes.float32), sample_count=sample_count, combiner=feature_config.table.combiner, num_sc_per_chip=num_sc_per_chip)
        elif isinstance(input_feature, sparse_tensor.SparseTensor):
            if weight is None:
                weight = array_ops.ones_like(input_feature.values, dtype=dtypes.float32)
            elif isinstance(weight, sparse_tensor.SparseTensor):
                weight = weight.values
            else:
                raise ValueError(f'Expect weight to be SparseTensor type but got {type(weight)}')
            (row_ids_list, col_ids_list, gains_list) = xla_ops.convert_to_list_of_coo_tensors(indices_or_row_splits=math_ops.cast(input_feature.indices, dtype=dtypes.int32), values=math_ops.cast(input_feature.values, dtype=dtypes.int32), weights=math_ops.cast(weight, dtypes.float32), sample_count=sample_count, combiner=feature_config.table.combiner, num_sc_per_chip=num_sc_per_chip)
        elif isinstance(input_feature, ragged_tensor.RaggedTensor):
            if not weight:
                weight = array_ops.ones_like(input_feature.values, dtype=dtypes.float32)
            elif isinstance(weight, ragged_tensor.RaggedTensor):
                weight = weight.values
            else:
                raise ValueError(f'Expect weight to be RaggedTensor type but got {type(weight)}')
            (row_ids_list, col_ids_list, gains_list) = xla_ops.convert_to_list_of_coo_tensors(indices_or_row_splits=math_ops.cast(input_feature.row_splits, dtype=dtypes.int32), values=math_ops.cast(input_feature.values, dtype=dtypes.int32), weights=math_ops.cast(weight, dtypes.float32), sample_count=sample_count, combiner=feature_config.table.combiner, num_sc_per_chip=num_sc_per_chip)
        else:
            raise ValueError(f'Input of unknown type {type(input_feature)}. Please only pass Tensor, SparseTensor or RaggedTensor as input to embedding lookup.')
        for i in range(num_sc_per_chip):
            row_ids_list[i] = row_ids_list[i] % (sample_count // num_sc_per_chip) + int(row_offset // num_sc_per_chip) + int(stacked_table_sample_count // num_sc_per_chip) * i
            col_ids_list[i] = (col_ids_list[i] + col_shift) % num_sc_shards + col_ids_list[i] // num_sc_shards * num_sc_shards + col_offset
        return (row_ids_list, col_ids_list, gains_list, sample_count)

    @staticmethod
    def _experimental_preprocess_inputs_and_weights_to_list_of_coo_tensors(flat_inputs: Any, flat_weights: Any, flat_features: Any, stacked_table_to_tables: Dict[str, Any], table_to_stacked_table_offset: Dict[str, Tuple[str, int, int]], feature_to_sample_offset: Dict[str, int], num_sc_per_chip: int, stacked_table_to_sample_count: Dict[str, int], num_sc_shards: int) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        'Convert the raw inputs into list of coo tensors.'
        table_to_list_of_coos = {table_name: ([[], [], [], []], [[], [], [], []], [[], [], [], []], [], []) for table_name in stacked_table_to_tables}
        for (inp, weight, (feature_path, feature)) in zip(flat_inputs, flat_weights, flat_features):
            (table_name, col_offset, col_shift) = table_to_stacked_table_offset[feature.table.name]
            stacked_table_sample_count = stacked_table_to_sample_count[table_name]
            row_offset = feature_to_sample_offset[feature_path]
            (row_ids_list, col_ids_list, gains_list, sample_count) = TPUEmbeddingV2._experimental_convert_input_feature_to_list_of_coo_tensors(inp, weight, feature, row_offset, col_offset, col_shift, feature.table.vocabulary_size, num_sc_per_chip, num_sc_shards, stacked_table_sample_count)
            for i in range(num_sc_per_chip):
                table_to_list_of_coos[table_name][0][i].append(row_ids_list[i])
                table_to_list_of_coos[table_name][1][i].append(col_ids_list[i])
                table_to_list_of_coos[table_name][2][i].append(gains_list[i])
            table_to_list_of_coos[table_name][3].append(sample_count // num_sc_per_chip)
            table_to_list_of_coos[table_name][4].append(col_offset)
        return table_to_list_of_coos

    @staticmethod
    def _experimental_sort_list_of_coo_tensors(num_replicas_in_sync: int, table_to_list_of_coos: Dict[str, Any], stacked_table_to_tables: Dict[str, Any], num_sc_per_chip: int) -> Tuple[Dict[str, Any], List[tensor.Tensor]]:
        if False:
            while True:
                i = 10
        'Sort the coo tensors by replica.'
        table_to_sorted_coo_tensor = {table_name: ([], [], [], []) for table_name in stacked_table_to_tables}
        is_minibatching_needed_per_table = []
        for table_name in stacked_table_to_tables:
            feature_width = stacked_table_to_tables[table_name][0].dim
            total_vocab_size = sum([table.vocabulary_size for table in stacked_table_to_tables[table_name]])
            for i in range(num_sc_per_chip):
                row_ids_list = table_to_list_of_coos[table_name][0][i]
                col_ids_list = table_to_list_of_coos[table_name][1][i]
                gains_list = table_to_list_of_coos[table_name][2][i]
                sample_count_list = table_to_list_of_coos[table_name][3]
                col_offset_list = table_to_list_of_coos[table_name][4]
                (sorted_row_ids, sorted_col_ids, sorted_gains, id_counts, is_minibatch_needed) = xla_ops.sort_list_of_coo_tensors_with_physical_replica(row_ids_list=row_ids_list, col_ids_list=col_ids_list, gains_list=gains_list, sample_count_list=sample_count_list, col_offset_list=col_offset_list, num_replica=num_replicas_in_sync, table_vocab_size=total_vocab_size, feature_width=feature_width, num_sc_per_chip=num_sc_per_chip, table_name=table_name)
                table_to_sorted_coo_tensor[table_name][0].append(sorted_row_ids)
                table_to_sorted_coo_tensor[table_name][1].append(sorted_col_ids)
                table_to_sorted_coo_tensor[table_name][2].append(sorted_gains)
                table_to_sorted_coo_tensor[table_name][3].append(id_counts)
                is_minibatching_needed_per_table.append(math_ops.cast(is_minibatch_needed, dtypes.int32))
        return (table_to_sorted_coo_tensor, is_minibatching_needed_per_table)

    @staticmethod
    def _experimental_get_minibatch_splits_from_sorted_coo_tensor(num_replicas_in_sync: int, table_to_sorted_coo_tensor: Dict[str, Any], stacked_table_to_tables: Dict[str, Any], table_to_sample_count: Dict[str, int], num_sc_per_chip: int) -> Tuple[Dict[str, Any], List[tensor.Tensor]]:
        if False:
            i = 10
            return i + 15
        'Compute minibatch splits from the sorted coo tensor.'
        table_to_sorted_coo_tensor_with_minibatch = {table_name: ([], [], [], []) for table_name in stacked_table_to_tables}
        per_replica_table_splits = []
        for table_name in stacked_table_to_tables:
            feature_width = stacked_table_to_tables[table_name][0].dim
            total_vocab_size = sum([table.vocabulary_size for table in stacked_table_to_tables[table_name]])
            (sorted_row_ids_list, sorted_col_ids_list, sorted_gains_list, id_counts_list) = table_to_sorted_coo_tensor[table_name]
            for i in range(num_sc_per_chip):
                (sorted_row_ids, sorted_col_ids, sorted_gains, id_counts, splits) = xla_ops.get_multiple_minibatches_splits_with_physical_replica(sorted_row_ids=sorted_row_ids_list[i], sorted_col_ids=sorted_col_ids_list[i], sorted_gains=sorted_gains_list[i], id_counts=id_counts_list[i], num_replica=num_replicas_in_sync, sample_count_per_sc=table_to_sample_count[table_name] // num_sc_per_chip, table_vocab_size=total_vocab_size, feature_width=feature_width, num_sc_per_chip=num_sc_per_chip, table_name=table_name)
                table_to_sorted_coo_tensor_with_minibatch[table_name][0].append(sorted_row_ids)
                table_to_sorted_coo_tensor_with_minibatch[table_name][1].append(sorted_col_ids)
                table_to_sorted_coo_tensor_with_minibatch[table_name][2].append(sorted_gains)
                table_to_sorted_coo_tensor_with_minibatch[table_name][3].append(id_counts)
                per_replica_table_splits.append(splits)
        return (table_to_sorted_coo_tensor_with_minibatch, per_replica_table_splits)

    @staticmethod
    def _experimental_get_multiple_minibatches_from_sorted_coo_tensor(num_replicas_in_sync: int, max_ids_per_chip_per_sample: int, max_minibatches_per_sc: int, table_to_sorted_coo_tensor: Dict[str, Any], stacked_table_to_tables: Dict[str, Any], table_to_sample_count: Dict[str, int], num_sc_per_chip: int) -> Any:
        if False:
            i = 10
            return i + 15
        'Get multiple minibatches from the sorted coo tensor.'
        (table_to_sorted_coo_tensor_with_minibatch, per_replica_table_splits) = TPUEmbeddingV2._experimental_get_minibatch_splits_from_sorted_coo_tensor(num_replicas_in_sync, table_to_sorted_coo_tensor, stacked_table_to_tables, table_to_sample_count, num_sc_per_chip)
        cross_replica_table_splits = gen_collective_ops.collective_gather_v2(input=per_replica_table_splits, group_size=num_replicas_in_sync, group_key=1, instance_key=math_ops.cast(xla_ops.global_iter_id(), dtypes.int32), ordering_token=[])
        table_to_csr_format_tensor = {}
        for table_name in stacked_table_to_tables:
            (sorted_row_ids_list, sorted_col_ids_list, sorted_gains_list, id_counts_list) = table_to_sorted_coo_tensor_with_minibatch[table_name]
            feature_width = stacked_table_to_tables[table_name][0].dim
            total_vocab_size = sum([table.vocabulary_size for table in stacked_table_to_tables[table_name]])
            (row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains, row_pointers_unpadded_size, ids_unpadded_size, num_minibatches_per_physical_sparse_core) = xla_ops.convert_to_csr_wrapped_coo_with_physical_replica(sorted_row_ids_list=sorted_row_ids_list, sorted_col_ids_list=sorted_col_ids_list, sorted_gains_list=sorted_gains_list, id_counts_list=id_counts_list, splits=cross_replica_table_splits, sample_count_per_sc=table_to_sample_count[table_name] // num_sc_per_chip, num_replica=num_replicas_in_sync, max_minibatches_per_sc=max_minibatches_per_sc, max_ids_per_chip_per_sample=max_ids_per_chip_per_sample, table_vocab_size=total_vocab_size, feature_width=feature_width, table_name=table_name)
            table_to_csr_format_tensor[table_name] = (PartitionedCsrFormatTensor(row_pointers=row_pointers, sorted_sample_ids=sorted_sample_ids, sorted_token_ids=sorted_token_ids, sorted_gains=sorted_gains, sample_count=table_to_sample_count[table_name], num_minibatches_per_physical_sparse_core=num_minibatches_per_physical_sparse_core), row_pointers_unpadded_size, ids_unpadded_size)
        return table_to_csr_format_tensor

    @staticmethod
    def _experimental_get_single_minibatch_from_sorted_coo_tensor(num_replicas_in_sync: int, max_ids_per_chip_per_sample: int, max_minibatches_per_sc: int, table_to_sorted_coo_tensor: Dict[str, Any], stacked_table_to_tables: Dict[str, Any], table_to_sample_count: Dict[str, int], num_sc_per_chip: int) -> Any:
        if False:
            print('Hello World!')
        'Get a single minibatch from the sorted coo tensor.'
        table_to_csr_format_tensor = {}
        for table_name in stacked_table_to_tables:
            (sorted_row_ids_list, sorted_col_ids_list, sorted_gains_list, id_counts_list) = table_to_sorted_coo_tensor[table_name]
            feature_width = stacked_table_to_tables[table_name][0].dim
            total_vocab_size = sum([table.vocabulary_size for table in stacked_table_to_tables[table_name]])
            (row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains, row_pointers_unpadded_size, ids_unpadded_size, num_minibatches_per_physical_sparse_core) = xla_ops.convert_to_csr_wrapped_coo_with_physical_replica(sorted_row_ids_list=sorted_row_ids_list, sorted_col_ids_list=sorted_col_ids_list, sorted_gains_list=sorted_gains_list, id_counts_list=id_counts_list, splits=constant_op.constant(0, dtype=dtypes.int64), sample_count_per_sc=table_to_sample_count[table_name] // num_sc_per_chip, num_replica=num_replicas_in_sync, max_minibatches_per_sc=max_minibatches_per_sc, max_ids_per_chip_per_sample=max_ids_per_chip_per_sample, table_vocab_size=total_vocab_size, feature_width=feature_width, table_name=table_name)
            table_to_csr_format_tensor[table_name] = (PartitionedCsrFormatTensor(row_pointers=row_pointers, sorted_sample_ids=sorted_sample_ids, sorted_token_ids=sorted_token_ids, sorted_gains=sorted_gains, sample_count=table_to_sample_count[table_name], num_minibatches_per_physical_sparse_core=num_minibatches_per_physical_sparse_core), row_pointers_unpadded_size, ids_unpadded_size)
        return table_to_csr_format_tensor

    def _experimental_unstack_activations(self, activations: Dict[str, tensor.Tensor]):
        if False:
            for i in range(10):
                print('nop')
        'Untack the incoming per table activations into per feature.'
        flattened_activations = []
        table_to_current_offset = {table_name: 0 for table_name in self._stacked_table_to_tables}
        for table_name in self._stacked_table_to_tables:
            activation_shape = activations[table_name].shape
            activations[table_name] = array_ops.reshape(activations[table_name], [self._num_sc_per_chip, -1, activation_shape[-1]])
        for (_, feature) in self._flat_features:
            sample_count = functools.reduce(operator.mul, feature.output_shape)
            table_name = self._table_to_stacked_table_offset[feature.table.name][0]
            extra_cols = self._table_to_padding_columns[feature.table.name]
            activation = array_ops.slice(activations[table_name], [0, table_to_current_offset[table_name], 0], [self._num_sc_per_chip, sample_count // self._num_sc_per_chip, feature.table.dim - extra_cols])
            activation = array_ops.reshape(activation, list(feature.output_shape) + [feature.table.dim - extra_cols])
            flattened_activations.append(activation)
            table_to_current_offset[table_name] += sample_count // self._num_sc_per_chip
        return nest.pack_sequence_as(self._feature_config, flattened_activations)

    def _experimental_stack_gradients(self, gradients):
        if False:
            while True:
                i = 10
        'Stack the incoming gradients to per table gradients.'
        table_to_gradient_list = {table_name: [[], [], [], []] for table_name in self._stacked_table_to_tables}
        flattend_gradients = nest.flatten(gradients)
        for (gradient, (path, feature)) in zip(flattend_gradients, self._flat_features):
            sample_count = functools.reduce(operator.mul, feature.output_shape)
            if gradient is not None and (not isinstance(gradient, tensor.Tensor)):
                raise ValueError(f'found non-tensor type: {type(gradient)} at path {path}.')
            if gradient is None:
                logging.warning('No gradient passed for feature %s, sending zero gradient. This may not be correct behavior for certain optimizers like Adam.', path)
                gradient = array_ops.zeros((sample_count, feature.table.dim), dtype=dtypes.float32)
            table_name = self._table_to_stacked_table_offset[feature.table.name][0]
            extra_cols = self._table_to_padding_columns[feature.table.name]
            gradient = array_ops.reshape(gradient, [-1, feature.table.dim - extra_cols])
            if extra_cols != 0:
                gradient = array_ops.pad(gradient, [[0, 0], [0, extra_cols]])
                gradient.set_shape([sample_count, feature.table.dim])
            per_sc_sample_count = sample_count // self._num_sc_per_chip
            for i in range(self._num_sc_per_chip):
                table_to_gradient_list[table_name][i].append(array_ops.slice(gradient, [i * per_sc_sample_count, 0], [per_sc_sample_count, feature.table.dim]))
        for table_name in table_to_gradient_list:
            table_to_gradient_list[table_name] = array_ops.concat([array_ops.concat(table_to_gradient_list[table_name][i], axis=0) for i in range(self._num_sc_per_chip)], axis=0)
        return table_to_gradient_list

def extract_variable_info(kwargs: Any) -> Tuple[str, Tuple[int, ...], dtypes.DType, Callable[[], Any]]:
    if False:
        while True:
            i = 10
    'Extracts the variable creation attributes from the kwargs.\n\n  Args:\n    kwargs: a dict of keyword arguments that were passed to a variable creator\n      scope.\n\n  Returns:\n    A tuple of variable name, shape, dtype, initialization function.\n  '
    if isinstance(kwargs['initial_value'], functools.partial) and ('shape' in kwargs['initial_value'].keywords or kwargs['initial_value'].args):
        if 'shape' in kwargs['initial_value'].keywords:
            shape = kwargs['initial_value'].keywords['shape']
        else:
            shape = kwargs['initial_value'].args[0]
        return (kwargs['name'], shape, kwargs['initial_value'].keywords.get('dtype', kwargs['dtype']), kwargs['initial_value'].func)
    elif 'shape' not in kwargs or kwargs['shape'] is None or (not callable(kwargs['initial_value'])):
        raise ValueError('Unable to extract initializer function and shape from {}. Please either pass a function that expects a shape and dtype as the initial value for your variable or functools.partial object with the shape and dtype kwargs set. This is needed so that we can initialize the shards of the ShardedVariable locally.'.format(kwargs['initial_value']))
    else:
        return (kwargs['name'], kwargs['shape'], kwargs['dtype'], kwargs['initial_value'])

def is_checkpoint_initial_value(initial_value: Any) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Whether the initial value is from checkpoint.'
    return isinstance(initial_value, base.CheckpointInitialValue) or isinstance(initial_value, base.CheckpointInitialValueCallable) or (isinstance(initial_value, functools.partial) and isinstance(initial_value.func, base.CheckpointInitialValueCallable))

def make_sharded_variable_creator(strategy: distribute_lib.Strategy) -> Callable[..., Any]:
    if False:
        for i in range(10):
            print('nop')
    'Create a variable creator which shards across all the tpu device.\n\n  Args:\n    strategy: a TPUStrategy object.\n\n  Returns:\n    The sharded variable creator.\n  '
    tpu_devices = strategy.extended._tpu_devices

    def _create_sharded_variable(next_creator, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Create a TPUEmbeddingShardedVariable.'
        kwargs['skip_mirrored_creator'] = True
        shard_dim = 0
        (num_replicas, num_cores_per_replica) = tpu_devices.shape
        is_ckpt_init_value = is_checkpoint_initial_value(kwargs['initial_value'])
        arg_spec = tf_inspect.getfullargspec(kwargs['initial_value'])
        if is_ckpt_init_value and 'shard_info' not in arg_spec.args and ('shard_info' not in arg_spec.kwonlyargs):
            raise ValueError('When a sharded variable is initialized from a checkpoint, shard_info must be in arguments of the init function.')
        (name, shape, dtype, unwrapped_initial_value) = extract_variable_info(kwargs)
        shape = ops.tensor_shape.TensorShape(shape)
        num_devices = num_replicas * num_cores_per_replica
        if shape[shard_dim] % num_devices != 0:
            raise ValueError('Only evenly sharding across devices is currently supported. Got shape {} and {} devices'.format(shape, num_devices))
        partition_shape = shape.as_list()
        partition_shape[shard_dim] = partition_shape[shard_dim] // num_devices
        unwrapped_arg_spec = tf_inspect.getargspec(unwrapped_initial_value)
        sharding_aware = 'shard_info' in unwrapped_arg_spec.args
        variables = []
        partition_offset = [0] * len(shape)
        for replica_id in range(num_replicas):
            for logic_core_id in range(num_cores_per_replica):
                with ops.device(tpu_devices[replica_id][logic_core_id]):
                    kwargs['name'] = f'{name}/{replica_id}'
                    kwargs['shape'] = partition_shape
                    if sharding_aware:
                        shard_info = base.ShardInfo(tensor_shape.as_shape(partition_shape), copy.deepcopy(partition_offset))
                        kwargs['initial_value'] = functools.partial(kwargs['initial_value'], shard_info=shard_info)
                        partition_offset[shard_dim] += partition_shape[shard_dim]
                    else:
                        kwargs['initial_value'] = functools.partial(unwrapped_initial_value, shape=partition_shape, dtype=dtype)
                    variables.append(next_creator(*args, **kwargs))
        result = TPUEmbeddingShardedVariable(strategy, variables, tf_variables.VariableAggregation.NONE, None)
        return result
    return _create_sharded_variable