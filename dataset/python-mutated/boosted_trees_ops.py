"""Ops for boosted_trees."""
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_boosted_trees_ops
from tensorflow.python.ops import resources
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_aggregate_stats
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_bucketize
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_calculate_best_feature_split as calculate_best_feature_split
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_calculate_best_feature_split_v2 as calculate_best_feature_split_v2
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_calculate_best_gains_per_feature as calculate_best_gains_per_feature
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_center_bias as center_bias
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_create_quantile_stream_resource as create_quantile_stream_resource
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_example_debug_outputs as example_debug_outputs
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_make_quantile_summaries as make_quantile_summaries
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_make_stats_summary as make_stats_summary
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_predict as predict
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_quantile_stream_resource_add_summaries as quantile_add_summaries
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_quantile_stream_resource_deserialize as quantile_resource_deserialize
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_quantile_stream_resource_flush as quantile_flush
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_quantile_stream_resource_get_bucket_boundaries as get_bucket_boundaries
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_quantile_stream_resource_handle_op as quantile_resource_handle_op
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_sparse_aggregate_stats
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_sparse_calculate_best_feature_split as sparse_calculate_best_feature_split
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_training_predict as training_predict
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_update_ensemble as update_ensemble
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_update_ensemble_v2 as update_ensemble_v2
from tensorflow.python.ops.gen_boosted_trees_ops import is_boosted_trees_quantile_stream_resource_initialized as is_quantile_resource_initialized
from tensorflow.python.training import saver

class PruningMode:
    """Class for working with Pruning modes."""
    (NO_PRUNING, PRE_PRUNING, POST_PRUNING) = range(0, 3)
    _map = {'none': NO_PRUNING, 'pre': PRE_PRUNING, 'post': POST_PRUNING}

    @classmethod
    def from_str(cls, mode):
        if False:
            i = 10
            return i + 15
        if mode in cls._map:
            return cls._map[mode]
        else:
            raise ValueError('pruning_mode mode must be one of: {}. Found: {}'.format(', '.join(sorted(cls._map)), mode))

class QuantileAccumulatorSaveable(saver.BaseSaverBuilder.SaveableObject):
    """SaveableObject implementation for QuantileAccumulator."""

    def __init__(self, resource_handle, create_op, num_streams, name):
        if False:
            return 10
        self.resource_handle = resource_handle
        self._num_streams = num_streams
        self._create_op = create_op
        bucket_boundaries = get_bucket_boundaries(self.resource_handle, self._num_streams)
        slice_spec = ''
        specs = []

        def make_save_spec(tensor, suffix):
            if False:
                i = 10
                return i + 15
            return saver.BaseSaverBuilder.SaveSpec(tensor, slice_spec, name + suffix)
        for i in range(self._num_streams):
            specs += [make_save_spec(bucket_boundaries[i], '_bucket_boundaries_' + str(i))]
        super(QuantileAccumulatorSaveable, self).__init__(self.resource_handle, specs, name)

    def restore(self, restored_tensors, unused_tensor_shapes):
        if False:
            return 10
        bucket_boundaries = restored_tensors
        with ops.control_dependencies([self._create_op]):
            return quantile_resource_deserialize(self.resource_handle, bucket_boundaries=bucket_boundaries)

class QuantileAccumulator:
    """SaveableObject implementation for QuantileAccumulator.

     The bucket boundaries are serialized and deserialized from checkpointing.
  """

    def __init__(self, epsilon, num_streams, num_quantiles, name=None, max_elements=None):
        if False:
            while True:
                i = 10
        del max_elements
        self._eps = epsilon
        self._num_streams = num_streams
        self._num_quantiles = num_quantiles
        with ops.name_scope(name, 'QuantileAccumulator') as name:
            self._name = name
            self.resource_handle = self._create_resource()
            self._init_op = self._initialize()
            is_initialized_op = self.is_initialized()
        resources.register_resource(self.resource_handle, self._init_op, is_initialized_op)
        ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, QuantileAccumulatorSaveable(self.resource_handle, self._init_op, self._num_streams, self.resource_handle.name))

    def _create_resource(self):
        if False:
            print('Hello World!')
        return quantile_resource_handle_op(container='', shared_name=self._name, name=self._name)

    def _initialize(self):
        if False:
            for i in range(10):
                print('nop')
        return create_quantile_stream_resource(self.resource_handle, self._eps, self._num_streams)

    @property
    def initializer(self):
        if False:
            print('Hello World!')
        if self._init_op is None:
            self._init_op = self._initialize()
        return self._init_op

    def is_initialized(self):
        if False:
            i = 10
            return i + 15
        return is_quantile_resource_initialized(self.resource_handle)

    def _serialize_to_tensors(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('When the need arises, TF2 compatibility can be added by implementing this method, along with _restore_from_tensors below.')

    def _restore_from_tensors(self, restored_tensors):
        if False:
            print('Hello World!')
        raise NotImplementedError('When the need arises, TF2 compatibility can be added by implementing this method, along with _serialize_to_tensors above.')

    def add_summaries(self, float_columns, example_weights):
        if False:
            return 10
        summaries = make_quantile_summaries(float_columns, example_weights, self._eps)
        summary_op = quantile_add_summaries(self.resource_handle, summaries)
        return summary_op

    def flush(self):
        if False:
            while True:
                i = 10
        return quantile_flush(self.resource_handle, self._num_quantiles)

    def get_bucket_boundaries(self):
        if False:
            print('Hello World!')
        return get_bucket_boundaries(self.resource_handle, self._num_streams)

class _TreeEnsembleSavable(saver.BaseSaverBuilder.SaveableObject):
    """SaveableObject implementation for TreeEnsemble."""

    def __init__(self, resource_handle, create_op, name):
        if False:
            print('Hello World!')
        'Creates a _TreeEnsembleSavable object.\n\n    Args:\n      resource_handle: handle to the decision tree ensemble variable.\n      create_op: the op to initialize the variable.\n      name: the name to save the tree ensemble variable under.\n    '
        (stamp_token, serialized) = gen_boosted_trees_ops.boosted_trees_serialize_ensemble(resource_handle)
        slice_spec = ''
        specs = [saver.BaseSaverBuilder.SaveSpec(stamp_token, slice_spec, name + '_stamp'), saver.BaseSaverBuilder.SaveSpec(serialized, slice_spec, name + '_serialized')]
        super(_TreeEnsembleSavable, self).__init__(resource_handle, specs, name)
        self.resource_handle = resource_handle
        self._create_op = create_op

    def restore(self, restored_tensors, unused_restored_shapes):
        if False:
            return 10
        "Restores the associated tree ensemble from 'restored_tensors'.\n\n    Args:\n      restored_tensors: the tensors that were loaded from a checkpoint.\n      unused_restored_shapes: the shapes this object should conform to after\n        restore. Not meaningful for trees.\n\n    Returns:\n      The operation that restores the state of the tree ensemble variable.\n    "
        with ops.control_dependencies([self._create_op]):
            return gen_boosted_trees_ops.boosted_trees_deserialize_ensemble(self.resource_handle, stamp_token=restored_tensors[0], tree_ensemble_serialized=restored_tensors[1])

class TreeEnsemble:
    """Creates TreeEnsemble resource."""

    def __init__(self, name, stamp_token=0, is_local=False, serialized_proto=''):
        if False:
            for i in range(10):
                print('nop')
        self._stamp_token = stamp_token
        self._serialized_proto = serialized_proto
        self._is_local = is_local
        with ops.name_scope(name, 'TreeEnsemble') as name:
            self._name = name
            self.resource_handle = self._create_resource()
            self._init_op = self._initialize()
            is_initialized_op = self.is_initialized()
            if not is_local:
                ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, _TreeEnsembleSavable(self.resource_handle, self.initializer, self.resource_handle.name))
            resources.register_resource(self.resource_handle, self.initializer, is_initialized_op, is_shared=not is_local)

    def _create_resource(self):
        if False:
            return 10
        return gen_boosted_trees_ops.boosted_trees_ensemble_resource_handle_op(container='', shared_name=self._name, name=self._name)

    def _initialize(self):
        if False:
            print('Hello World!')
        return gen_boosted_trees_ops.boosted_trees_create_ensemble(self.resource_handle, self._stamp_token, tree_ensemble_serialized=self._serialized_proto)

    @property
    def initializer(self):
        if False:
            i = 10
            return i + 15
        if self._init_op is None:
            self._init_op = self._initialize()
        return self._init_op

    def is_initialized(self):
        if False:
            print('Hello World!')
        return gen_boosted_trees_ops.is_boosted_trees_ensemble_initialized(self.resource_handle)

    def _serialize_to_tensors(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('When the need arises, TF2 compatibility can be added by implementing this method, along with _restore_from_tensors below.')

    def _restore_from_tensors(self, restored_tensors):
        if False:
            return 10
        raise NotImplementedError('When the need arises, TF2 compatibility can be added by implementing this method, along with _serialize_to_tensors above.')

    def get_stamp_token(self):
        if False:
            print('Hello World!')
        'Returns the current stamp token of the resource.'
        (stamp_token, _, _, _, _) = gen_boosted_trees_ops.boosted_trees_get_ensemble_states(self.resource_handle)
        return stamp_token

    def get_states(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns states of the tree ensemble.\n\n    Returns:\n      stamp_token, num_trees, num_finalized_trees, num_attempted_layers and\n      range of the nodes in the latest layer.\n    '
        (stamp_token, num_trees, num_finalized_trees, num_attempted_layers, nodes_range) = gen_boosted_trees_ops.boosted_trees_get_ensemble_states(self.resource_handle)
        return (array_ops.identity(stamp_token, name='stamp_token'), array_ops.identity(num_trees, name='num_trees'), array_ops.identity(num_finalized_trees, name='num_finalized_trees'), array_ops.identity(num_attempted_layers, name='num_attempted_layers'), array_ops.identity(nodes_range, name='last_layer_nodes_range'))

    def serialize(self):
        if False:
            print('Hello World!')
        'Serializes the ensemble into proto and returns the serialized proto.\n\n    Returns:\n      stamp_token: int64 scalar Tensor to denote the stamp of the resource.\n      serialized_proto: string scalar Tensor of the serialized proto.\n    '
        return gen_boosted_trees_ops.boosted_trees_serialize_ensemble(self.resource_handle)

    def deserialize(self, stamp_token, serialized_proto):
        if False:
            print('Hello World!')
        'Deserialize the input proto and resets the ensemble from it.\n\n    Args:\n      stamp_token: int64 scalar Tensor to denote the stamp of the resource.\n      serialized_proto: string scalar Tensor of the serialized proto.\n\n    Returns:\n      Operation (for dependencies).\n    '
        return gen_boosted_trees_ops.boosted_trees_deserialize_ensemble(self.resource_handle, stamp_token, serialized_proto)