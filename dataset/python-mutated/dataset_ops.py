"""Python wrappers for Datasets."""
import abc
import functools
import queue
import threading
from typing import Union
import warnings
import numpy as np
from tensorflow.core.framework import dataset_metadata_pb2
from tensorflow.core.framework import dataset_options_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_autograph
from tensorflow.python.data.ops import debug_mode
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.data.util import traverse
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import auto_control_deps_utils as acd_utils
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import none_tensor
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed as core_random_seed
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import base as tracking_base
from tensorflow.python.trackable import resource as resource_lib
from tensorflow.python.types import data as data_types
from tensorflow.python.types import trace
from tensorflow.python.util import deprecation
from tensorflow.python.util import lazy_loader
from tensorflow.python.util import nest as tf_nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
StructuredFunctionWrapper = structured_function.StructuredFunctionWrapper
prefetch_op = lazy_loader.LazyLoader('prefetch_op', globals(), 'tensorflow.python.data.ops.prefetch_op')
shuffle_op = lazy_loader.LazyLoader('shuffle_op', globals(), 'tensorflow.python.data.ops.shuffle_op')
ops.NotDifferentiable('ReduceDataset')
AUTOTUNE = -1
tf_export('data.AUTOTUNE').export_constant(__name__, 'AUTOTUNE')
tf_export('data.experimental.AUTOTUNE').export_constant(__name__, 'AUTOTUNE')
INFINITE = -1
UNKNOWN = -2
COMPRESSION_GZIP = 'GZIP'
COMPRESSION_SNAPPY = 'NONE'
DATASET_SPEC_FILENAME = 'dataset_spec.pb'
tf_export('data.INFINITE_CARDINALITY').export_constant(__name__, 'INFINITE')
tf_export('data.UNKNOWN_CARDINALITY').export_constant(__name__, 'UNKNOWN')

def _validate_and_encode(name):
    if False:
        return 10
    if not name.isidentifier():
        raise ValueError('Invalid `name`. The argument `name` needs to be a valid identifier. Value is considered a valid identifier if it only contains alphanumeric characters (a-z), (A-Z), and (0-9), or underscores (_). A valid identifier cannot start with a number, or contain any spaces.')
    return name.encode('utf-8')

def get_type(value):
    if False:
        print('Hello World!')
    'Returns the type of `value` if it is a TypeSpec.'
    if isinstance(value, type_spec.TypeSpec):
        return value.value_type()
    else:
        return type(value)

@tf_export('data.Dataset', v1=[])
class DatasetV2(collections_abc.Iterable, tracking_base.Trackable, composite_tensor.CompositeTensor, data_types.DatasetV2, metaclass=abc.ABCMeta):
    """Represents a potentially large set of elements.

  The `tf.data.Dataset` API supports writing descriptive and efficient input
  pipelines. `Dataset` usage follows a common pattern:

  1. Create a source dataset from your input data.
  2. Apply dataset transformations to preprocess the data.
  3. Iterate over the dataset and process the elements.

  Iteration happens in a streaming fashion, so the full dataset does not need to
  fit into memory.

  Source Datasets:

  The simplest way to create a dataset is to create it from a python `list`:

  >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
  >>> for element in dataset:
  ...   print(element)
  tf.Tensor(1, shape=(), dtype=int32)
  tf.Tensor(2, shape=(), dtype=int32)
  tf.Tensor(3, shape=(), dtype=int32)

  To process lines from files, use `tf.data.TextLineDataset`:

  >>> dataset = tf.data.TextLineDataset(["file1.txt", "file2.txt"])

  To process records written in the `TFRecord` format, use `TFRecordDataset`:

  >>> dataset = tf.data.TFRecordDataset(["file1.tfrecords", "file2.tfrecords"])

  To create a dataset of all files matching a pattern, use
  `tf.data.Dataset.list_files`:

  ```python
  dataset = tf.data.Dataset.list_files("/path/*.txt")
  ```

  See `tf.data.FixedLengthRecordDataset` and `tf.data.Dataset.from_generator`
  for more ways to create datasets.

  Transformations:

  Once you have a dataset, you can apply transformations to prepare the data for
  your model:

  >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
  >>> dataset = dataset.map(lambda x: x*2)
  >>> list(dataset.as_numpy_iterator())
  [2, 4, 6]

  Common Terms:

  **Element**: A single output from calling `next()` on a dataset iterator.
    Elements may be nested structures containing multiple components. For
    example, the element `(1, (3, "apple"))` has one tuple nested in another
    tuple. The components are `1`, `3`, and `"apple"`.

  **Component**: The leaf in the nested structure of an element.

  Supported types:

  Elements can be nested structures of tuples, named tuples, and dictionaries.
  Note that Python lists are *not* treated as nested structures of components.
  Instead, lists are converted to tensors and treated as components. For
  example, the element `(1, [1, 2, 3])` has only two components; the tensor `1`
  and the tensor `[1, 2, 3]`. Element components can be of any type
  representable by `tf.TypeSpec`, including `tf.Tensor`, `tf.data.Dataset`,
  `tf.sparse.SparseTensor`, `tf.RaggedTensor`, and `tf.TensorArray`.

  ```python
  a = 1 # Integer element
  b = 2.0 # Float element
  c = (1, 2) # Tuple element with 2 components
  d = {"a": (2, 2), "b": 3} # Dict element with 3 components
  Point = collections.namedtuple("Point", ["x", "y"])
  e = Point(1, 2) # Named tuple
  f = tf.data.Dataset.range(10) # Dataset element
  ```

  For more information,
  read [this guide](https://www.tensorflow.org/guide/data).
  """

    def __init__(self, variant_tensor):
        if False:
            while True:
                i = 10
        'Creates a DatasetV2 object.\n\n    This is a difference between DatasetV1 and DatasetV2. DatasetV1 does not\n    take anything in its constructor whereas in the DatasetV2, we expect\n    subclasses to create a variant_tensor and pass it in to the super() call.\n\n    Args:\n      variant_tensor: A DT_VARIANT tensor that represents the dataset.\n    '
        self._variant_tensor_attr = variant_tensor
        self._graph_attr = ops.get_default_graph()
        self._options_attr = options_lib.Options()
        for input_dataset in self._inputs():
            input_options = None
            if isinstance(input_dataset, data_types.DatasetV1):
                if hasattr(input_dataset, '_dataset'):
                    if not isinstance(input_dataset._dataset, data_types.DatasetV2):
                        raise TypeError(f'Each input of dataset {type(self)} should be a subclass of `tf.data.Dataset` but encountered {type(input_dataset._dataset)}.')
                    input_options = input_dataset._dataset._options_attr
            elif isinstance(input_dataset, data_types.DatasetV2):
                input_options = input_dataset._options_attr
            else:
                raise TypeError(f'Each input of dataset {type(self)} should be a subclass of `tf.data.Dataset` but encountered {type(input_dataset)}.')
            if input_options is not None:
                self._options_attr = self._options_attr.merge(input_options)
        self._options_attr._set_mutable(False)

    @property
    def _variant_tensor(self):
        if False:
            i = 10
            return i + 15
        return self._variant_tensor_attr

    @_variant_tensor.setter
    def _variant_tensor(self, _):
        if False:
            while True:
                i = 10
        raise ValueError('The `_variant_tensor` property cannot be modified.')

    @deprecation.deprecated_args(None, 'Use external_state_policy instead', 'allow_stateful')
    def _as_serialized_graph(self, allow_stateful=None, strip_device_assignment=None, external_state_policy=options_lib.ExternalStatePolicy.WARN):
        if False:
            while True:
                i = 10
        'Produces serialized graph representation of the dataset.\n\n    Args:\n      allow_stateful: If true, we allow stateful ops to be present in the graph\n        def. In that case, the state in these ops would be thrown away.\n      strip_device_assignment: If true, non-local (i.e. job and task) device\n        assignment is stripped from ops in the serialized graph.\n      external_state_policy: The ExternalStatePolicy enum that determines how we\n        handle input pipelines that depend on external state. By default, its\n        set to WARN.\n\n    Returns:\n      A scalar `tf.Tensor` of `tf.string` type, representing this dataset as a\n      serialized graph.\n    '
        if external_state_policy:
            policy = external_state_policy.value
            return gen_dataset_ops.dataset_to_graph_v2(self._variant_tensor, external_state_policy=policy, strip_device_assignment=strip_device_assignment)
        if strip_device_assignment:
            return gen_dataset_ops.dataset_to_graph(self._variant_tensor, allow_stateful=allow_stateful, strip_device_assignment=strip_device_assignment)
        return gen_dataset_ops.dataset_to_graph(self._variant_tensor, allow_stateful=allow_stateful)

    def _maybe_track_assets(self, graph_def):
        if False:
            return 10
        'Finds and tracks nodes in `graph_def` that refer to asset files.\n\n    Args:\n      graph_def: Serialized graph representation of this dataset.\n\n    Returns:\n      A dictionary mapping the node name of an asset constant to a tracked\n      `asset.Asset` object.\n    '
        asset_tracker = {}
        for node in graph_def.node:
            if node.name.startswith('FileIdentity'):
                asset_tracker[node.input[0]] = None
        if not asset_tracker:
            return {}
        for node in graph_def.node:
            if node.name in asset_tracker:
                tensor_proto = node.attr['value'].tensor
                with context.eager_mode(), ops.device('CPU'):
                    node_value = gen_parsing_ops.parse_tensor(tensor_proto.SerializeToString(), dtypes.string).numpy()
                asset_tracker[node.name] = [self._track_trackable(asset.Asset(n), name=node.name + '_' + str(i), overwrite=True) for (i, n) in enumerate(node_value)]
        return asset_tracker

    def _trackable_children(self, save_type=tracking_base.SaveType.CHECKPOINT, **kwargs):
        if False:
            i = 10
            return i + 15
        if save_type != tracking_base.SaveType.SAVEDMODEL:
            return {}

        @def_function.function(input_signature=[], autograph=False)
        def _creator():
            if False:
                return 10
            resource = self._trace_variant_creation()()
            return resource
        _creator.get_concrete_function()
        children = super(DatasetV2, self)._trackable_children(save_type, **kwargs)
        children['_variant_tracker'] = _VariantTracker(self._variant_tensor, _creator)
        return children

    def _trace_variant_creation(self):
        if False:
            for i in range(10):
                print('nop')
        'Traces a function which outputs a variant `tf.Tensor` for this dataset.\n\n    Note that creating this function involves evaluating an op, and is currently\n    only supported when executing eagerly.\n\n    Returns:\n      A zero-argument `ConcreteFunction` which outputs a variant `tf.Tensor`.\n    '
        variant = self._variant_tensor
        if not isinstance(variant, ops.EagerTensor):
            raise NotImplementedError('Constructing a tf.function that reproduces a given dataset is only supported for datasets created eagerly. Please file a feature request if this is important to you.')
        with context.eager_mode(), ops.device('CPU'):
            graph_def = graph_pb2.GraphDef().FromString(self._as_serialized_graph(external_state_policy=options_lib.ExternalStatePolicy.FAIL).numpy())
        output_node_names = []
        for node in graph_def.node:
            if node.op == '_Retval':
                output_node_names = node.input
        if len(output_node_names) != 1:
            raise AssertionError(f'Dataset graph is expected to only have one return value but found {len(output_node_names)} return values: {output_node_names}.')
        output_node_name = output_node_names[0]
        file_path_nodes = {}
        if ops.get_default_graph().building_function:
            asset_tracker = self._maybe_track_assets(graph_def)
            for key in asset_tracker:
                assets_list = [array_ops.expand_dims(asset.asset_path, axis=0) for asset in asset_tracker[key]]
                file_path_nodes[key] = array_ops.concat(assets_list, axis=0)
        variant_function = wrap_function.function_from_graph_def(graph_def, inputs=[], outputs=output_node_name + ':0', captures=file_path_nodes)
        for used_function in self._functions():
            used_function.function.add_to_graph(variant_function.graph)
        return variant_function

    @abc.abstractmethod
    def _inputs(self):
        if False:
            while True:
                i = 10
        'Returns a list of the input datasets of the dataset.'
        raise NotImplementedError(f'{type(self)}._inputs()')

    @property
    def _graph(self):
        if False:
            print('Hello World!')
        return self._graph_attr

    @_graph.setter
    def _graph(self, _):
        if False:
            i = 10
            return i + 15
        raise ValueError('The `_graph` property cannot be modified.')

    def _functions(self) -> list[StructuredFunctionWrapper]:
        if False:
            i = 10
            return i + 15
        'Returns a list of functions associated with this dataset.\n\n    Returns:\n      A list of `StructuredFunctionWrapper` objects.\n    '
        return []

    def _options(self):
        if False:
            return 10
        'Returns the options tensor for this dataset.'
        return gen_dataset_ops.get_options(self._variant_tensor)

    @classmethod
    def _options_tensor_to_options(cls, serialized_options):
        if False:
            for i in range(10):
                print('nop')
        'Converts options tensor to tf.data.Options object.'
        options = options_lib.Options()
        if tensor_util.constant_value(serialized_options) is not None:
            pb = dataset_options_pb2.Options.FromString(tensor_util.constant_value(serialized_options))
            options._from_proto(pb)
        return options

    def options(self):
        if False:
            i = 10
            return i + 15
        'Returns the options for this dataset and its inputs.\n\n    Returns:\n      A `tf.data.Options` object representing the dataset options.\n    '
        if context.executing_eagerly():
            options = self._options_tensor_to_options(self._options())
            options._set_mutable(False)
            return options
        warnings.warn('To make it possible to preserve tf.data options across serialization boundaries, their implementation has moved to be part of the TensorFlow graph. As a consequence, the options value is in general no longer known at graph construction time. Invoking this method in graph mode retains the legacy behavior of the original implementation, but note that the returned value might not reflect the actual value of the options.')
        return self._options_attr

    def _apply_debug_options(self):
        if False:
            for i in range(10):
                print('nop')
        if debug_mode.DEBUG_MODE:
            options = options_lib.Options()
            options.autotune.enabled = False
            options.experimental_optimization.filter_parallelization = False
            options.experimental_optimization.map_and_batch_fusion = False
            options.experimental_optimization.map_parallelization = False
            dataset = _OptionsDataset(self, options)
        else:
            dataset = self
        return dataset

    def __iter__(self) -> iterator_ops.OwnedIterator:
        if False:
            for i in range(10):
                print('nop')
        'Creates an iterator for elements of this dataset.\n\n    The returned iterator implements the Python Iterator protocol.\n\n    Returns:\n      An `tf.data.Iterator` for the elements of this dataset.\n\n    Raises:\n      RuntimeError: If not inside of tf.function and not executing eagerly.\n    '
        if context.executing_eagerly() or ops.inside_function():
            with ops.colocate_with(self._variant_tensor):
                return iterator_ops.OwnedIterator(self)
        else:
            raise RuntimeError('`tf.data.Dataset` only supports Python-style iteration in eager mode or within tf.function.')

    def __bool__(self):
        if False:
            return 10
        return True
    __nonzero__ = __bool__

    def __len__(self):
        if False:
            i = 10
            return i + 15
        'Returns the length of the dataset if it is known and finite.\n\n    This method requires that you are running in eager mode, and that the\n    length of the dataset is known and non-infinite. When the length may be\n    unknown or infinite, or if you are running in graph mode, use\n    `tf.data.Dataset.cardinality` instead.\n\n    Returns:\n      An integer representing the length of the dataset.\n\n    Raises:\n      RuntimeError: If the dataset length is unknown or infinite, or if eager\n        execution is not enabled.\n    '
        if not context.executing_eagerly():
            raise TypeError('`tf.data.Dataset` only supports `len` in eager mode. Use `tf.data.Dataset.cardinality()` instead.')
        length = self.cardinality()
        if length.numpy() == INFINITE:
            raise TypeError('The dataset is infinite.')
        if length.numpy() == UNKNOWN:
            raise TypeError('The dataset length is unknown.')
        return length

    @abc.abstractproperty
    def element_spec(self):
        if False:
            i = 10
            return i + 15
        'The type specification of an element of this dataset.\n\n    >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])\n    >>> dataset.element_spec\n    TensorSpec(shape=(), dtype=tf.int32, name=None)\n\n    For more information,\n    read [this guide](https://www.tensorflow.org/guide/data#dataset_structure).\n\n    Returns:\n      A (nested) structure of `tf.TypeSpec` objects matching the structure of an\n      element of this dataset and specifying the type of individual components.\n    '
        raise NotImplementedError(f'{type(self)}.element_spec()')

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        type_ = type(self._dataset if isinstance(self, DatasetV1Adapter) else self)
        return f'<{type_.__name__} element_spec={self.element_spec}>'

    def __debug_string__(self):
        if False:
            return 10
        'Returns a string showing the type of the dataset and its inputs.\n\n    This string is intended only for debugging purposes, and may change without\n    warning.\n    '
        lines = []
        to_process = [(self, 0)]
        while to_process:
            (dataset, depth) = to_process.pop()
            lines.append('-' * 2 * depth + repr(dataset))
            to_process.extend([(ds, depth + 1) for ds in dataset._inputs()])
        return '\n'.join(lines)

    def as_numpy_iterator(self):
        if False:
            i = 10
            return i + 15
        "Returns an iterator which converts all elements of the dataset to numpy.\n\n    Use `as_numpy_iterator` to inspect the content of your dataset. To see\n    element shapes and types, print dataset elements directly instead of using\n    `as_numpy_iterator`.\n\n    >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])\n    >>> for element in dataset:\n    ...   print(element)\n    tf.Tensor(1, shape=(), dtype=int32)\n    tf.Tensor(2, shape=(), dtype=int32)\n    tf.Tensor(3, shape=(), dtype=int32)\n\n    This method requires that you are running in eager mode and the dataset's\n    element_spec contains only `TensorSpec` components.\n\n    >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])\n    >>> for element in dataset.as_numpy_iterator():\n    ...   print(element)\n    1\n    2\n    3\n\n    >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])\n    >>> print(list(dataset.as_numpy_iterator()))\n    [1, 2, 3]\n\n    `as_numpy_iterator()` will preserve the nested structure of dataset\n    elements.\n\n    >>> dataset = tf.data.Dataset.from_tensor_slices({'a': ([1, 2], [3, 4]),\n    ...                                               'b': [5, 6]})\n    >>> list(dataset.as_numpy_iterator()) == [{'a': (1, 3), 'b': 5},\n    ...                                       {'a': (2, 4), 'b': 6}]\n    True\n\n    Returns:\n      An iterable over the elements of the dataset, with their tensors converted\n      to numpy arrays.\n\n    Raises:\n      TypeError: if an element contains a non-`Tensor` value.\n      RuntimeError: if eager execution is not enabled.\n    "
        if not context.executing_eagerly():
            raise RuntimeError('`tf.data.Dataset.as_numpy_iterator()` is only supported in eager mode.')
        for component_spec in nest.flatten(self.element_spec):
            if not isinstance(component_spec, (tensor_spec.TensorSpec, ragged_tensor.RaggedTensorSpec, sparse_tensor_lib.SparseTensorSpec, none_tensor.NoneTensorSpec)):
                raise TypeError(f'`tf.data.Dataset.as_numpy_iterator()` is not supported for datasets that produce values of type {component_spec.value_type}')
        return NumpyIterator(self)

    @property
    def _flat_shapes(self):
        if False:
            i = 10
            return i + 15
        'Returns a list `tf.TensorShapes`s for the element tensor representation.\n\n    Returns:\n      A list `tf.TensorShapes`s for the element tensor representation.\n    '
        return structure.get_flat_tensor_shapes(self.element_spec)

    @property
    def _flat_types(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns a list `tf.DType`s for the element tensor representation.\n\n    Returns:\n      A list `tf.DType`s for the element tensor representation.\n    '
        return structure.get_flat_tensor_types(self.element_spec)

    @property
    def _flat_structure(self):
        if False:
            return 10
        'Helper for setting `output_shapes` and `output_types` attrs of an op.\n\n    Most dataset op constructors expect `output_shapes` and `output_types`\n    arguments that represent the flattened structure of an element. This helper\n    function generates these attrs as a keyword argument dictionary, allowing\n    `Dataset._variant_tensor` implementations to pass `**self._flat_structure`\n    to the op constructor.\n\n    Returns:\n      A dictionary of keyword arguments that can be passed to a dataset op\n      constructor.\n    '
        return {'output_shapes': self._flat_shapes, 'output_types': self._flat_types}

    @property
    def _metadata(self):
        if False:
            return 10
        'Helper for generating dataset metadata.'
        metadata = dataset_metadata_pb2.Metadata()
        if self._name:
            metadata.name = _validate_and_encode(self._name)
        return metadata

    @property
    def _common_args(self):
        if False:
            for i in range(10):
                print('nop')
        'Helper for generating arguments that are common across most dataset ops.\n\n    Most dataset op constructors expect `output_shapes` and `output_types`\n    arguments that represent the flattened structure of an element, as well as a\n    `metadata` argument for additional metadata such as user-defined dataset\n    name. This helper function generates common attributes as a keyword argument\n    dictionary, allowing `Dataset._variant_tensor` implementations to pass\n    `**self._common_args` to the op constructor.\n\n    Returns:\n      A dictionary of keyword arguments that can be passed to a dataset op\n      constructor.\n    '
        return {'metadata': self._metadata.SerializeToString(), 'output_shapes': self._flat_shapes, 'output_types': self._flat_types}

    @property
    def _type_spec(self):
        if False:
            print('Hello World!')
        return DatasetSpec(self.element_spec)

    @staticmethod
    def from_tensors(tensors, name=None) -> 'DatasetV2':
        if False:
            print('Hello World!')
        'Creates a `Dataset` with a single element, comprising the given tensors.\n\n    `from_tensors` produces a dataset containing only a single element. To slice\n    the input tensor into multiple elements, use `from_tensor_slices` instead.\n\n    >>> dataset = tf.data.Dataset.from_tensors([1, 2, 3])\n    >>> list(dataset.as_numpy_iterator())\n    [array([1, 2, 3], dtype=int32)]\n    >>> dataset = tf.data.Dataset.from_tensors(([1, 2, 3], \'A\'))\n    >>> list(dataset.as_numpy_iterator())\n    [(array([1, 2, 3], dtype=int32), b\'A\')]\n\n    >>> # You can use `from_tensors` to produce a dataset which repeats\n    >>> # the same example many times.\n    >>> example = tf.constant([1,2,3])\n    >>> dataset = tf.data.Dataset.from_tensors(example).repeat(2)\n    >>> list(dataset.as_numpy_iterator())\n    [array([1, 2, 3], dtype=int32), array([1, 2, 3], dtype=int32)]\n\n    Note that if `tensors` contains a NumPy array, and eager execution is not\n    enabled, the values will be embedded in the graph as one or more\n    `tf.constant` operations. For large datasets (> 1 GB), this can waste\n    memory and run into byte limits of graph serialization. If `tensors`\n    contains one or more large NumPy arrays, consider the alternative described\n    in [this\n    guide](https://tensorflow.org/guide/data#consuming_numpy_arrays).\n\n    Args:\n      tensors: A dataset "element". Supported values are documented\n        [here](https://www.tensorflow.org/guide/data#dataset_structure).\n      name: (Optional.) A name for the tf.data operation.\n\n    Returns:\n      Dataset: A `Dataset`.\n    '
        from tensorflow.python.data.ops import from_tensors_op
        return from_tensors_op._from_tensors(tensors, name)

    @staticmethod
    def from_tensor_slices(tensors, name=None) -> 'DatasetV2':
        if False:
            for i in range(10):
                print('nop')
        'Creates a `Dataset` whose elements are slices of the given tensors.\n\n    The given tensors are sliced along their first dimension. This operation\n    preserves the structure of the input tensors, removing the first dimension\n    of each tensor and using it as the dataset dimension. All input tensors\n    must have the same size in their first dimensions.\n\n    >>> # Slicing a 1D tensor produces scalar tensor elements.\n    >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])\n    >>> list(dataset.as_numpy_iterator())\n    [1, 2, 3]\n\n    >>> # Slicing a 2D tensor produces 1D tensor elements.\n    >>> dataset = tf.data.Dataset.from_tensor_slices([[1, 2], [3, 4]])\n    >>> list(dataset.as_numpy_iterator())\n    [array([1, 2], dtype=int32), array([3, 4], dtype=int32)]\n\n    >>> # Slicing a tuple of 1D tensors produces tuple elements containing\n    >>> # scalar tensors.\n    >>> dataset = tf.data.Dataset.from_tensor_slices(([1, 2], [3, 4], [5, 6]))\n    >>> list(dataset.as_numpy_iterator())\n    [(1, 3, 5), (2, 4, 6)]\n\n    >>> # Dictionary structure is also preserved.\n    >>> dataset = tf.data.Dataset.from_tensor_slices({"a": [1, 2], "b": [3, 4]})\n    >>> list(dataset.as_numpy_iterator()) == [{\'a\': 1, \'b\': 3},\n    ...                                       {\'a\': 2, \'b\': 4}]\n    True\n\n    >>> # Two tensors can be combined into one Dataset object.\n    >>> features = tf.constant([[1, 3], [2, 1], [3, 3]]) # ==> 3x2 tensor\n    >>> labels = tf.constant([\'A\', \'B\', \'A\']) # ==> 3x1 tensor\n    >>> dataset = Dataset.from_tensor_slices((features, labels))\n    >>> # Both the features and the labels tensors can be converted\n    >>> # to a Dataset object separately and combined after.\n    >>> features_dataset = Dataset.from_tensor_slices(features)\n    >>> labels_dataset = Dataset.from_tensor_slices(labels)\n    >>> dataset = Dataset.zip((features_dataset, labels_dataset))\n    >>> # A batched feature and label set can be converted to a Dataset\n    >>> # in similar fashion.\n    >>> batched_features = tf.constant([[[1, 3], [2, 3]],\n    ...                                 [[2, 1], [1, 2]],\n    ...                                 [[3, 3], [3, 2]]], shape=(3, 2, 2))\n    >>> batched_labels = tf.constant([[\'A\', \'A\'],\n    ...                               [\'B\', \'B\'],\n    ...                               [\'A\', \'B\']], shape=(3, 2, 1))\n    >>> dataset = Dataset.from_tensor_slices((batched_features, batched_labels))\n    >>> for element in dataset.as_numpy_iterator():\n    ...   print(element)\n    (array([[1, 3],\n           [2, 3]], dtype=int32), array([[b\'A\'],\n           [b\'A\']], dtype=object))\n    (array([[2, 1],\n           [1, 2]], dtype=int32), array([[b\'B\'],\n           [b\'B\']], dtype=object))\n    (array([[3, 3],\n           [3, 2]], dtype=int32), array([[b\'A\'],\n           [b\'B\']], dtype=object))\n\n    Note that if `tensors` contains a NumPy array, and eager execution is not\n    enabled, the values will be embedded in the graph as one or more\n    `tf.constant` operations. For large datasets (> 1 GB), this can waste\n    memory and run into byte limits of graph serialization. If `tensors`\n    contains one or more large NumPy arrays, consider the alternative described\n    in [this guide](\n    https://tensorflow.org/guide/data#consuming_numpy_arrays).\n\n    Args:\n      tensors: A dataset element, whose components have the same first\n        dimension. Supported values are documented\n        [here](https://www.tensorflow.org/guide/data#dataset_structure).\n      name: (Optional.) A name for the tf.data operation.\n\n    Returns:\n      Dataset: A `Dataset`.\n    '
        from tensorflow.python.data.ops import from_tensor_slices_op
        return from_tensor_slices_op._from_tensor_slices(tensors, name)

    class _GeneratorState:
        """Stores outstanding iterators created from a Python generator.

    This class keeps track of potentially multiple iterators that may have
    been created from a generator, e.g. in the case that the dataset is
    repeated, or nested within a parallel computation.
    """

        def __init__(self, generator):
            if False:
                for i in range(10):
                    print('nop')
            self._generator = generator
            self._lock = threading.Lock()
            self._next_id = 0
            self._args = {}
            self._iterators = {}

        def _normalize_id(self, iterator_id):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(iterator_id, np.ndarray):
                return iterator_id.item()
            return iterator_id

        def get_next_id(self, *args):
            if False:
                while True:
                    i = 10
            with self._lock:
                ret = self._next_id
                self._next_id += 1
            self._args[ret] = args
            return np.array(ret, dtype=np.int64)

        def get_iterator(self, iterator_id):
            if False:
                return 10
            iterator_id = self._normalize_id(iterator_id)
            try:
                return self._iterators[iterator_id]
            except KeyError:
                iterator = iter(self._generator(*self._args.pop(iterator_id)))
                self._iterators[iterator_id] = iterator
                return iterator

        def iterator_completed(self, iterator_id):
            if False:
                i = 10
                return i + 15
            del self._iterators[self._normalize_id(iterator_id)]

    @staticmethod
    @deprecation.deprecated_args(None, 'Use output_signature instead', 'output_types', 'output_shapes')
    def from_generator(generator, output_types=None, output_shapes=None, args=None, output_signature=None, name=None) -> 'DatasetV2':
        if False:
            i = 10
            return i + 15
        'Creates a `Dataset` whose elements are generated by `generator`.\n\n    Note: The current implementation of `Dataset.from_generator()` uses\n    `tf.numpy_function` and inherits the same constraints. In particular, it\n    requires the dataset and iterator related operations to be placed\n    on a device in the same process as the Python program that called\n    `Dataset.from_generator()`. In particular, using `from_generator` will\n    preclude the use of tf.data service for scaling out dataset processing.\n    The body of `generator` will not be serialized in a `GraphDef`, and you\n    should not use this method if you need to serialize your model and restore\n    it in a different environment.\n\n    The `generator` argument must be a callable object that returns\n    an object that supports the `iter()` protocol (e.g. a generator function).\n\n    The elements generated by `generator` must be compatible with either the\n    given `output_signature` argument or with the given `output_types` and\n    (optionally) `output_shapes` arguments, whichever was specified.\n\n    The recommended way to call `from_generator` is to use the\n    `output_signature` argument. In this case the output will be assumed to\n    consist of objects with the classes, shapes and types defined by\n    `tf.TypeSpec` objects from `output_signature` argument:\n\n    >>> def gen():\n    ...   ragged_tensor = tf.ragged.constant([[1, 2], [3]])\n    ...   yield 42, ragged_tensor\n    >>>\n    >>> dataset = tf.data.Dataset.from_generator(\n    ...      gen,\n    ...      output_signature=(\n    ...          tf.TensorSpec(shape=(), dtype=tf.int32),\n    ...          tf.RaggedTensorSpec(shape=(2, None), dtype=tf.int32)))\n    >>>\n    >>> list(dataset.take(1))\n    [(<tf.Tensor: shape=(), dtype=int32, numpy=42>,\n    <tf.RaggedTensor [[1, 2], [3]]>)]\n\n    There is also a deprecated way to call `from_generator` by either with\n    `output_types` argument alone or together with `output_shapes` argument.\n    In this case the output of the function will be assumed to consist of\n    `tf.Tensor` objects with the types defined by `output_types` and with the\n    shapes which are either unknown or defined by `output_shapes`.\n\n    Note: If `generator` depends on mutable global variables or other external\n    state, be aware that the runtime may invoke `generator` multiple times\n    (in order to support repeating the `Dataset`) and at any time\n    between the call to `Dataset.from_generator()` and the production of the\n    first element from the generator. Mutating global variables or external\n    state can cause undefined behavior, and we recommend that you explicitly\n    cache any external state in `generator` before calling\n    `Dataset.from_generator()`.\n\n    Note: While the `output_signature` parameter makes it possible to yield\n    `Dataset` elements, the scope of `Dataset.from_generator()` should be\n    limited to logic that cannot be expressed through tf.data operations. Using\n    tf.data operations within the generator function is an anti-pattern and may\n    result in incremental memory growth.\n\n    Args:\n      generator: A callable object that returns an object that supports the\n        `iter()` protocol. If `args` is not specified, `generator` must take no\n        arguments; otherwise it must take as many arguments as there are values\n        in `args`.\n      output_types: (Optional.) A (nested) structure of `tf.DType` objects\n        corresponding to each component of an element yielded by `generator`.\n      output_shapes: (Optional.) A (nested) structure of `tf.TensorShape`\n        objects corresponding to each component of an element yielded by\n        `generator`.\n      args: (Optional.) A tuple of `tf.Tensor` objects that will be evaluated\n        and passed to `generator` as NumPy-array arguments.\n      output_signature: (Optional.) A (nested) structure of `tf.TypeSpec`\n        objects corresponding to each component of an element yielded by\n        `generator`.\n      name: (Optional.) A name for the tf.data operations used by\n        `from_generator`.\n\n    Returns:\n      Dataset: A `Dataset`.\n    '
        from tensorflow.python.data.ops import from_generator_op
        return from_generator_op._from_generator(generator, output_types, output_shapes, args, output_signature, name)

    @staticmethod
    def range(*args, **kwargs) -> 'DatasetV2':
        if False:
            i = 10
            return i + 15
        "Creates a `Dataset` of a step-separated range of values.\n\n    >>> list(Dataset.range(5).as_numpy_iterator())\n    [0, 1, 2, 3, 4]\n    >>> list(Dataset.range(2, 5).as_numpy_iterator())\n    [2, 3, 4]\n    >>> list(Dataset.range(1, 5, 2).as_numpy_iterator())\n    [1, 3]\n    >>> list(Dataset.range(1, 5, -2).as_numpy_iterator())\n    []\n    >>> list(Dataset.range(5, 1).as_numpy_iterator())\n    []\n    >>> list(Dataset.range(5, 1, -2).as_numpy_iterator())\n    [5, 3]\n    >>> list(Dataset.range(2, 5, output_type=tf.int32).as_numpy_iterator())\n    [2, 3, 4]\n    >>> list(Dataset.range(1, 5, 2, output_type=tf.float32).as_numpy_iterator())\n    [1.0, 3.0]\n\n    Args:\n      *args: follows the same semantics as python's range.\n        len(args) == 1 -> start = 0, stop = args[0], step = 1.\n        len(args) == 2 -> start = args[0], stop = args[1], step = 1.\n        len(args) == 3 -> start = args[0], stop = args[1], step = args[2].\n      **kwargs:\n        - output_type: Its expected dtype. (Optional, default: `tf.int64`).\n        - name: (Optional.) A name for the tf.data operation.\n\n    Returns:\n      Dataset: A `RangeDataset`.\n\n    Raises:\n      ValueError: if len(args) == 0.\n    "
        from tensorflow.python.data.ops import range_op
        return range_op._range(*args, **kwargs)

    @staticmethod
    def zip(*args, datasets=None, name=None) -> 'DatasetV2':
        if False:
            return 10
        "Creates a `Dataset` by zipping together the given datasets.\n\n    This method has similar semantics to the built-in `zip()` function\n    in Python, with the main difference being that the `datasets`\n    argument can be a (nested) structure of `Dataset` objects. The supported\n    nesting mechanisms are documented\n    [here] (https://www.tensorflow.org/guide/data#dataset_structure).\n\n    >>> # The datasets or nested structure of datasets `*args` argument\n    >>> # determines the structure of elements in the resulting dataset.\n    >>> a = tf.data.Dataset.range(1, 4)  # ==> [ 1, 2, 3 ]\n    >>> b = tf.data.Dataset.range(4, 7)  # ==> [ 4, 5, 6 ]\n    >>> ds = tf.data.Dataset.zip(a, b)\n    >>> list(ds.as_numpy_iterator())\n    [(1, 4), (2, 5), (3, 6)]\n    >>> ds = tf.data.Dataset.zip(b, a)\n    >>> list(ds.as_numpy_iterator())\n    [(4, 1), (5, 2), (6, 3)]\n    >>>\n    >>> # The `datasets` argument may contain an arbitrary number of datasets.\n    >>> c = tf.data.Dataset.range(7, 13).batch(2)  # ==> [ [7, 8],\n    ...                                            #       [9, 10],\n    ...                                            #       [11, 12] ]\n    >>> ds = tf.data.Dataset.zip(a, b, c)\n    >>> for element in ds.as_numpy_iterator():\n    ...   print(element)\n    (1, 4, array([7, 8]))\n    (2, 5, array([ 9, 10]))\n    (3, 6, array([11, 12]))\n    >>>\n    >>> # The number of elements in the resulting dataset is the same as\n    >>> # the size of the smallest dataset in `datasets`.\n    >>> d = tf.data.Dataset.range(13, 15)  # ==> [ 13, 14 ]\n    >>> ds = tf.data.Dataset.zip(a, d)\n    >>> list(ds.as_numpy_iterator())\n    [(1, 13), (2, 14)]\n\n    Args:\n      *args: Datasets or nested structures of datasets to zip together. This\n        can't be set if `datasets` is set.\n      datasets: A (nested) structure of datasets. This can't be set if `*args`\n        is set. Note that this exists only for backwards compatibility and it is\n        preferred to use *args.\n      name: (Optional.) A name for the tf.data operation.\n\n    Returns:\n      A new `Dataset` with the transformation applied as described above.\n    "
        from tensorflow.python.data.ops import zip_op
        if not args and datasets is None:
            raise TypeError('Must pass at least one dataset to `zip`.')
        if args and datasets is not None:
            raise TypeError('Both `*args` and `datasets` cannot be set.')
        if len(args) == 1:
            datasets = args[0]
        elif len(args) > 1:
            datasets = args
        return zip_op._zip(datasets, name)

    def concatenate(self, dataset, name=None) -> 'DatasetV2':
        if False:
            while True:
                i = 10
        'Creates a `Dataset` by concatenating the given dataset with this dataset.\n\n    >>> a = tf.data.Dataset.range(1, 4)  # ==> [ 1, 2, 3 ]\n    >>> b = tf.data.Dataset.range(4, 8)  # ==> [ 4, 5, 6, 7 ]\n    >>> ds = a.concatenate(b)\n    >>> list(ds.as_numpy_iterator())\n    [1, 2, 3, 4, 5, 6, 7]\n    >>> # The input dataset and dataset to be concatenated should have\n    >>> # compatible element specs.\n    >>> c = tf.data.Dataset.zip((a, b))\n    >>> a.concatenate(c)\n    Traceback (most recent call last):\n    TypeError: Two datasets to concatenate have different types\n    <dtype: \'int64\'> and (tf.int64, tf.int64)\n    >>> d = tf.data.Dataset.from_tensor_slices(["a", "b", "c"])\n    >>> a.concatenate(d)\n    Traceback (most recent call last):\n    TypeError: Two datasets to concatenate have different types\n    <dtype: \'int64\'> and <dtype: \'string\'>\n\n    Args:\n      dataset: `Dataset` to be concatenated.\n      name: (Optional.) A name for the tf.data operation.\n\n    Returns:\n      A new `Dataset` with the transformation applied as described above.\n    '
        from tensorflow.python.data.ops import concatenate_op
        return concatenate_op._concatenate(self, dataset, name)

    @staticmethod
    def counter(start=0, step=1, dtype=dtypes.int64, name=None) -> 'DatasetV2':
        if False:
            i = 10
            return i + 15
        'Creates a `Dataset` that counts from `start` in steps of size `step`.\n\n    Unlike `tf.data.Dataset.range`, which stops at some ending number,\n    `tf.data.Dataset.counter` produces elements indefinitely.\n\n    >>> dataset = tf.data.experimental.Counter().take(5)\n    >>> list(dataset.as_numpy_iterator())\n    [0, 1, 2, 3, 4]\n    >>> dataset.element_spec\n    TensorSpec(shape=(), dtype=tf.int64, name=None)\n    >>> dataset = tf.data.experimental.Counter(dtype=tf.int32)\n    >>> dataset.element_spec\n    TensorSpec(shape=(), dtype=tf.int32, name=None)\n    >>> dataset = tf.data.experimental.Counter(start=2).take(5)\n    >>> list(dataset.as_numpy_iterator())\n    [2, 3, 4, 5, 6]\n    >>> dataset = tf.data.experimental.Counter(start=2, step=5).take(5)\n    >>> list(dataset.as_numpy_iterator())\n    [2, 7, 12, 17, 22]\n    >>> dataset = tf.data.experimental.Counter(start=10, step=-1).take(5)\n    >>> list(dataset.as_numpy_iterator())\n    [10, 9, 8, 7, 6]\n\n    Args:\n      start: (Optional.) The starting value for the counter. Defaults to 0.\n      step: (Optional.) The step size for the counter. Defaults to 1.\n      dtype: (Optional.) The data type for counter elements. Defaults to\n        `tf.int64`.\n      name: (Optional.) A name for the tf.data operation.\n\n    Returns:\n      A `Dataset` of scalar `dtype` elements.\n    '
        from tensorflow.python.data.ops import counter_op
        return counter_op._counter(start, step, dtype, name=name)

    def rebatch(self, batch_size, drop_remainder=False, name=None) -> 'DatasetV2':
        if False:
            print('Hello World!')
        'Creates a `Dataset` that rebatches the elements from this dataset.\n\n    `rebatch(N)` is functionally equivalent to `unbatch().batch(N)`, but is\n    more efficient, performing one copy instead of two.\n\n    >>> ds = tf.data.Dataset.range(6)\n    >>> ds = ds.batch(2)\n    >>> ds = ds.rebatch(3)\n    >>> list(ds.as_numpy_iterator())\n    [array([0, 1, 2]), array([3, 4, 5])]\n\n    >>> ds = tf.data.Dataset.range(7)\n    >>> ds = ds.batch(4)\n    >>> ds = ds.rebatch(3)\n    >>> list(ds.as_numpy_iterator())\n    [array([0, 1, 2]), array([3, 4, 5]), array([6])]\n\n    >>> ds = tf.data.Dataset.range(7)\n    >>> ds = ds.batch(2)\n    >>> ds = ds.rebatch(3, drop_remainder=True)\n    >>> list(ds.as_numpy_iterator())\n    [array([0, 1, 2]), array([3, 4, 5])]\n\n    If the `batch_size` argument is a list, `rebatch` cycles through the list\n    to determine the size of each batch.\n\n    >>> ds = tf.data.Dataset.range(8)\n    >>> ds = ds.batch(4)\n    >>> ds = ds.rebatch([2, 1, 1])\n    >>> list(ds.as_numpy_iterator())\n    [array([0, 1]), array([2]), array([3]), array([4, 5]), array([6]),\n    array([7])]\n\n    Args:\n      batch_size: A `tf.int64` scalar or vector, representing the size of\n        batches to produce. If this argument is a vector, these values are\n        cycled through in round robin fashion.\n      drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing\n        whether the last batch should be dropped in the case it has fewer than\n        `batch_size[cycle_index]` elements; the default behavior is not to drop\n        the smaller batch.\n      name: (Optional.) A name for the tf.data operation.\n\n    Returns:\n      A `Dataset` of scalar `dtype` elements.\n    '
        from tensorflow.python.data.ops import rebatch_op
        return rebatch_op._rebatch(self, batch_size, drop_remainder, name=name)

    def prefetch(self, buffer_size, name=None) -> 'DatasetV2':
        if False:
            return 10
        'Creates a `Dataset` that prefetches elements from this dataset.\n\n    Most dataset input pipelines should end with a call to `prefetch`. This\n    allows later elements to be prepared while the current element is being\n    processed. This often improves latency and throughput, at the cost of\n    using additional memory to store prefetched elements.\n\n    Note: Like other `Dataset` methods, prefetch operates on the\n    elements of the input dataset. It has no concept of examples vs. batches.\n    `examples.prefetch(2)` will prefetch two elements (2 examples),\n    while `examples.batch(20).prefetch(2)` will prefetch 2 elements\n    (2 batches, of 20 examples each).\n\n    >>> dataset = tf.data.Dataset.range(3)\n    >>> dataset = dataset.prefetch(2)\n    >>> list(dataset.as_numpy_iterator())\n    [0, 1, 2]\n\n    Args:\n      buffer_size: A `tf.int64` scalar `tf.Tensor`, representing the maximum\n        number of elements that will be buffered when prefetching. If the value\n        `tf.data.AUTOTUNE` is used, then the buffer size is dynamically tuned.\n      name: Optional. A name for the tf.data transformation.\n\n    Returns:\n      A new `Dataset` with the transformation applied as described above.\n    '
        return prefetch_op._prefetch(self, buffer_size, name=name)

    @staticmethod
    def list_files(file_pattern, shuffle=None, seed=None, name=None) -> 'DatasetV2':
        if False:
            while True:
                i = 10
        'A dataset of all files matching one or more glob patterns.\n\n    The `file_pattern` argument should be a small number of glob patterns.\n    If your filenames have already been globbed, use\n    `Dataset.from_tensor_slices(filenames)` instead, as re-globbing every\n    filename with `list_files` may result in poor performance with remote\n    storage systems.\n\n    Note: The default behavior of this method is to return filenames in\n    a non-deterministic random shuffled order. Pass a `seed` or `shuffle=False`\n    to get results in a deterministic order.\n\n    Example:\n      If we had the following files on our filesystem:\n\n        - /path/to/dir/a.txt\n        - /path/to/dir/b.py\n        - /path/to/dir/c.py\n\n      If we pass "/path/to/dir/*.py" as the directory, the dataset\n      would produce:\n\n        - /path/to/dir/b.py\n        - /path/to/dir/c.py\n\n    Args:\n      file_pattern: A string, a list of strings, or a `tf.Tensor` of string type\n        (scalar or vector), representing the filename glob (i.e. shell wildcard)\n        pattern(s) that will be matched.\n      shuffle: (Optional.) If `True`, the file names will be shuffled randomly.\n        Defaults to `True`.\n      seed: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the random\n        seed that will be used to create the distribution. See\n        `tf.random.set_seed` for behavior.\n      name: Optional. A name for the tf.data operations used by `list_files`.\n\n    Returns:\n     Dataset: A `Dataset` of strings corresponding to file names.\n    '
        with ops.name_scope('list_files'):
            if shuffle is None:
                shuffle = True
            file_pattern = ops.convert_to_tensor(file_pattern, dtype=dtypes.string, name='file_pattern')
            matching_files = gen_io_ops.matching_files(file_pattern)
            condition = math_ops.greater(array_ops.shape(matching_files)[0], 0, name='match_not_empty')
            message = math_ops.add('No files matched pattern: ', string_ops.reduce_join(file_pattern, separator=', '), name='message')
            assert_not_empty = control_flow_assert.Assert(condition, [message], summarize=1, name='assert_not_empty')
            with ops.control_dependencies([assert_not_empty]):
                matching_files = array_ops.identity(matching_files)
            from tensorflow.python.data.ops import from_tensor_slices_op
            dataset = from_tensor_slices_op._TensorSliceDataset(matching_files, is_files=True, name=name)
            if issubclass(Dataset, DatasetV1):
                dataset = DatasetV1Adapter(dataset)
            if shuffle:
                buffer_size = math_ops.maximum(array_ops.shape(matching_files, out_type=dtypes.int64)[0], 1)
                dataset = dataset.shuffle(buffer_size, seed=seed, name=name)
            return dataset

    def repeat(self, count=None, name=None) -> 'DatasetV2':
        if False:
            return 10
        'Repeats this dataset so each original value is seen `count` times.\n\n    >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])\n    >>> dataset = dataset.repeat(3)\n    >>> list(dataset.as_numpy_iterator())\n    [1, 2, 3, 1, 2, 3, 1, 2, 3]\n\n    Note: If the input dataset depends on global state (e.g. a random number\n    generator) or its output is non-deterministic (e.g. because of upstream\n    `shuffle`), then different repetitions may produce different elements.\n\n    Args:\n      count: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the\n        number of times the dataset should be repeated. The default behavior (if\n        `count` is `None` or `-1`) is for the dataset be repeated indefinitely.\n      name: (Optional.) A name for the tf.data operation.\n\n    Returns:\n      A new `Dataset` with the transformation applied as described above.\n    '
        from tensorflow.python.data.ops import repeat_op
        return repeat_op._repeat(self, count, name)

    def enumerate(self, start=0, name=None) -> 'DatasetV2':
        if False:
            for i in range(10):
                print('nop')
        "Enumerates the elements of this dataset.\n\n    It is similar to python's `enumerate`.\n\n    >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])\n    >>> dataset = dataset.enumerate(start=5)\n    >>> for element in dataset.as_numpy_iterator():\n    ...   print(element)\n    (5, 1)\n    (6, 2)\n    (7, 3)\n\n    >>> # The (nested) structure of the input dataset determines the\n    >>> # structure of elements in the resulting dataset.\n    >>> dataset = tf.data.Dataset.from_tensor_slices([(7, 8), (9, 10)])\n    >>> dataset = dataset.enumerate()\n    >>> for element in dataset.as_numpy_iterator():\n    ...   print(element)\n    (0, array([7, 8], dtype=int32))\n    (1, array([ 9, 10], dtype=int32))\n\n    Args:\n      start: A `tf.int64` scalar `tf.Tensor`, representing the start value for\n        enumeration.\n      name: Optional. A name for the tf.data operations used by `enumerate`.\n\n    Returns:\n      A new `Dataset` with the transformation applied as described above.\n    "
        max_value = np.iinfo(dtypes.int64.as_numpy_dtype).max
        range_dataset = Dataset.range(start, max_value, name=name)
        range_dataset = _apply_rewrite(range_dataset, 'replicate_on_split')
        return Dataset.zip((range_dataset, self), name=name)

    def shuffle(self, buffer_size, seed=None, reshuffle_each_iteration=None, name=None) -> 'DatasetV2':
        if False:
            while True:
                i = 10
        'Randomly shuffles the elements of this dataset.\n\n    This dataset fills a buffer with `buffer_size` elements, then randomly\n    samples elements from this buffer, replacing the selected elements with new\n    elements. For perfect shuffling, a buffer size greater than or equal to the\n    full size of the dataset is required.\n\n    For instance, if your dataset contains 10,000 elements but `buffer_size` is\n    set to 1,000, then `shuffle` will initially select a random element from\n    only the first 1,000 elements in the buffer. Once an element is selected,\n    its space in the buffer is replaced by the next (i.e. 1,001-st) element,\n    maintaining the 1,000 element buffer.\n\n    `reshuffle_each_iteration` controls whether the shuffle order should be\n    different for each epoch. In TF 1.X, the idiomatic way to create epochs\n    was through the `repeat` transformation:\n\n    ```python\n    dataset = tf.data.Dataset.range(3)\n    dataset = dataset.shuffle(3, reshuffle_each_iteration=True)\n    dataset = dataset.repeat(2)\n    # [1, 0, 2, 1, 2, 0]\n\n    dataset = tf.data.Dataset.range(3)\n    dataset = dataset.shuffle(3, reshuffle_each_iteration=False)\n    dataset = dataset.repeat(2)\n    # [1, 0, 2, 1, 0, 2]\n    ```\n\n    In TF 2.0, `tf.data.Dataset` objects are Python iterables which makes it\n    possible to also create epochs through Python iteration:\n\n    ```python\n    dataset = tf.data.Dataset.range(3)\n    dataset = dataset.shuffle(3, reshuffle_each_iteration=True)\n    list(dataset.as_numpy_iterator())\n    # [1, 0, 2]\n    list(dataset.as_numpy_iterator())\n    # [1, 2, 0]\n    ```\n\n    ```python\n    dataset = tf.data.Dataset.range(3)\n    dataset = dataset.shuffle(3, reshuffle_each_iteration=False)\n    list(dataset.as_numpy_iterator())\n    # [1, 0, 2]\n    list(dataset.as_numpy_iterator())\n    # [1, 0, 2]\n    ```\n\n    #### Fully shuffling all the data\n\n    To shuffle an entire dataset, set `buffer_size=dataset.cardinality(). This\n    is equivalent to setting the `buffer_size` equal to the number of elements\n    in the dataset, resulting in uniform shuffle.\n\n    Note: `shuffle(dataset.cardinality())` loads the full dataset into memory so\n    that it can be shuffled. This will cause a memory overflow (OOM) error if\n    the dataset is too large, so full-shuffle should only be used for datasets\n    that are known to fit in the memory, such as datasets of filenames or other\n    small datasets.\n\n    ```python\n    dataset = tf.data.Dataset.range(20)\n    dataset = dataset.shuffle(dataset.cardinality())\n    # [18, 4, 9, 2, 17, 8, 5, 10, 0, 6, 16, 3, 19, 7, 14, 11, 15, 13, 12, 1]\n    ```\n\n    Args:\n      buffer_size: A `tf.int64` scalar `tf.Tensor`, representing the number of\n        elements from this dataset from which the new dataset will sample. To\n        uniformly shuffle the entire dataset, use\n        `buffer_size=dataset.cardinality()`.\n      seed: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the random\n        seed that will be used to create the distribution. See\n        `tf.random.set_seed` for behavior.\n      reshuffle_each_iteration: (Optional.) A boolean, which if true indicates\n        that the dataset should be pseudorandomly reshuffled each time it is\n        iterated over. (Defaults to `True`.)\n      name: (Optional.) A name for the tf.data operation.\n\n    Returns:\n      A new `Dataset` with the transformation applied as described above.\n    '
        return shuffle_op._shuffle(self, buffer_size, seed, reshuffle_each_iteration, name=name)

    def cache(self, filename='', name=None) -> 'DatasetV2':
        if False:
            while True:
                i = 10
        'Caches the elements in this dataset.\n\n    The first time the dataset is iterated over, its elements will be cached\n    either in the specified file or in memory. Subsequent iterations will\n    use the cached data.\n\n    Note: To guarantee that the cache gets finalized, the input dataset must be\n    iterated through in its entirety, until it raises StopIteration. Otherwise,\n    subsequent iterations may not use cached data.\n\n    >>> dataset = tf.data.Dataset.range(5)\n    >>> dataset = dataset.map(lambda x: x**2)\n    >>> dataset = dataset.cache()\n    >>> # The first time reading through the data will generate the data using\n    >>> # `range` and `map`.\n    >>> list(dataset.as_numpy_iterator())\n    [0, 1, 4, 9, 16]\n    >>> # Subsequent iterations read from the cache.\n    >>> list(dataset.as_numpy_iterator())\n    [0, 1, 4, 9, 16]\n\n    When caching to a file, the cached data will persist across runs. Even the\n    first iteration through the data will read from the cache file. Changing\n    the input pipeline before the call to `.cache()` will have no effect until\n    the cache file is removed or the filename is changed.\n\n    ```python\n    dataset = tf.data.Dataset.range(5)\n    dataset = dataset.cache("/path/to/file")\n    list(dataset.as_numpy_iterator())\n    # [0, 1, 2, 3, 4]\n    dataset = tf.data.Dataset.range(10)\n    dataset = dataset.cache("/path/to/file")  # Same file!\n    list(dataset.as_numpy_iterator())\n    # [0, 1, 2, 3, 4]\n    ```\n\n    Note: `cache` will produce exactly the same elements during each iteration\n    through the dataset. If you wish to randomize the iteration order, make sure\n    to call `shuffle` *after* calling `cache`.\n\n    Args:\n      filename: A `tf.string` scalar `tf.Tensor`, representing the name of a\n        directory on the filesystem to use for caching elements in this Dataset.\n        If a filename is not provided, the dataset will be cached in memory.\n      name: (Optional.) A name for the tf.data operation.\n\n    Returns:\n      A new `Dataset` with the transformation applied as described above.\n    '
        from tensorflow.python.data.ops import cache_op
        return cache_op._cache(self, filename, name)

    def take(self, count, name=None) -> 'DatasetV2':
        if False:
            i = 10
            return i + 15
        'Creates a `Dataset` with at most `count` elements from this dataset.\n\n    >>> dataset = tf.data.Dataset.range(10)\n    >>> dataset = dataset.take(3)\n    >>> list(dataset.as_numpy_iterator())\n    [0, 1, 2]\n\n    Args:\n      count: A `tf.int64` scalar `tf.Tensor`, representing the number of\n        elements of this dataset that should be taken to form the new dataset.\n        If `count` is -1, or if `count` is greater than the size of this\n        dataset, the new dataset will contain all elements of this dataset.\n      name: (Optional.) A name for the tf.data operation.\n\n    Returns:\n      A new `Dataset` with the transformation applied as described above.\n    '
        from tensorflow.python.data.ops import take_op
        return take_op._take(self, count, name=name)

    def skip(self, count, name=None) -> 'DatasetV2':
        if False:
            print('Hello World!')
        'Creates a `Dataset` that skips `count` elements from this dataset.\n\n    >>> dataset = tf.data.Dataset.range(10)\n    >>> dataset = dataset.skip(7)\n    >>> list(dataset.as_numpy_iterator())\n    [7, 8, 9]\n\n    Args:\n      count: A `tf.int64` scalar `tf.Tensor`, representing the number of\n        elements of this dataset that should be skipped to form the new dataset.\n        If `count` is greater than the size of this dataset, the new dataset\n        will contain no elements.  If `count` is -1, skips the entire dataset.\n      name: (Optional.) A name for the tf.data operation.\n\n    Returns:\n      A new `Dataset` with the transformation applied as described above.\n    '
        from tensorflow.python.data.ops import skip_op
        return skip_op._skip(self, count, name)

    def shard(self, num_shards, index, name=None) -> 'DatasetV2':
        if False:
            i = 10
            return i + 15
        "Creates a `Dataset` that includes only 1/`num_shards` of this dataset.\n\n    `shard` is deterministic. The Dataset produced by `A.shard(n, i)` will\n    contain all elements of A whose index mod n = i.\n\n    >>> A = tf.data.Dataset.range(10)\n    >>> B = A.shard(num_shards=3, index=0)\n    >>> list(B.as_numpy_iterator())\n    [0, 3, 6, 9]\n    >>> C = A.shard(num_shards=3, index=1)\n    >>> list(C.as_numpy_iterator())\n    [1, 4, 7]\n    >>> D = A.shard(num_shards=3, index=2)\n    >>> list(D.as_numpy_iterator())\n    [2, 5, 8]\n\n    This dataset operator is very useful when running distributed training, as\n    it allows each worker to read a unique subset.\n\n    When reading a single input file, you can shard elements as follows:\n\n    ```python\n    d = tf.data.TFRecordDataset(input_file)\n    d = d.shard(num_workers, worker_index)\n    d = d.repeat(num_epochs)\n    d = d.shuffle(shuffle_buffer_size)\n    d = d.map(parser_fn, num_parallel_calls=num_map_threads)\n    ```\n\n    Important caveats:\n\n    - Be sure to shard before you use any randomizing operator (such as\n      shuffle).\n    - Generally it is best if the shard operator is used early in the dataset\n      pipeline. For example, when reading from a set of TFRecord files, shard\n      before converting the dataset to input samples. This avoids reading every\n      file on every worker. The following is an example of an efficient\n      sharding strategy within a complete pipeline:\n\n    ```python\n    d = Dataset.list_files(pattern, shuffle=False)\n    d = d.shard(num_workers, worker_index)\n    d = d.repeat(num_epochs)\n    d = d.shuffle(shuffle_buffer_size)\n    d = d.interleave(tf.data.TFRecordDataset,\n                     cycle_length=num_readers, block_length=1)\n    d = d.map(parser_fn, num_parallel_calls=num_map_threads)\n    ```\n\n    Args:\n      num_shards: A `tf.int64` scalar `tf.Tensor`, representing the number of\n        shards operating in parallel.\n      index: A `tf.int64` scalar `tf.Tensor`, representing the worker index.\n      name: (Optional.) A name for the tf.data operation.\n\n    Returns:\n      A new `Dataset` with the transformation applied as described above.\n\n    Raises:\n      InvalidArgumentError: if `num_shards` or `index` are illegal values.\n\n        Note: error checking is done on a best-effort basis, and errors aren't\n        guaranteed to be caught upon dataset creation. (e.g. providing in a\n        placeholder tensor bypasses the early checking, and will instead result\n        in an error during a session.run call.)\n    "
        from tensorflow.python.data.ops import shard_op
        return shard_op._shard(self, num_shards, index, name=name)

    def save(self, path, compression=None, shard_func=None, checkpoint_args=None):
        if False:
            for i in range(10):
                print('nop')
        'Saves the content of the given dataset.\n\n      Example usage:\n\n      >>> import tempfile\n      >>> path = os.path.join(tempfile.gettempdir(), "saved_data")\n      >>> # Save a dataset\n      >>> dataset = tf.data.Dataset.range(2)\n      >>> dataset.save(path)\n      >>> new_dataset = tf.data.Dataset.load(path)\n      >>> for elem in new_dataset:\n      ...   print(elem)\n      tf.Tensor(0, shape=(), dtype=int64)\n      tf.Tensor(1, shape=(), dtype=int64)\n\n      The saved dataset is saved in multiple file "shards". By default, the\n      dataset output is divided to shards in a round-robin fashion but custom\n      sharding can be specified via the `shard_func` function. For example, you\n      can save the dataset to using a single shard as follows:\n\n      ```python\n      dataset = make_dataset()\n      def custom_shard_func(element):\n        return np.int64(0)\n      dataset.save(\n          path="/path/to/data", ..., shard_func=custom_shard_func)\n      ```\n\n      To enable checkpointing, pass in `checkpoint_args` to the `save` method\n      as follows:\n\n      ```python\n      dataset = tf.data.Dataset.range(100)\n      save_dir = "..."\n      checkpoint_prefix = "..."\n      step_counter = tf.Variable(0, trainable=False)\n      checkpoint_args = {\n        "checkpoint_interval": 50,\n        "step_counter": step_counter,\n        "directory": checkpoint_prefix,\n        "max_to_keep": 20,\n      }\n      dataset.save(dataset, save_dir, checkpoint_args=checkpoint_args)\n      ```\n\n      NOTE: The directory layout and file format used for saving the dataset is\n      considered an implementation detail and may change. For this reason,\n      datasets saved through `tf.data.Dataset.save` should only be consumed\n      through `tf.data.Dataset.load`, which is guaranteed to be\n      backwards compatible.\n\n    Args:\n     path: Required. A directory to use for saving the dataset.\n     compression: Optional. The algorithm to use to compress data when writing\n          it. Supported options are `GZIP` and `NONE`. Defaults to `NONE`.\n     shard_func: Optional. A function to control the mapping of dataset\n          elements to file shards. The function is expected to map elements of\n          the input dataset to int64 shard IDs. If present, the function will be\n          traced and executed as graph computation.\n     checkpoint_args: Optional args for checkpointing which will be passed into\n          the `tf.train.CheckpointManager`. If `checkpoint_args` are not\n          specified, then checkpointing will not be performed. The `save()`\n          implementation creates a `tf.train.Checkpoint` object internally, so\n          users should not set the `checkpoint` argument in `checkpoint_args`.\n\n    Returns:\n      An operation which when executed performs the save. When writing\n      checkpoints, returns None. The return value is useful in unit tests.\n\n    Raises:\n      ValueError if `checkpoint` is passed into `checkpoint_args`.\n    '
        from tensorflow.python.data.ops import save_op
        return save_op._save(self, path, compression, shard_func, checkpoint_args)

    @staticmethod
    def load(path, element_spec=None, compression=None, reader_func=None) -> 'DatasetV2':
        if False:
            while True:
                i = 10
        'Loads a previously saved dataset.\n\n    Example usage:\n\n    >>> import tempfile\n    >>> path = os.path.join(tempfile.gettempdir(), "saved_data")\n    >>> # Save a dataset\n    >>> dataset = tf.data.Dataset.range(2)\n    >>> tf.data.Dataset.save(dataset, path)\n    >>> new_dataset = tf.data.Dataset.load(path)\n    >>> for elem in new_dataset:\n    ...   print(elem)\n    tf.Tensor(0, shape=(), dtype=int64)\n    tf.Tensor(1, shape=(), dtype=int64)\n\n\n    If the default option of sharding the saved dataset was used, the element\n    order of the saved dataset will be preserved when loading it.\n\n    The `reader_func` argument can be used to specify a custom order in which\n    elements should be loaded from the individual shards. The `reader_func` is\n    expected to take a single argument -- a dataset of datasets, each containing\n    elements of one of the shards -- and return a dataset of elements. For\n    example, the order of shards can be shuffled when loading them as follows:\n\n    ```python\n    def custom_reader_func(datasets):\n      datasets = datasets.shuffle(NUM_SHARDS)\n      return datasets.interleave(lambda x: x, num_parallel_calls=AUTOTUNE)\n\n    dataset = tf.data.Dataset.load(\n        path="/path/to/data", ..., reader_func=custom_reader_func)\n    ```\n\n    Args:\n      path: Required. A path pointing to a previously saved dataset.\n      element_spec: Optional. A nested structure of `tf.TypeSpec` objects\n        matching the structure of an element of the saved dataset and specifying\n        the type of individual element components. If not provided, the nested\n        structure of `tf.TypeSpec` saved with the saved dataset is used. Note\n        that this argument is required in graph mode.\n      compression: Optional. The algorithm to use to decompress the data when\n        reading it. Supported options are `GZIP` and `NONE`. Defaults to `NONE`.\n      reader_func: Optional. A function to control how to read data from shards.\n        If present, the function will be traced and executed as graph\n        computation.\n\n    Returns:\n      A `tf.data.Dataset` instance.\n\n    Raises:\n      FileNotFoundError: If `element_spec` is not specified and the saved nested\n        structure of `tf.TypeSpec` can not be located with the saved dataset.\n      ValueError: If `element_spec` is not specified and the method is executed\n        in graph mode.\n    '
        from tensorflow.python.data.ops import load_op
        return load_op._load(path=path, element_spec=element_spec, compression=compression, reader_func=reader_func)

    def batch(self, batch_size, drop_remainder=False, num_parallel_calls=None, deterministic=None, name=None) -> 'DatasetV2':
        if False:
            i = 10
            return i + 15
        'Combines consecutive elements of this dataset into batches.\n\n    >>> dataset = tf.data.Dataset.range(8)\n    >>> dataset = dataset.batch(3)\n    >>> list(dataset.as_numpy_iterator())\n    [array([0, 1, 2]), array([3, 4, 5]), array([6, 7])]\n\n    >>> dataset = tf.data.Dataset.range(8)\n    >>> dataset = dataset.batch(3, drop_remainder=True)\n    >>> list(dataset.as_numpy_iterator())\n    [array([0, 1, 2]), array([3, 4, 5])]\n\n    The components of the resulting element will have an additional outer\n    dimension, which will be `batch_size` (or `N % batch_size` for the last\n    element if `batch_size` does not divide the number of input elements `N`\n    evenly and `drop_remainder` is `False`). If your program depends on the\n    batches having the same outer dimension, you should set the `drop_remainder`\n    argument to `True` to prevent the smaller batch from being produced.\n\n    Note: If your program requires data to have a statically known shape (e.g.,\n    when using XLA), you should use `drop_remainder=True`. Without\n    `drop_remainder=True` the shape of the output dataset will have an unknown\n    leading dimension due to the possibility of a smaller final batch.\n\n    Args:\n      batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of\n        consecutive elements of this dataset to combine in a single batch.\n      drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing\n        whether the last batch should be dropped in the case it has fewer than\n        `batch_size` elements; the default behavior is not to drop the smaller\n        batch.\n      num_parallel_calls: (Optional.) A `tf.int64` scalar `tf.Tensor`,\n        representing the number of batches to compute asynchronously in\n        parallel.\n        If not specified, batches will be computed sequentially. If the value\n        `tf.data.AUTOTUNE` is used, then the number of parallel\n        calls is set dynamically based on available resources.\n      deterministic: (Optional.) When `num_parallel_calls` is specified, if this\n        boolean is specified (`True` or `False`), it controls the order in which\n        the transformation produces elements. If set to `False`, the\n        transformation is allowed to yield elements out of order to trade\n        determinism for performance. If not specified, the\n        `tf.data.Options.deterministic` option (`True` by default) controls the\n        behavior.\n      name: (Optional.) A name for the tf.data operation.\n\n    Returns:\n      A new `Dataset` with the transformation applied as described above.\n    '
        from tensorflow.python.data.ops import batch_op
        return batch_op._batch(self, batch_size, drop_remainder, num_parallel_calls, deterministic, name)

    def padded_batch(self, batch_size, padded_shapes=None, padding_values=None, drop_remainder=False, name=None) -> 'DatasetV2':
        if False:
            return 10
        'Combines consecutive elements of this dataset into padded batches.\n\n    This transformation combines multiple consecutive elements of the input\n    dataset into a single element.\n\n    Like `tf.data.Dataset.batch`, the components of the resulting element will\n    have an additional outer dimension, which will be `batch_size` (or\n    `N % batch_size` for the last element if `batch_size` does not divide the\n    number of input elements `N` evenly and `drop_remainder` is `False`). If\n    your program depends on the batches having the same outer dimension, you\n    should set the `drop_remainder` argument to `True` to prevent the smaller\n    batch from being produced.\n\n    Unlike `tf.data.Dataset.batch`, the input elements to be batched may have\n    different shapes, and this transformation will pad each component to the\n    respective shape in `padded_shapes`. The `padded_shapes` argument\n    determines the resulting shape for each dimension of each component in an\n    output element:\n\n    * If the dimension is a constant, the component will be padded out to that\n      length in that dimension.\n    * If the dimension is unknown, the component will be padded out to the\n      maximum length of all elements in that dimension.\n\n    >>> A = (tf.data.Dataset\n    ...      .range(1, 5, output_type=tf.int32)\n    ...      .map(lambda x: tf.fill([x], x)))\n    >>> # Pad to the smallest per-batch size that fits all elements.\n    >>> B = A.padded_batch(2)\n    >>> for element in B.as_numpy_iterator():\n    ...   print(element)\n    [[1 0]\n     [2 2]]\n    [[3 3 3 0]\n     [4 4 4 4]]\n    >>> # Pad to a fixed size.\n    >>> C = A.padded_batch(2, padded_shapes=5)\n    >>> for element in C.as_numpy_iterator():\n    ...   print(element)\n    [[1 0 0 0 0]\n     [2 2 0 0 0]]\n    [[3 3 3 0 0]\n     [4 4 4 4 0]]\n    >>> # Pad with a custom value.\n    >>> D = A.padded_batch(2, padded_shapes=5, padding_values=-1)\n    >>> for element in D.as_numpy_iterator():\n    ...   print(element)\n    [[ 1 -1 -1 -1 -1]\n     [ 2  2 -1 -1 -1]]\n    [[ 3  3  3 -1 -1]\n     [ 4  4  4  4 -1]]\n    >>> # Components of nested elements can be padded independently.\n    >>> elements = [([1, 2, 3], [10]),\n    ...             ([4, 5], [11, 12])]\n    >>> dataset = tf.data.Dataset.from_generator(\n    ...     lambda: iter(elements), (tf.int32, tf.int32))\n    >>> # Pad the first component of the tuple to length 4, and the second\n    >>> # component to the smallest size that fits.\n    >>> dataset = dataset.padded_batch(2,\n    ...     padded_shapes=([4], [None]),\n    ...     padding_values=(-1, 100))\n    >>> list(dataset.as_numpy_iterator())\n    [(array([[ 1,  2,  3, -1], [ 4,  5, -1, -1]], dtype=int32),\n      array([[ 10, 100], [ 11,  12]], dtype=int32))]\n    >>> # Pad with a single value and multiple components.\n    >>> E = tf.data.Dataset.zip((A, A)).padded_batch(2, padding_values=-1)\n    >>> for element in E.as_numpy_iterator():\n    ...   print(element)\n    (array([[ 1, -1],\n           [ 2,  2]], dtype=int32), array([[ 1, -1],\n           [ 2,  2]], dtype=int32))\n    (array([[ 3,  3,  3, -1],\n           [ 4,  4,  4,  4]], dtype=int32), array([[ 3,  3,  3, -1],\n           [ 4,  4,  4,  4]], dtype=int32))\n\n    See also `tf.data.experimental.dense_to_sparse_batch`, which combines\n    elements that may have different shapes into a `tf.sparse.SparseTensor`.\n\n    Args:\n      batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of\n        consecutive elements of this dataset to combine in a single batch.\n      padded_shapes: (Optional.) A (nested) structure of `tf.TensorShape` or\n        `tf.int64` vector tensor-like objects representing the shape to which\n        the respective component of each input element should be padded prior\n        to batching. Any unknown dimensions will be padded to the maximum size\n        of that dimension in each batch. If unset, all dimensions of all\n        components are padded to the maximum size in the batch. `padded_shapes`\n        must be set if any component has an unknown rank.\n      padding_values: (Optional.) A (nested) structure of scalar-shaped\n        `tf.Tensor`, representing the padding values to use for the respective\n        components. None represents that the (nested) structure should be padded\n        with default values.  Defaults are `0` for numeric types and the empty\n        string for string types. The `padding_values` should have the same\n        (nested) structure as the input dataset. If `padding_values` is a single\n        element and the input dataset has multiple components, then the same\n        `padding_values` will be used to pad every component of the dataset.\n        If `padding_values` is a scalar, then its value will be broadcasted\n        to match the shape of each component.\n      drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing\n        whether the last batch should be dropped in the case it has fewer than\n        `batch_size` elements; the default behavior is not to drop the smaller\n        batch.\n      name: (Optional.) A name for the tf.data operation.\n\n    Returns:\n      A new `Dataset` with the transformation applied as described above.\n\n    Raises:\n      ValueError: If a component has an unknown rank, and the `padded_shapes`\n        argument is not set.\n      TypeError: If a component is of an unsupported type. The list of supported\n        types is documented in\n        https://www.tensorflow.org/guide/data#dataset_structure.\n    '
        from tensorflow.python.data.ops import padded_batch_op
        return padded_batch_op._padded_batch(self, batch_size, padded_shapes, padding_values, drop_remainder, name)

    def ragged_batch(self, batch_size, drop_remainder=False, row_splits_dtype=dtypes.int64, name=None) -> 'DatasetV2':
        if False:
            print('Hello World!')
        'Combines consecutive elements of this dataset into `tf.RaggedTensor`s.\n\n    Like `tf.data.Dataset.batch`, the components of the resulting element will\n    have an additional outer dimension, which will be `batch_size` (or\n    `N % batch_size` for the last element if `batch_size` does not divide the\n    number of input elements `N` evenly and `drop_remainder` is `False`). If\n    your program depends on the batches having the same outer dimension, you\n    should set the `drop_remainder` argument to `True` to prevent the smaller\n    batch from being produced.\n\n    Unlike `tf.data.Dataset.batch`, the input elements to be batched may have\n    different shapes:\n\n    *  If an input element is a `tf.Tensor` whose static `tf.TensorShape` is\n    fully defined, then it is batched as normal.\n    *  If an input element is a `tf.Tensor` whose static `tf.TensorShape`\n    contains one or more axes with unknown size (i.e., `shape[i]=None`), then\n    the output will contain a `tf.RaggedTensor` that is ragged up to any of such\n    dimensions.\n    *  If an input element is a `tf.RaggedTensor` or any other type, then it is\n    batched as normal.\n\n    Example:\n\n    >>> dataset = tf.data.Dataset.range(6)\n    >>> dataset = dataset.map(lambda x: tf.range(x))\n    >>> dataset.element_spec.shape\n    TensorShape([None])\n    >>> dataset = dataset.ragged_batch(2)\n    >>> for batch in dataset:\n    ...   print(batch)\n    <tf.RaggedTensor [[], [0]]>\n    <tf.RaggedTensor [[0, 1], [0, 1, 2]]>\n    <tf.RaggedTensor [[0, 1, 2, 3], [0, 1, 2, 3, 4]]>\n\n    Args:\n      batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of\n        consecutive elements of this dataset to combine in a single batch.\n      drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing\n        whether the last batch should be dropped in the case it has fewer than\n        `batch_size` elements; the default behavior is not to drop the smaller\n        batch.\n      row_splits_dtype: The dtype that should be used for the `row_splits` of\n        any new ragged tensors.  Existing `tf.RaggedTensor` elements do not have\n        their row_splits dtype changed.\n      name: (Optional.) A string indicating a name for the `tf.data` operation.\n\n    Returns:\n      A new `Dataset` with the transformation applied as described above.\n    '
        from tensorflow.python.data.ops import ragged_batch_op
        return ragged_batch_op._ragged_batch(self, batch_size, drop_remainder, row_splits_dtype, name)

    def sparse_batch(self, batch_size, row_shape, name=None) -> 'DatasetV2':
        if False:
            while True:
                i = 10
        "Combines consecutive elements into `tf.sparse.SparseTensor`s.\n\n    Like `Dataset.padded_batch()`, this transformation combines multiple\n    consecutive elements of the dataset, which might have different\n    shapes, into a single element. The resulting element has three\n    components (`indices`, `values`, and `dense_shape`), which\n    comprise a `tf.sparse.SparseTensor` that represents the same data. The\n    `row_shape` represents the dense shape of each row in the\n    resulting `tf.sparse.SparseTensor`, to which the effective batch size is\n    prepended. For example:\n\n    ```python\n    # NOTE: The following examples use `{ ... }` to represent the\n    # contents of a dataset.\n    a = { ['a', 'b', 'c'], ['a', 'b'], ['a', 'b', 'c', 'd'] }\n\n    a.apply(tf.data.experimental.dense_to_sparse_batch(\n        batch_size=2, row_shape=[6])) ==\n    {\n        ([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]],  # indices\n         ['a', 'b', 'c', 'a', 'b'],                 # values\n         [2, 6]),                                   # dense_shape\n        ([[0, 0], [0, 1], [0, 2], [0, 3]],\n         ['a', 'b', 'c', 'd'],\n         [1, 6])\n    }\n    ```\n\n    Args:\n      batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of\n        consecutive elements of this dataset to combine in a single batch.\n      row_shape: A `tf.TensorShape` or `tf.int64` vector tensor-like object\n        representing the equivalent dense shape of a row in the resulting\n        `tf.sparse.SparseTensor`. Each element of this dataset must have the\n        same rank as `row_shape`, and must have size less than or equal to\n        `row_shape` in each dimension.\n      name: (Optional.) A string indicating a name for the `tf.data` operation.\n\n    Returns:\n      A new `Dataset` with the transformation applied as described above.\n    "
        from tensorflow.python.data.ops import sparse_batch_op
        return sparse_batch_op._sparse_batch(self, batch_size, row_shape, name)

    def map(self, map_func, num_parallel_calls=None, deterministic=None, name=None) -> 'DatasetV2':
        if False:
            print('Hello World!')
        'Maps `map_func` across the elements of this dataset.\n\n    This transformation applies `map_func` to each element of this dataset, and\n    returns a new dataset containing the transformed elements, in the same\n    order as they appeared in the input. `map_func` can be used to change both\n    the values and the structure of a dataset\'s elements. Supported structure\n    constructs are documented\n    [here](https://www.tensorflow.org/guide/data#dataset_structure).\n\n    For example, `map` can be used for adding 1 to each element, or projecting a\n    subset of element components.\n\n    >>> dataset = Dataset.range(1, 6)  # ==> [ 1, 2, 3, 4, 5 ]\n    >>> dataset = dataset.map(lambda x: x + 1)\n    >>> list(dataset.as_numpy_iterator())\n    [2, 3, 4, 5, 6]\n\n    The input signature of `map_func` is determined by the structure of each\n    element in this dataset.\n\n    >>> dataset = Dataset.range(5)\n    >>> # `map_func` takes a single argument of type `tf.Tensor` with the same\n    >>> # shape and dtype.\n    >>> result = dataset.map(lambda x: x + 1)\n\n    >>> # Each element is a tuple containing two `tf.Tensor` objects.\n    >>> elements = [(1, "foo"), (2, "bar"), (3, "baz")]\n    >>> dataset = tf.data.Dataset.from_generator(\n    ...     lambda: elements, (tf.int32, tf.string))\n    >>> # `map_func` takes two arguments of type `tf.Tensor`. This function\n    >>> # projects out just the first component.\n    >>> result = dataset.map(lambda x_int, y_str: x_int)\n    >>> list(result.as_numpy_iterator())\n    [1, 2, 3]\n\n    >>> # Each element is a dictionary mapping strings to `tf.Tensor` objects.\n    >>> elements =  ([{"a": 1, "b": "foo"},\n    ...               {"a": 2, "b": "bar"},\n    ...               {"a": 3, "b": "baz"}])\n    >>> dataset = tf.data.Dataset.from_generator(\n    ...     lambda: elements, {"a": tf.int32, "b": tf.string})\n    >>> # `map_func` takes a single argument of type `dict` with the same keys\n    >>> # as the elements.\n    >>> result = dataset.map(lambda d: str(d["a"]) + d["b"])\n\n    The value or values returned by `map_func` determine the structure of each\n    element in the returned dataset.\n\n    >>> dataset = tf.data.Dataset.range(3)\n    >>> # `map_func` returns two `tf.Tensor` objects.\n    >>> def g(x):\n    ...   return tf.constant(37.0), tf.constant(["Foo", "Bar", "Baz"])\n    >>> result = dataset.map(g)\n    >>> result.element_spec\n    (TensorSpec(shape=(), dtype=tf.float32, name=None), TensorSpec(shape=(3,), dtype=tf.string, name=None))\n    >>> # Python primitives, lists, and NumPy arrays are implicitly converted to\n    >>> # `tf.Tensor`.\n    >>> def h(x):\n    ...   return 37.0, ["Foo", "Bar"], np.array([1.0, 2.0], dtype=np.float64)\n    >>> result = dataset.map(h)\n    >>> result.element_spec\n    (TensorSpec(shape=(), dtype=tf.float32, name=None), TensorSpec(shape=(2,), dtype=tf.string, name=None), TensorSpec(shape=(2,), dtype=tf.float64, name=None))\n    >>> # `map_func` can return nested structures.\n    >>> def i(x):\n    ...   return (37.0, [42, 16]), "foo"\n    >>> result = dataset.map(i)\n    >>> result.element_spec\n    ((TensorSpec(shape=(), dtype=tf.float32, name=None),\n      TensorSpec(shape=(2,), dtype=tf.int32, name=None)),\n     TensorSpec(shape=(), dtype=tf.string, name=None))\n\n    `map_func` can accept as arguments and return any type of dataset element.\n\n    Note that irrespective of the context in which `map_func` is defined (eager\n    vs. graph), tf.data traces the function and executes it as a graph. To use\n    Python code inside of the function you have a few options:\n\n    1) Rely on AutoGraph to convert Python code into an equivalent graph\n    computation. The downside of this approach is that AutoGraph can convert\n    some but not all Python code.\n\n    2) Use `tf.py_function`, which allows you to write arbitrary Python code but\n    will generally result in worse performance than 1). For example:\n\n    >>> d = tf.data.Dataset.from_tensor_slices([\'hello\', \'world\'])\n    >>> # transform a string tensor to upper case string using a Python function\n    >>> def upper_case_fn(t: tf.Tensor):\n    ...   return t.numpy().decode(\'utf-8\').upper()\n    >>> d = d.map(lambda x: tf.py_function(func=upper_case_fn,\n    ...           inp=[x], Tout=tf.string))\n    >>> list(d.as_numpy_iterator())\n    [b\'HELLO\', b\'WORLD\']\n\n    3) Use `tf.numpy_function`, which also allows you to write arbitrary\n    Python code. Note that `tf.py_function` accepts `tf.Tensor` whereas\n    `tf.numpy_function` accepts numpy arrays and returns only numpy arrays.\n    For example:\n\n    >>> d = tf.data.Dataset.from_tensor_slices([\'hello\', \'world\'])\n    >>> def upper_case_fn(t: np.ndarray):\n    ...   return t.decode(\'utf-8\').upper()\n    >>> d = d.map(lambda x: tf.numpy_function(func=upper_case_fn,\n    ...           inp=[x], Tout=tf.string))\n    >>> list(d.as_numpy_iterator())\n    [b\'HELLO\', b\'WORLD\']\n\n    Note that the use of `tf.numpy_function` and `tf.py_function`\n    in general precludes the possibility of executing user-defined\n    transformations in parallel (because of Python GIL).\n\n    Performance can often be improved by setting `num_parallel_calls` so that\n    `map` will use multiple threads to process elements. If deterministic order\n    isn\'t required, it can also improve performance to set\n    `deterministic=False`.\n\n    >>> dataset = Dataset.range(1, 6)  # ==> [ 1, 2, 3, 4, 5 ]\n    >>> dataset = dataset.map(lambda x: x + 1,\n    ...     num_parallel_calls=tf.data.AUTOTUNE,\n    ...     deterministic=False)\n\n    The order of elements yielded by this transformation is deterministic if\n    `deterministic=True`. If `map_func` contains stateful operations and\n    `num_parallel_calls > 1`, the order in which that state is accessed is\n    undefined, so the values of output elements may not be deterministic\n    regardless of the `deterministic` flag value.\n\n    Args:\n      map_func: A function mapping a dataset element to another dataset element.\n      num_parallel_calls: (Optional.) A `tf.int64` scalar `tf.Tensor`,\n        representing the number elements to process asynchronously in parallel.\n        If not specified, elements will be processed sequentially. If the value\n        `tf.data.AUTOTUNE` is used, then the number of parallel\n        calls is set dynamically based on available CPU.\n      deterministic: (Optional.) When `num_parallel_calls` is specified, if this\n        boolean is specified (`True` or `False`), it controls the order in which\n        the transformation produces elements. If set to `False`, the\n        transformation is allowed to yield elements out of order to trade\n        determinism for performance. If not specified, the\n        `tf.data.Options.deterministic` option (`True` by default) controls the\n        behavior.\n      name: (Optional.) A name for the tf.data operation.\n\n    Returns:\n      A new `Dataset` with the transformation applied as described above.\n    '
        from tensorflow.python.data.ops import map_op
        return map_op._map_v2(self, map_func, num_parallel_calls=num_parallel_calls, deterministic=deterministic, name=name)

    def flat_map(self, map_func, name=None) -> 'DatasetV2':
        if False:
            print('Hello World!')
        'Maps `map_func` across this dataset and flattens the result.\n\n    The type signature is:\n\n    ```\n    def flat_map(\n      self: Dataset[T],\n      map_func: Callable[[T], Dataset[S]]\n    ) -> Dataset[S]\n    ```\n\n    Use `flat_map` if you want to make sure that the order of your dataset\n    stays the same. For example, to flatten a dataset of batches into a\n    dataset of their elements:\n\n    >>> dataset = tf.data.Dataset.from_tensor_slices(\n    ...     [[1, 2, 3], [4, 5, 6], [7, 8, 9]])\n    >>> dataset = dataset.flat_map(tf.data.Dataset.from_tensor_slices)\n    >>> list(dataset.as_numpy_iterator())\n    [1, 2, 3, 4, 5, 6, 7, 8, 9]\n\n    `tf.data.Dataset.interleave()` is a generalization of `flat_map`, since\n    `flat_map` produces the same output as\n    `tf.data.Dataset.interleave(cycle_length=1)`\n\n    Args:\n      map_func: A function mapping a dataset element to a dataset.\n      name: (Optional.) A name for the tf.data operation.\n\n    Returns:\n      A new `Dataset` with the transformation applied as described above.\n    '
        from tensorflow.python.data.ops import flat_map_op
        return flat_map_op._flat_map(self, map_func, name=name)

    def ignore_errors(self, log_warning=False, name=None) -> 'DatasetV2':
        if False:
            while True:
                i = 10
        'Drops elements that cause errors.\n\n    >>> dataset = tf.data.Dataset.from_tensor_slices([1., 2., 0., 4.])\n    >>> dataset = dataset.map(lambda x: tf.debugging.check_numerics(1. / x, ""))\n    >>> list(dataset.as_numpy_iterator())\n    Traceback (most recent call last):\n    ...\n    InvalidArgumentError: ... Tensor had Inf values\n    >>> dataset = dataset.ignore_errors()\n    >>> list(dataset.as_numpy_iterator())\n    [1.0, 0.5, 0.25]\n\n    Args:\n      log_warning: (Optional.) A bool indicating whether or not ignored errors\n        should be logged to stderr. Defaults to `False`.\n      name: (Optional.) A string indicating a name for the `tf.data` operation.\n\n    Returns:\n      A new `Dataset` with the transformation applied as described above.\n    '
        from tensorflow.python.data.ops import ignore_errors_op
        return ignore_errors_op._ignore_errors(self, log_warning, name)

    def interleave(self, map_func, cycle_length=None, block_length=None, num_parallel_calls=None, deterministic=None, name=None) -> 'DatasetV2':
        if False:
            i = 10
            return i + 15
        'Maps `map_func` across this dataset, and interleaves the results.\n\n    The type signature is:\n\n    ```\n    def interleave(\n      self: Dataset[T],\n      map_func: Callable[[T], Dataset[S]]\n    ) -> Dataset[S]\n    ```\n\n    For example, you can use `Dataset.interleave()` to process many input files\n    concurrently:\n\n    >>> # Preprocess 4 files concurrently, and interleave blocks of 16 records\n    >>> # from each file.\n    >>> filenames = ["/var/data/file1.txt", "/var/data/file2.txt",\n    ...              "/var/data/file3.txt", "/var/data/file4.txt"]\n    >>> dataset = tf.data.Dataset.from_tensor_slices(filenames)\n    >>> def parse_fn(filename):\n    ...   return tf.data.Dataset.range(10)\n    >>> dataset = dataset.interleave(lambda x:\n    ...     tf.data.TextLineDataset(x).map(parse_fn, num_parallel_calls=1),\n    ...     cycle_length=4, block_length=16)\n\n    The `cycle_length` and `block_length` arguments control the order in which\n    elements are produced. `cycle_length` controls the number of input elements\n    that are processed concurrently. If you set `cycle_length` to 1, this\n    transformation will handle one input element at a time, and will produce\n    identical results to `tf.data.Dataset.flat_map`. In general,\n    this transformation will apply `map_func` to `cycle_length` input elements,\n    open iterators on the returned `Dataset` objects, and cycle through them\n    producing `block_length` consecutive elements from each iterator, and\n    consuming the next input element each time it reaches the end of an\n    iterator.\n\n    For example:\n\n    >>> dataset = Dataset.range(1, 6)  # ==> [ 1, 2, 3, 4, 5 ]\n    >>> # NOTE: New lines indicate "block" boundaries.\n    >>> dataset = dataset.interleave(\n    ...     lambda x: Dataset.from_tensors(x).repeat(6),\n    ...     cycle_length=2, block_length=4)\n    >>> list(dataset.as_numpy_iterator())\n    [1, 1, 1, 1,\n     2, 2, 2, 2,\n     1, 1,\n     2, 2,\n     3, 3, 3, 3,\n     4, 4, 4, 4,\n     3, 3,\n     4, 4,\n     5, 5, 5, 5,\n     5, 5]\n\n    Note: The order of elements yielded by this transformation is\n    deterministic, as long as `map_func` is a pure function and\n    `deterministic=True`. If `map_func` contains any stateful operations, the\n    order in which that state is accessed is undefined.\n\n    Performance can often be improved by setting `num_parallel_calls` so that\n    `interleave` will use multiple threads to fetch elements. If determinism\n    isn\'t required, it can also improve performance to set\n    `deterministic=False`.\n\n    >>> filenames = ["/var/data/file1.txt", "/var/data/file2.txt",\n    ...              "/var/data/file3.txt", "/var/data/file4.txt"]\n    >>> dataset = tf.data.Dataset.from_tensor_slices(filenames)\n    >>> dataset = dataset.interleave(lambda x: tf.data.TFRecordDataset(x),\n    ...     cycle_length=4, num_parallel_calls=tf.data.AUTOTUNE,\n    ...     deterministic=False)\n\n    Args:\n      map_func: A function that takes a dataset element and returns a\n        `tf.data.Dataset`.\n      cycle_length: (Optional.) The number of input elements that will be\n        processed concurrently. If not set, the tf.data runtime decides what it\n        should be based on available CPU. If `num_parallel_calls` is set to\n        `tf.data.AUTOTUNE`, the `cycle_length` argument identifies\n        the maximum degree of parallelism.\n      block_length: (Optional.) The number of consecutive elements to produce\n        from each input element before cycling to another input element. If not\n        set, defaults to 1.\n      num_parallel_calls: (Optional.) If specified, the implementation creates a\n        threadpool, which is used to fetch inputs from cycle elements\n        asynchronously and in parallel. The default behavior is to fetch inputs\n        from cycle elements synchronously with no parallelism. If the value\n        `tf.data.AUTOTUNE` is used, then the number of parallel\n        calls is set dynamically based on available CPU.\n      deterministic: (Optional.) When `num_parallel_calls` is specified, if this\n        boolean is specified (`True` or `False`), it controls the order in which\n        the transformation produces elements. If set to `False`, the\n        transformation is allowed to yield elements out of order to trade\n        determinism for performance. If not specified, the\n        `tf.data.Options.deterministic` option (`True` by default) controls the\n        behavior.\n      name: (Optional.) A name for the tf.data operation.\n\n    Returns:\n      A new `Dataset` with the transformation applied as described above.\n    '
        from tensorflow.python.data.ops import interleave_op
        return interleave_op._interleave(self, map_func, cycle_length, block_length, num_parallel_calls, deterministic, name)

    def filter(self, predicate, name=None) -> 'DatasetV2':
        if False:
            return 10
        'Filters this dataset according to `predicate`.\n\n    >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])\n    >>> dataset = dataset.filter(lambda x: x < 3)\n    >>> list(dataset.as_numpy_iterator())\n    [1, 2]\n    >>> # `tf.math.equal(x, y)` is required for equality comparison\n    >>> def filter_fn(x):\n    ...   return tf.math.equal(x, 1)\n    >>> dataset = dataset.filter(filter_fn)\n    >>> list(dataset.as_numpy_iterator())\n    [1]\n\n    Args:\n      predicate: A function mapping a dataset element to a boolean.\n      name: (Optional.) A name for the tf.data operation.\n\n    Returns:\n      A new `Dataset` with the transformation applied as described above.\n    '
        from tensorflow.python.data.ops import filter_op
        return filter_op._filter(self, predicate, name)

    def apply(self, transformation_func) -> 'DatasetV2':
        if False:
            i = 10
            return i + 15
        'Applies a transformation function to this dataset.\n\n    `apply` enables chaining of custom `Dataset` transformations, which are\n    represented as functions that take one `Dataset` argument and return a\n    transformed `Dataset`.\n\n    >>> dataset = tf.data.Dataset.range(100)\n    >>> def dataset_fn(ds):\n    ...   return ds.filter(lambda x: x < 5)\n    >>> dataset = dataset.apply(dataset_fn)\n    >>> list(dataset.as_numpy_iterator())\n    [0, 1, 2, 3, 4]\n\n    Args:\n      transformation_func: A function that takes one `Dataset` argument and\n        returns a `Dataset`.\n\n    Returns:\n      A new `Dataset` with the transformation applied as described above.\n    '
        dataset = transformation_func(self)
        if not isinstance(dataset, data_types.DatasetV2):
            raise TypeError(f'`transformation_func` must return a `tf.data.Dataset` object. Got {type(dataset)}.')
        dataset._input_datasets = [self]
        return dataset

    def window(self, size, shift=None, stride=1, drop_remainder=False, name=None) -> 'DatasetV2':
        if False:
            print('Hello World!')
        'Returns a dataset of "windows".\n\n    Each "window" is a dataset that contains a subset of elements of the\n    input dataset. These are finite datasets of size `size` (or possibly fewer\n    if there are not enough input elements to fill the window and\n    `drop_remainder` evaluates to `False`).\n\n    For example:\n\n    >>> dataset = tf.data.Dataset.range(7).window(3)\n    >>> for window in dataset:\n    ...   print(window)\n    <...Dataset element_spec=TensorSpec(shape=(), dtype=tf.int64, name=None)>\n    <...Dataset element_spec=TensorSpec(shape=(), dtype=tf.int64, name=None)>\n    <...Dataset element_spec=TensorSpec(shape=(), dtype=tf.int64, name=None)>\n\n    Since windows are datasets, they can be iterated over:\n\n    >>> for window in dataset:\n    ...   print(list(window.as_numpy_iterator()))\n    [0, 1, 2]\n    [3, 4, 5]\n    [6]\n\n    #### Shift\n\n    The `shift` argument determines the number of input elements to shift\n    between the start of each window. If windows and elements are both numbered\n    starting at 0, the first element in window `k` will be element `k * shift`\n    of the input dataset. In particular, the first element of the first window\n    will always be the first element of the input dataset.\n\n    >>> dataset = tf.data.Dataset.range(7).window(3, shift=1,\n    ...                                           drop_remainder=True)\n    >>> for window in dataset:\n    ...   print(list(window.as_numpy_iterator()))\n    [0, 1, 2]\n    [1, 2, 3]\n    [2, 3, 4]\n    [3, 4, 5]\n    [4, 5, 6]\n\n    #### Stride\n\n    The `stride` argument determines the stride between input elements within a\n    window.\n\n    >>> dataset = tf.data.Dataset.range(7).window(3, shift=1, stride=2,\n    ...                                           drop_remainder=True)\n    >>> for window in dataset:\n    ...   print(list(window.as_numpy_iterator()))\n    [0, 2, 4]\n    [1, 3, 5]\n    [2, 4, 6]\n\n    #### Nested elements\n\n    When the `window` transformation is applied to a dataset whos elements are\n    nested structures, it produces a dataset where the elements have the same\n    nested structure but each leaf is replaced by a window. In other words,\n    the nesting is applied outside of the windows as opposed inside of them.\n\n    The type signature is:\n\n    ```\n    def window(\n        self: Dataset[Nest[T]], ...\n    ) -> Dataset[Nest[Dataset[T]]]\n    ```\n\n    Applying `window` to a `Dataset` of tuples gives a tuple of windows:\n\n    >>> dataset = tf.data.Dataset.from_tensor_slices(([1, 2, 3, 4, 5],\n    ...                                               [6, 7, 8, 9, 10]))\n    >>> dataset = dataset.window(2)\n    >>> windows = next(iter(dataset))\n    >>> windows\n    (<...Dataset element_spec=TensorSpec(shape=(), dtype=tf.int32, name=None)>,\n     <...Dataset element_spec=TensorSpec(shape=(), dtype=tf.int32, name=None)>)\n\n    >>> def to_numpy(ds):\n    ...   return list(ds.as_numpy_iterator())\n    >>>\n    >>> for windows in dataset:\n    ...   print(to_numpy(windows[0]), to_numpy(windows[1]))\n    [1, 2] [6, 7]\n    [3, 4] [8, 9]\n    [5] [10]\n\n    Applying `window` to a `Dataset` of dictionaries gives a dictionary of\n    `Datasets`:\n\n    >>> dataset = tf.data.Dataset.from_tensor_slices({\'a\': [1, 2, 3],\n    ...                                               \'b\': [4, 5, 6],\n    ...                                               \'c\': [7, 8, 9]})\n    >>> dataset = dataset.window(2)\n    >>> def to_numpy(ds):\n    ...   return list(ds.as_numpy_iterator())\n    >>>\n    >>> for windows in dataset:\n    ...   print(tf.nest.map_structure(to_numpy, windows))\n    {\'a\': [1, 2], \'b\': [4, 5], \'c\': [7, 8]}\n    {\'a\': [3], \'b\': [6], \'c\': [9]}\n\n    #### Flatten a dataset of windows\n\n    The `Dataset.flat_map` and `Dataset.interleave` methods can be used to\n    flatten a dataset of windows into a single dataset.\n\n    The argument to `flat_map` is a function that takes an element from the\n    dataset and returns a `Dataset`. `flat_map` chains together the resulting\n    datasets sequentially.\n\n    For example, to turn each window into a dense tensor:\n\n    >>> dataset = tf.data.Dataset.range(7).window(3, shift=1,\n    ...                                           drop_remainder=True)\n    >>> batched = dataset.flat_map(lambda x:x.batch(3))\n    >>> for batch in batched:\n    ...   print(batch.numpy())\n    [0 1 2]\n    [1 2 3]\n    [2 3 4]\n    [3 4 5]\n    [4 5 6]\n\n    Args:\n      size: A `tf.int64` scalar `tf.Tensor`, representing the number of elements\n        of the input dataset to combine into a window. Must be positive.\n      shift: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the\n        number of input elements by which the window moves in each iteration.\n        Defaults to `size`. Must be positive.\n      stride: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the\n        stride of the input elements in the sliding window. Must be positive.\n        The default value of 1 means "retain every input element".\n      drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing\n        whether the last windows should be dropped if their size is smaller than\n        `size`.\n      name: (Optional.) A name for the tf.data operation.\n\n    Returns:\n      A new `Dataset` with the transformation applied as described above.\n    '
        from tensorflow.python.data.ops import window_op
        return window_op._window(self, size, shift, stride, drop_remainder, name)

    def reduce(self, initial_state, reduce_func, name=None):
        if False:
            while True:
                i = 10
        'Reduces the input dataset to a single element.\n\n    The transformation calls `reduce_func` successively on every element of\n    the input dataset until the dataset is exhausted, aggregating information in\n    its internal state. The `initial_state` argument is used for the initial\n    state and the final state is returned as the result.\n\n    >>> tf.data.Dataset.range(5).reduce(np.int64(0), lambda x, _: x + 1).numpy()\n    5\n    >>> tf.data.Dataset.range(5).reduce(np.int64(0), lambda x, y: x + y).numpy()\n    10\n\n    Args:\n      initial_state: An element representing the initial state of the\n        transformation.\n      reduce_func: A function that maps `(old_state, input_element)` to\n        `new_state`. It must take two arguments and return a new element\n        The structure of `new_state` must match the structure of\n        `initial_state`.\n      name: (Optional.) A name for the tf.data operation.\n\n    Returns:\n      A dataset element corresponding to the final state of the transformation.\n\n    '
        with ops.name_scope('initial_state'):
            initial_state = structure.normalize_element(initial_state)
        state_structure = structure.type_spec_from_value(initial_state)
        need_to_rerun = True
        while need_to_rerun:
            wrapped_func = structured_function.StructuredFunctionWrapper(reduce_func, 'reduce()', input_structure=(state_structure, self.element_spec), add_to_graph=False)
            output_classes = wrapped_func.output_classes
            state_classes = nest.map_structure(lambda component_spec: component_spec._to_legacy_output_classes(), state_structure)
            for (new_state_class, state_class) in zip(nest.flatten(output_classes), nest.flatten(state_classes)):
                if not issubclass(new_state_class, state_class):
                    raise TypeError(f'The element classes for the new state must match the initial state. Expected {state_classes} but got {wrapped_func.output_classes}.')
            output_types = wrapped_func.output_types
            state_types = nest.map_structure(lambda component_spec: component_spec._to_legacy_output_types(), state_structure)
            for (new_state_type, state_type) in zip(nest.flatten(output_types), nest.flatten(state_types)):
                if new_state_type != state_type:
                    raise TypeError(f'The element types for the new state must match the initial state. Expected {state_types} but got {wrapped_func.output_types}.')
            output_shapes = wrapped_func.output_shapes
            state_shapes = nest.map_structure(lambda component_spec: component_spec._to_legacy_output_shapes(), state_structure)
            flat_state_shapes = nest.flatten(state_shapes)
            flat_new_state_shapes = nest.flatten(output_shapes)
            weakened_state_shapes = [original.most_specific_compatible_shape(new) for (original, new) in zip(flat_state_shapes, flat_new_state_shapes)]
            need_to_rerun = False
            for (original_shape, weakened_shape) in zip(flat_state_shapes, weakened_state_shapes):
                if original_shape.ndims is not None and (weakened_shape.ndims is None or original_shape.as_list() != weakened_shape.as_list()):
                    need_to_rerun = True
                    break
            if need_to_rerun:
                state_structure = structure.convert_legacy_structure(state_types, nest.pack_sequence_as(state_shapes, weakened_state_shapes), state_classes)
        reduce_func = wrapped_func.function
        reduce_func.add_to_graph(ops.get_default_graph())
        dataset = self._apply_debug_options()
        metadata = dataset_metadata_pb2.Metadata()
        if name:
            metadata.name = _validate_and_encode(name)
        return structure.from_compatible_tensor_list(state_structure, gen_dataset_ops.reduce_dataset(dataset._variant_tensor, structure.to_tensor_list(state_structure, initial_state), reduce_func.captured_inputs, f=reduce_func, output_shapes=structure.get_flat_tensor_shapes(state_structure), output_types=structure.get_flat_tensor_types(state_structure), metadata=metadata.SerializeToString()))

    def get_single_element(self, name=None):
        if False:
            for i in range(10):
                print('nop')
        'Returns the single element of the `dataset`.\n\n    The function enables you to use a `tf.data.Dataset` in a stateless\n    "tensor-in tensor-out" expression, without creating an iterator.\n    This facilitates the ease of data transformation on tensors using the\n    optimized `tf.data.Dataset` abstraction on top of them.\n\n    For example, lets consider a `preprocessing_fn` which would take as an\n    input the raw features and returns the processed feature along with\n    it\'s label.\n\n    ```python\n    def preprocessing_fn(raw_feature):\n      # ... the raw_feature is preprocessed as per the use-case\n      return feature\n\n    raw_features = ...  # input batch of BATCH_SIZE elements.\n    dataset = (tf.data.Dataset.from_tensor_slices(raw_features)\n              .map(preprocessing_fn, num_parallel_calls=BATCH_SIZE)\n              .batch(BATCH_SIZE))\n\n    processed_features = dataset.get_single_element()\n    ```\n\n    In the above example, the `raw_features` tensor of length=BATCH_SIZE\n    was converted to a `tf.data.Dataset`. Next, each of the `raw_feature` was\n    mapped using the `preprocessing_fn` and the processed features were\n    grouped into a single batch. The final `dataset` contains only one element\n    which is a batch of all the processed features.\n\n    NOTE: The `dataset` should contain only one element.\n\n    Now, instead of creating an iterator for the `dataset` and retrieving the\n    batch of features, the `tf.data.get_single_element()` function is used\n    to skip the iterator creation process and directly output the batch of\n    features.\n\n    This can be particularly useful when your tensor transformations are\n    expressed as `tf.data.Dataset` operations, and you want to use those\n    transformations while serving your model.\n\n    #### Keras\n\n    ```python\n\n    model = ... # A pre-built or custom model\n\n    class PreprocessingModel(tf.keras.Model):\n      def __init__(self, model):\n        super().__init__(self)\n        self.model = model\n\n      @tf.function(input_signature=[...])\n      def serving_fn(self, data):\n        ds = tf.data.Dataset.from_tensor_slices(data)\n        ds = ds.map(preprocessing_fn, num_parallel_calls=BATCH_SIZE)\n        ds = ds.batch(batch_size=BATCH_SIZE)\n        return tf.argmax(self.model(ds.get_single_element()), axis=-1)\n\n    preprocessing_model = PreprocessingModel(model)\n    your_exported_model_dir = ... # save the model to this path.\n    tf.saved_model.save(preprocessing_model, your_exported_model_dir,\n                  signatures={\'serving_default\': preprocessing_model.serving_fn}\n                  )\n    ```\n\n    #### Estimator\n\n    In the case of estimators, you need to generally define a `serving_input_fn`\n    which would require the features to be processed by the model while\n    inferencing.\n\n    ```python\n    def serving_input_fn():\n\n      raw_feature_spec = ... # Spec for the raw_features\n      input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(\n          raw_feature_spec, default_batch_size=None)\n      )\n      serving_input_receiver = input_fn()\n      raw_features = serving_input_receiver.features\n\n      def preprocessing_fn(raw_feature):\n        # ... the raw_feature is preprocessed as per the use-case\n        return feature\n\n      dataset = (tf.data.Dataset.from_tensor_slices(raw_features)\n                .map(preprocessing_fn, num_parallel_calls=BATCH_SIZE)\n                .batch(BATCH_SIZE))\n\n      processed_features = dataset.get_single_element()\n\n      # Please note that the value of `BATCH_SIZE` should be equal to\n      # the size of the leading dimension of `raw_features`. This ensures\n      # that `dataset` has only element, which is a pre-requisite for\n      # using `dataset.get_single_element()`.\n\n      return tf.estimator.export.ServingInputReceiver(\n          processed_features, serving_input_receiver.receiver_tensors)\n\n    estimator = ... # A pre-built or custom estimator\n    estimator.export_saved_model(your_exported_model_dir, serving_input_fn)\n    ```\n\n    Args:\n      name: (Optional.) A name for the tf.data operation.\n\n    Returns:\n      A nested structure of `tf.Tensor` objects, corresponding to the single\n      element of `dataset`.\n\n    Raises:\n      InvalidArgumentError: (at runtime) if `dataset` does not contain exactly\n        one element.\n    '
        metadata = dataset_metadata_pb2.Metadata()
        if name:
            metadata.name = _validate_and_encode(name)
        return structure.from_compatible_tensor_list(self.element_spec, gen_dataset_ops.dataset_to_single_element(self._variant_tensor, metadata=metadata.SerializeToString(), **self._flat_structure))

    def unbatch(self, name=None) -> 'DatasetV2':
        if False:
            return 10
        'Splits elements of a dataset into multiple elements.\n\n    For example, if elements of the dataset are shaped `[B, a0, a1, ...]`,\n    where `B` may vary for each input element, then for each element in the\n    dataset, the unbatched dataset will contain `B` consecutive elements\n    of shape `[a0, a1, ...]`.\n\n    >>> elements = [ [1, 2, 3], [1, 2], [1, 2, 3, 4] ]\n    >>> dataset = tf.data.Dataset.from_generator(lambda: elements, tf.int64)\n    >>> dataset = dataset.unbatch()\n    >>> list(dataset.as_numpy_iterator())\n    [1, 2, 3, 1, 2, 1, 2, 3, 4]\n\n    Note: `unbatch` requires a data copy to slice up the batched tensor into\n    smaller, unbatched tensors. When optimizing performance, try to avoid\n    unnecessary usage of `unbatch`.\n\n    Args:\n      name: (Optional.) A name for the tf.data operation.\n\n    Returns:\n      A new `Dataset` with the transformation applied as described above.\n    '
        from tensorflow.python.data.ops import unbatch_op
        return unbatch_op._unbatch(self, name=name)

    def with_options(self, options, name=None) -> 'DatasetV2':
        if False:
            for i in range(10):
                print('nop')
        'Returns a new `tf.data.Dataset` with the given options set.\n\n    The options are "global" in the sense they apply to the entire dataset.\n    If options are set multiple times, they are merged as long as different\n    options do not use different non-default values.\n\n    >>> ds = tf.data.Dataset.range(5)\n    >>> ds = ds.interleave(lambda x: tf.data.Dataset.range(5),\n    ...                    cycle_length=3,\n    ...                    num_parallel_calls=3)\n    >>> options = tf.data.Options()\n    >>> # This will make the interleave order non-deterministic.\n    >>> options.deterministic = False\n    >>> ds = ds.with_options(options)\n\n    Args:\n      options: A `tf.data.Options` that identifies the options the use.\n      name: (Optional.) A name for the tf.data operation.\n\n    Returns:\n      A new `Dataset` with the transformation applied as described above.\n\n    Raises:\n      ValueError: when an option is set more than once to a non-default value\n    '
        return _OptionsDataset(self, options, name=name)

    def cardinality(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the cardinality of the dataset, if known.\n\n    `cardinality` may return `tf.data.INFINITE_CARDINALITY` if the dataset\n    contains an infinite number of elements or `tf.data.UNKNOWN_CARDINALITY` if\n    the analysis fails to determine the number of elements in the dataset\n    (e.g. when the dataset source is a file).\n\n    >>> dataset = tf.data.Dataset.range(42)\n    >>> print(dataset.cardinality().numpy())\n    42\n    >>> dataset = dataset.repeat()\n    >>> cardinality = dataset.cardinality()\n    >>> print((cardinality == tf.data.INFINITE_CARDINALITY).numpy())\n    True\n    >>> dataset = dataset.filter(lambda x: True)\n    >>> cardinality = dataset.cardinality()\n    >>> print((cardinality == tf.data.UNKNOWN_CARDINALITY).numpy())\n    True\n\n    Returns:\n      A scalar `tf.int64` `Tensor` representing the cardinality of the dataset.\n      If the cardinality is infinite or unknown, `cardinality` returns the\n      named constants `tf.data.INFINITE_CARDINALITY` and\n      `tf.data.UNKNOWN_CARDINALITY` respectively.\n    '
        return gen_dataset_ops.dataset_cardinality(self._variant_tensor)

    def group_by_window(self, key_func, reduce_func, window_size=None, window_size_func=None, name=None) -> 'DatasetV2':
        if False:
            print('Hello World!')
        'Groups windows of elements by key and reduces them.\n\n    This transformation maps each consecutive element in a dataset to a key\n    using `key_func` and groups the elements by key. It then applies\n    `reduce_func` to at most `window_size_func(key)` elements matching the same\n    key. All except the final window for each key will contain\n    `window_size_func(key)` elements; the final window may be smaller.\n\n    You may provide either a constant `window_size` or a window size determined\n    by the key through `window_size_func`.\n\n    >>> dataset = tf.data.Dataset.range(10)\n    >>> window_size = 5\n    >>> key_func = lambda x: x%2\n    >>> reduce_func = lambda key, dataset: dataset.batch(window_size)\n    >>> dataset = dataset.group_by_window(\n    ...           key_func=key_func,\n    ...           reduce_func=reduce_func,\n    ...           window_size=window_size)\n    >>> for elem in dataset.as_numpy_iterator():\n    ...   print(elem)\n    [0 2 4 6 8]\n    [1 3 5 7 9]\n\n    Args:\n      key_func: A function mapping a nested structure of tensors (having shapes\n        and types defined by `self.output_shapes` and `self.output_types`) to a\n        scalar `tf.int64` tensor.\n      reduce_func: A function mapping a key and a dataset of up to `window_size`\n        consecutive elements matching that key to another dataset.\n      window_size: A `tf.int64` scalar `tf.Tensor`, representing the number of\n        consecutive elements matching the same key to combine in a single batch,\n        which will be passed to `reduce_func`. Mutually exclusive with\n        `window_size_func`.\n      window_size_func: A function mapping a key to a `tf.int64` scalar\n        `tf.Tensor`, representing the number of consecutive elements matching\n        the same key to combine in a single batch, which will be passed to\n        `reduce_func`. Mutually exclusive with `window_size`.\n      name: (Optional.) A name for the tf.data operation.\n\n    Returns:\n      A new `Dataset` with the transformation applied as described above.\n\n    Raises:\n      ValueError: if neither or both of {`window_size`, `window_size_func`} are\n        passed.\n    '
        from tensorflow.python.data.ops import group_by_window_op
        return group_by_window_op._group_by_window(self, key_func, reduce_func, window_size, window_size_func, name=name)

    def bucket_by_sequence_length(self, element_length_func, bucket_boundaries, bucket_batch_sizes, padded_shapes=None, padding_values=None, pad_to_bucket_boundary=False, no_padding=False, drop_remainder=False, name=None) -> 'DatasetV2':
        if False:
            return 10
        'A transformation that buckets elements in a `Dataset` by length.\n\n    Elements of the `Dataset` are grouped together by length and then are padded\n    and batched.\n\n    This is useful for sequence tasks in which the elements have variable\n    length. Grouping together elements that have similar lengths reduces the\n    total fraction of padding in a batch which increases training step\n    efficiency.\n\n    Below is an example to bucketize the input data to the 3 buckets\n    "[0, 3), [3, 5), [5, inf)" based on sequence length, with batch size 2.\n\n    >>> elements = [\n    ...   [0], [1, 2, 3, 4], [5, 6, 7],\n    ...   [7, 8, 9, 10, 11], [13, 14, 15, 16, 19, 20], [21, 22]]\n    >>> dataset = tf.data.Dataset.from_generator(\n    ...     lambda: elements, tf.int64, output_shapes=[None])\n    >>> dataset = dataset.bucket_by_sequence_length(\n    ...         element_length_func=lambda elem: tf.shape(elem)[0],\n    ...         bucket_boundaries=[3, 5],\n    ...         bucket_batch_sizes=[2, 2, 2])\n    >>> for elem in dataset.as_numpy_iterator():\n    ...   print(elem)\n    [[1 2 3 4]\n    [5 6 7 0]]\n    [[ 7  8  9 10 11  0]\n    [13 14 15 16 19 20]]\n    [[ 0  0]\n    [21 22]]\n\n    Args:\n      element_length_func: function from element in `Dataset` to `tf.int32`,\n        determines the length of the element, which will determine the bucket it\n        goes into.\n      bucket_boundaries: `list<int>`, upper length boundaries of the buckets.\n      bucket_batch_sizes: `list<int>`, batch size per bucket. Length should be\n        `len(bucket_boundaries) + 1`.\n      padded_shapes: Nested structure of `tf.TensorShape` to pass to\n        `tf.data.Dataset.padded_batch`. If not provided, will use\n        `dataset.output_shapes`, which will result in variable length dimensions\n        being padded out to the maximum length in each batch.\n      padding_values: Values to pad with, passed to\n        `tf.data.Dataset.padded_batch`. Defaults to padding with 0.\n      pad_to_bucket_boundary: bool, if `False`, will pad dimensions with unknown\n        size to maximum length in batch. If `True`, will pad dimensions with\n        unknown size to bucket boundary minus 1 (i.e., the maximum length in\n        each bucket), and caller must ensure that the source `Dataset` does not\n        contain any elements with length longer than `max(bucket_boundaries)`.\n      no_padding: `bool`, indicates whether to pad the batch features (features\n        need to be either of type `tf.sparse.SparseTensor` or of same shape).\n      drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing\n        whether the last batch should be dropped in the case it has fewer than\n        `batch_size` elements; the default behavior is not to drop the smaller\n        batch.\n      name: (Optional.) A name for the tf.data operation.\n\n    Returns:\n      A new `Dataset` with the transformation applied as described above.\n\n    Raises:\n      ValueError: if `len(bucket_batch_sizes) != len(bucket_boundaries) + 1`.\n    '
        if len(bucket_batch_sizes) != len(bucket_boundaries) + 1:
            raise ValueError(f'`len(bucket_batch_sizes)` must equal `len(bucket_boundaries) + 1` but `len(bucket_batch_sizes)={len(bucket_batch_sizes)}` and `len(bucket_boundaries)={len(bucket_boundaries)}`.')
        batch_sizes = constant_op.constant(bucket_batch_sizes, dtype=dtypes.int64)

        def element_to_bucket_id(*args):
            if False:
                while True:
                    i = 10
            'Return int64 id of the length bucket for this element.'
            seq_length = element_length_func(*args)
            boundaries = list(bucket_boundaries)
            buckets_min = [np.iinfo(np.int32).min] + boundaries
            buckets_max = boundaries + [np.iinfo(np.int32).max]
            conditions_c = math_ops.logical_and(math_ops.less_equal(buckets_min, seq_length), math_ops.less(seq_length, buckets_max))
            bucket_id = math_ops.reduce_min(array_ops.where(conditions_c))
            return bucket_id

        def window_size_fn(bucket_id):
            if False:
                while True:
                    i = 10
            window_size = batch_sizes[bucket_id]
            return window_size

        def make_padded_shapes(shapes, none_filler=None):
            if False:
                print('Hello World!')
            padded = []
            for shape in nest.flatten(shapes):
                shape = tensor_shape.TensorShape(shape)
                shape = [none_filler if tensor_shape.dimension_value(d) is None else d for d in shape]
                padded.append(shape)
            return nest.pack_sequence_as(shapes, padded)

        def batching_fn(bucket_id, grouped_dataset):
            if False:
                return 10
            'Batch elements in dataset.'
            batch_size = window_size_fn(bucket_id)
            if no_padding:
                return grouped_dataset.batch(batch_size, drop_remainder=drop_remainder, name=name)
            none_filler = None
            if pad_to_bucket_boundary:
                err_msg = 'When pad_to_bucket_boundary=True, elements must have length < max(bucket_boundaries).'
                check = check_ops.assert_less(bucket_id, constant_op.constant(len(bucket_batch_sizes) - 1, dtype=dtypes.int64), message=err_msg)
                with ops.control_dependencies([check]):
                    boundaries = constant_op.constant(bucket_boundaries, dtype=dtypes.int64)
                    bucket_boundary = boundaries[bucket_id]
                    none_filler = bucket_boundary - 1
            input_shapes = get_legacy_output_shapes(grouped_dataset)
            shapes = make_padded_shapes(padded_shapes or input_shapes, none_filler=none_filler)
            return grouped_dataset.padded_batch(batch_size, shapes, padding_values, drop_remainder=drop_remainder, name=name)
        return self.group_by_window(key_func=element_to_bucket_id, reduce_func=batching_fn, window_size_func=window_size_fn, name=name)

    @staticmethod
    def random(seed=None, rerandomize_each_iteration=None, name=None) -> 'DatasetV2':
        if False:
            print('Hello World!')
        'Creates a `Dataset` of pseudorandom values.\n\n    The dataset generates a sequence of uniformly distributed integer values.\n\n    `rerandomize_each_iteration` controls whether the sequence of random number\n    generated should be re-randomized for each epoch. The default value is False\n    where the dataset generates the same sequence of random numbers for each\n    epoch.\n\n    >>> ds1 = tf.data.Dataset.random(seed=4).take(10)\n    >>> ds2 = tf.data.Dataset.random(seed=4).take(10)\n    >>> print(list(ds1.as_numpy_iterator())==list(ds2.as_numpy_iterator()))\n    True\n\n    >>> ds3 = tf.data.Dataset.random(seed=4).take(10)\n    >>> ds3_first_epoch = list(ds3.as_numpy_iterator())\n    >>> ds3_second_epoch = list(ds3.as_numpy_iterator())\n    >>> print(ds3_first_epoch == ds3_second_epoch)\n    True\n\n    >>> ds4 = tf.data.Dataset.random(\n    ...     seed=4, rerandomize_each_iteration=True).take(10)\n    >>> ds4_first_epoch = list(ds4.as_numpy_iterator())\n    >>> ds4_second_epoch = list(ds4.as_numpy_iterator())\n    >>> print(ds4_first_epoch == ds4_second_epoch)\n    False\n\n    Args:\n      seed: (Optional) If specified, the dataset produces a deterministic\n        sequence of values.\n      rerandomize_each_iteration: (Optional) If set to False, the dataset\n      generates the same sequence of random numbers for each epoch. If set to\n      True, it generates a different deterministic sequence of random numbers\n      for each epoch. It is defaulted to False if left unspecified.\n      name: (Optional.) A name for the tf.data operation.\n\n    Returns:\n      Dataset: A `Dataset`.\n    '
        from tensorflow.python.data.ops import random_op
        return random_op._random(seed=seed, rerandomize_each_iteration=rerandomize_each_iteration, name=name)

    def snapshot(self, path, compression='AUTO', reader_func=None, shard_func=None, name=None) -> 'DatasetV2':
        if False:
            return 10
        'API to persist the output of the input dataset.\n\n    The snapshot API allows users to transparently persist the output of their\n    preprocessing pipeline to disk, and materialize the pre-processed data on a\n    different training run.\n\n    This API enables repeated preprocessing steps to be consolidated, and allows\n    re-use of already processed data, trading off disk storage and network\n    bandwidth for freeing up more valuable CPU resources and accelerator compute\n    time.\n\n    https://github.com/tensorflow/community/blob/master/rfcs/20200107-tf-data-snapshot.md\n    has detailed design documentation of this feature.\n\n    Users can specify various options to control the behavior of snapshot,\n    including how snapshots are read from and written to by passing in\n    user-defined functions to the `reader_func` and `shard_func` parameters.\n\n    `shard_func` is a user specified function that maps input elements to\n    snapshot shards.\n\n    Users may want to specify this function to control how snapshot files should\n    be written to disk. Below is an example of how a potential `shard_func`\n    could be written.\n\n    ```python\n    dataset = ...\n    dataset = dataset.enumerate()\n    dataset = dataset.snapshot("/path/to/snapshot/dir",\n        shard_func=lambda x, y: x % NUM_SHARDS, ...)\n    dataset = dataset.map(lambda x, y: y)\n    ```\n\n    `reader_func` is a user specified function that accepts a single argument:\n    (1) a Dataset of Datasets, each representing a "split" of elements of the\n    original dataset. The cardinality of the input dataset matches the\n    number of the shards specified in the `shard_func` (see above). The function\n    should return a Dataset of elements of the original dataset.\n\n    Users may want specify this function to control how snapshot files should be\n    read from disk, including the amount of shuffling and parallelism.\n\n    Here is an example of a standard reader function a user can define. This\n    function enables both dataset shuffling and parallel reading of datasets:\n\n    ```python\n    def user_reader_func(datasets):\n      # shuffle the datasets splits\n      datasets = datasets.shuffle(NUM_CORES)\n      # read datasets in parallel and interleave their elements\n      return datasets.interleave(lambda x: x, num_parallel_calls=AUTOTUNE)\n\n    dataset = dataset.snapshot("/path/to/snapshot/dir",\n        reader_func=user_reader_func)\n    ```\n\n    By default, snapshot parallelizes reads by the number of cores available on\n    the system, but will not attempt to shuffle the data.\n\n    Args:\n      path: Required. A directory to use for storing / loading the snapshot to /\n        from.\n      compression: Optional. The type of compression to apply to the snapshot\n        written to disk. Supported options are `GZIP`, `SNAPPY`, `AUTO` or None.\n        Defaults to `AUTO`, which attempts to pick an appropriate compression\n        algorithm for the dataset.\n      reader_func: Optional. A function to control how to read data from\n        snapshot shards.\n      shard_func: Optional. A function to control how to shard data when writing\n        a snapshot.\n      name: (Optional.) A name for the tf.data operation.\n\n    Returns:\n      A new `Dataset` with the transformation applied as described above.\n    '
        from tensorflow.python.data.ops import snapshot_op
        return snapshot_op._snapshot(self, path, compression, reader_func, shard_func, name=name)

    def scan(self, initial_state, scan_func, name=None) -> 'DatasetV2':
        if False:
            i = 10
            return i + 15
        'A transformation that scans a function across an input dataset.\n\n    This transformation is a stateful relative of `tf.data.Dataset.map`.\n    In addition to mapping `scan_func` across the elements of the input dataset,\n    `scan()` accumulates one or more state tensors, whose initial values are\n    `initial_state`.\n\n    >>> dataset = tf.data.Dataset.range(10)\n    >>> initial_state = tf.constant(0, dtype=tf.int64)\n    >>> scan_func = lambda state, i: (state + i, state + i)\n    >>> dataset = dataset.scan(initial_state=initial_state, scan_func=scan_func)\n    >>> list(dataset.as_numpy_iterator())\n    [0, 1, 3, 6, 10, 15, 21, 28, 36, 45]\n\n    Args:\n      initial_state: A nested structure of tensors, representing the initial\n        state of the accumulator.\n      scan_func: A function that maps `(old_state, input_element)` to\n        `(new_state, output_element)`. It must take two arguments and return a\n        pair of nested structures of tensors. The `new_state` must match the\n        structure of `initial_state`.\n      name: (Optional.) A name for the tf.data operation.\n\n    Returns:\n      A new `Dataset` with the transformation applied as described above.\n    '
        from tensorflow.python.data.ops import scan_op
        return scan_op._scan(self, initial_state, scan_func, name=name)

    def take_while(self, predicate, name=None) -> 'DatasetV2':
        if False:
            print('Hello World!')
        'A transformation that stops dataset iteration based on a `predicate`.\n\n    >>> dataset = tf.data.Dataset.range(10)\n    >>> dataset = dataset.take_while(lambda x: x < 5)\n    >>> list(dataset.as_numpy_iterator())\n    [0, 1, 2, 3, 4]\n\n    Args:\n      predicate: A function that maps a nested structure of tensors (having\n        shapes and types defined by `self.output_shapes` and\n        `self.output_types`) to a scalar `tf.bool` tensor.\n      name: (Optional.) A name for the tf.data operation.\n\n    Returns:\n      A new `Dataset` with the transformation applied as described above.\n    '
        from tensorflow.python.data.ops import take_while_op
        return take_while_op._take_while(self, predicate, name=name)

    def unique(self, name=None) -> 'DatasetV2':
        if False:
            while True:
                i = 10
        'A transformation that discards duplicate elements of a `Dataset`.\n\n    Use this transformation to produce a dataset that contains one instance of\n    each unique element in the input. For example:\n\n    >>> dataset = tf.data.Dataset.from_tensor_slices([1, 37, 2, 37, 2, 1])\n    >>> dataset = dataset.unique()\n    >>> sorted(list(dataset.as_numpy_iterator()))\n    [1, 2, 37]\n\n    Note: This transformation only supports datasets which fit into memory\n    and have elements of either `tf.int32`, `tf.int64` or `tf.string` type.\n\n    Args:\n      name: (Optional.) A name for the tf.data operation.\n\n    Returns:\n      A new `Dataset` with the transformation applied as described above.\n    '
        from tensorflow.python.data.ops import unique_op
        return unique_op._unique(self, name)

    def rejection_resample(self, class_func, target_dist, initial_dist=None, seed=None, name=None) -> 'DatasetV2':
        if False:
            print('Hello World!')
        'Resamples elements to reach a target distribution.\n\n    Note: This implementation can reject **or repeat** elements in order to\n    reach the `target_dist`. So, in some cases, the output `Dataset` may be\n    larger than the input `Dataset`.\n\n    >>> initial_dist = [0.6, 0.4]\n    >>> n = 1000\n    >>> elems = np.random.choice(len(initial_dist), size=n, p=initial_dist)\n    >>> dataset = tf.data.Dataset.from_tensor_slices(elems)\n    >>> zero, one = np.bincount(list(dataset.as_numpy_iterator())) / n\n\n    Following from `initial_dist`, `zero` is ~0.6 and `one` is ~0.4.\n\n    >>> target_dist = [0.5, 0.5]\n    >>> dataset = dataset.rejection_resample(\n    ...    class_func=lambda x: x,\n    ...    target_dist=target_dist,\n    ...    initial_dist=initial_dist)\n    >>> dataset = dataset.map(lambda class_func_result, data: data)\n    >>> zero, one = np.bincount(list(dataset.as_numpy_iterator())) / n\n\n    Following from `target_dist`, `zero` is ~0.5 and `one` is ~0.5.\n\n    Args:\n      class_func: A function mapping an element of the input dataset to a scalar\n        `tf.int32` tensor. Values should be in `[0, num_classes)`.\n      target_dist: A floating point type tensor, shaped `[num_classes]`.\n      initial_dist: (Optional.)  A floating point type tensor, shaped\n        `[num_classes]`.  If not provided, the true class distribution is\n        estimated live in a streaming fashion.\n      seed: (Optional.) Python integer seed for the resampler.\n      name: (Optional.) A name for the tf.data operation.\n\n    Returns:\n      A new `Dataset` with the transformation applied as described above.\n    '
        target_dist_t = ops.convert_to_tensor(target_dist, name='target_dist')
        target_dist_t = math_ops.cast(target_dist_t, dtypes.float32)
        if initial_dist is not None:
            initial_dist_t = ops.convert_to_tensor(initial_dist, name='initial_dist')
            initial_dist_t = math_ops.cast(initial_dist_t, dtypes.float32)
            (acceptance_dist, prob_of_original) = _calculate_acceptance_probs_with_mixing(initial_dist_t, target_dist_t)
            initial_dist_ds = DatasetV2.from_tensors(initial_dist_t, name=name).repeat(name=name)
            acceptance_dist_ds = DatasetV2.from_tensors(acceptance_dist, name=name).repeat(name=name)
            prob_of_original_ds = DatasetV2.from_tensors(prob_of_original, name=name).repeat(name=name)
        else:
            initial_dist_ds = _estimate_initial_dist_ds(target_dist_t, self.map(class_func, name=name), name=name)
            acceptance_and_original_prob_ds = initial_dist_ds.map(lambda initial: _calculate_acceptance_probs_with_mixing(initial, target_dist_t), name=name)
            acceptance_dist_ds = acceptance_and_original_prob_ds.map(lambda accept_prob, _: accept_prob, name=name)
            prob_of_original_ds = acceptance_and_original_prob_ds.map(lambda _, prob_original: prob_original, name=name)
        filtered_ds = _filter_ds(self, acceptance_dist_ds, initial_dist_ds, class_func, seed)
        filtered_ds = filtered_ds.prefetch(3, name=name)
        prob_original_static = _get_prob_original_static(initial_dist_t, target_dist_t) if initial_dist is not None else None

        def add_class_value(*x):
            if False:
                print('Hello World!')
            if len(x) == 1:
                return (class_func(*x), x[0])
            else:
                return (class_func(*x), x)
        if prob_original_static == 1:
            return self.map(add_class_value, name=name)
        elif prob_original_static == 0:
            return filtered_ds
        else:
            return Dataset.sample_from_datasets([self.map(add_class_value), filtered_ds], weights=prob_of_original_ds.map(lambda prob: [(prob, 1.0 - prob)]), seed=seed, stop_on_empty_dataset=True)

    @staticmethod
    def sample_from_datasets(datasets, weights=None, seed=None, stop_on_empty_dataset=False, rerandomize_each_iteration=None) -> 'DatasetV2':
        if False:
            while True:
                i = 10
        'Samples elements at random from the datasets in `datasets`.\n\n    Creates a dataset by interleaving elements of `datasets` with `weight[i]`\n    probability of picking an element from dataset `i`. Sampling is done without\n    replacement. For example, suppose we have 2 datasets:\n\n    ```python\n    dataset1 = tf.data.Dataset.range(0, 3)\n    dataset2 = tf.data.Dataset.range(100, 103)\n    ```\n\n    Suppose that we sample from these 2 datasets with the following weights:\n\n    ```python\n    sample_dataset = tf.data.Dataset.sample_from_datasets(\n        [dataset1, dataset2], weights=[0.5, 0.5])\n    ```\n\n    One possible outcome of elements in sample_dataset is:\n\n    ```\n    print(list(sample_dataset.as_numpy_iterator()))\n    # [100, 0, 1, 101, 2, 102]\n    ```\n\n    Args:\n      datasets: A non-empty list of `tf.data.Dataset` objects with compatible\n        structure.\n      weights: (Optional.) A list or Tensor of `len(datasets)` floating-point\n        values where `weights[i]` represents the probability to sample from\n        `datasets[i]`, or a `tf.data.Dataset` object where each element is such\n        a list. Defaults to a uniform distribution across `datasets`.\n      seed: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the random\n        seed that will be used to create the distribution. See\n        `tf.random.set_seed` for behavior.\n      stop_on_empty_dataset: If `True`, sampling stops if it encounters an empty\n        dataset. If `False`, it continues sampling and skips any empty datasets.\n        It is recommended to set it to `True`. Otherwise, the distribution of\n        samples starts off as the user intends, but may change as input datasets\n        become empty. This can be difficult to detect since the dataset starts\n        off looking correct. Default to `False` for backward compatibility.\n      rerandomize_each_iteration: An optional `bool`. The boolean argument\n      controls whether the sequence of random numbers used to determine which\n      dataset to sample from will be rerandomized each epoch. That is, it\n      determinies whether datasets will be sampled in the same order across\n      different epochs (the default behavior) or not.\n\n    Returns:\n      A dataset that interleaves elements from `datasets` at random, according\n      to `weights` if provided, otherwise with uniform probability.\n\n    Raises:\n      TypeError: If the `datasets` or `weights` arguments have the wrong type.\n      ValueError:\n        - If `datasets` is empty, or\n        - If `weights` is specified and does not match the length of `datasets`.\n    '
        from tensorflow.python.data.ops import sample_from_datasets_op
        return sample_from_datasets_op._sample_from_datasets(datasets, weights, seed, stop_on_empty_dataset, rerandomize_each_iteration)

    @staticmethod
    def choose_from_datasets(datasets, choice_dataset, stop_on_empty_dataset=True) -> 'DatasetV2':
        if False:
            return 10
        'Creates a dataset that deterministically chooses elements from `datasets`.\n\n    For example, given the following datasets:\n\n    ```python\n    datasets = [tf.data.Dataset.from_tensors("foo").repeat(),\n                tf.data.Dataset.from_tensors("bar").repeat(),\n                tf.data.Dataset.from_tensors("baz").repeat()]\n\n    # Define a dataset containing `[0, 1, 2, 0, 1, 2, 0, 1, 2]`.\n    choice_dataset = tf.data.Dataset.range(3).repeat(3)\n\n    result = tf.data.Dataset.choose_from_datasets(datasets, choice_dataset)\n    ```\n\n    The elements of `result` will be:\n\n    ```\n    "foo", "bar", "baz", "foo", "bar", "baz", "foo", "bar", "baz"\n    ```\n\n    Args:\n      datasets: A non-empty list of `tf.data.Dataset` objects with compatible\n        structure.\n      choice_dataset: A `tf.data.Dataset` of scalar `tf.int64` tensors between\n        `0` and `len(datasets) - 1`.\n      stop_on_empty_dataset: If `True`, selection stops if it encounters an\n        empty dataset. If `False`, it skips empty datasets. It is recommended to\n        set it to `True`. Otherwise, the selected elements start off as the user\n        intends, but may change as input datasets become empty. This can be\n        difficult to detect since the dataset starts off looking correct.\n        Defaults to `True`.\n\n    Returns:\n      A new `Dataset` with the transformation applied as described above.\n\n    Raises:\n      TypeError: If `datasets` or `choice_dataset` has the wrong type.\n      ValueError: If `datasets` is empty.\n    '
        from tensorflow.python.data.ops import choose_from_datasets_op
        return choose_from_datasets_op._choose_from_datasets(datasets, choice_dataset, stop_on_empty_dataset)

@tf_export(v1=['data.Dataset'])
class DatasetV1(DatasetV2, data_types.DatasetV1):
    """Represents a potentially large set of elements.

  A `Dataset` can be used to represent an input pipeline as a
  collection of elements and a "logical plan" of transformations that act on
  those elements.
  """

    def __init__(self):
        if False:
            while True:
                i = 10
        try:
            variant_tensor = self._as_variant_tensor()
        except AttributeError as e:
            if '_as_variant_tensor' in str(e):
                raise AttributeError('Please use `_variant_tensor` instead of `_as_variant_tensor()` to obtain the variant associated with a dataset.')
            raise AttributeError('{}: A likely cause of this error is that the super call for this dataset is not the last line of the `__init__` method. The base class invokes the `_as_variant_tensor()` method in its constructor and if that method uses attributes defined in the `__init__` method, those attributes need to be defined before the super call.'.format(e))
        super(DatasetV1, self).__init__(variant_tensor)

    @abc.abstractmethod
    def _as_variant_tensor(self):
        if False:
            while True:
                i = 10
        'Creates a scalar `tf.Tensor` of `tf.variant` representing this dataset.\n\n    Returns:\n      A scalar `tf.Tensor` of `tf.variant` type, which represents this dataset.\n    '
        raise NotImplementedError(f'{type(self)}.as_variant_tensor()')

    @deprecation.deprecated(None, 'This is a deprecated API that should only be used in TF 1 graph mode and legacy TF 2 graph mode available through `tf.compat.v1`. In all other situations -- namely, eager mode and inside `tf.function` -- you can consume dataset elements using `for elem in dataset: ...` or by explicitly creating iterator via `iterator = iter(dataset)` and fetching its elements via `values = next(iterator)`. Furthermore, this API is not available in TF 2. During the transition from TF 1 to TF 2 you can use `tf.compat.v1.data.make_one_shot_iterator(dataset)` to create a TF 1 graph mode style iterator for a dataset created through TF 2 APIs. Note that this should be a transient state of your code base as there are in general no guarantees about the interoperability of TF 1 and TF 2 code.')
    def make_one_shot_iterator(self) -> Union[iterator_ops.Iterator, iterator_ops.OwnedIterator]:
        if False:
            return 10
        'Creates an iterator for elements of this dataset.\n\n    Note: The returned iterator will be initialized automatically.\n    A "one-shot" iterator does not currently support re-initialization. For\n    that see `make_initializable_iterator`.\n\n    Example:\n\n    ```python\n    # Building graph ...\n    dataset = ...\n    next_value = dataset.make_one_shot_iterator().get_next()\n\n    # ... from within a session ...\n    try:\n      while True:\n        value = sess.run(next_value)\n        ...\n    except tf.errors.OutOfRangeError:\n        pass\n    ```\n\n    Returns:\n      An `tf.data.Iterator` for elements of this dataset.\n    '
        return self._make_one_shot_iterator()

    def _make_one_shot_iterator(self) -> Union[iterator_ops.Iterator, iterator_ops.OwnedIterator]:
        if False:
            return 10
        if context.executing_eagerly():
            with ops.colocate_with(self._variant_tensor):
                return iterator_ops.OwnedIterator(self)
        _ensure_same_dataset_graph(self)
        allowlisted_stateful_ops = traverse.obtain_capture_by_value_ops(self)
        (graph_level_seed, op_level_seed) = core_random_seed.get_seed(None)

        @function.Defun(capture_by_value=True, allowlisted_stateful_ops=allowlisted_stateful_ops)
        def _make_dataset():
            if False:
                for i in range(10):
                    print('nop')
            'Factory function for a dataset.'
            if graph_level_seed is not None:
                assert op_level_seed is not None
                core_random_seed.set_random_seed((graph_level_seed + 87654321 * op_level_seed) % (2 ** 63 - 1))
            dataset = self._apply_debug_options()
            return dataset._variant_tensor
        try:
            _make_dataset.add_to_graph(ops.get_default_graph())
        except ValueError as err:
            if 'Cannot capture a stateful node' in str(err):
                raise ValueError('{}: A likely cause of this error is that the dataset for which you are calling `make_one_shot_iterator()` captures a stateful object, such as a `tf.Variable` or `tf.lookup.StaticHashTable`, which is not supported. Use `make_initializable_iterator()` instead.'.format(err)) from None
            else:
                raise
        with ops.colocate_with(self._variant_tensor):
            return iterator_ops.Iterator(gen_dataset_ops.one_shot_iterator(dataset_factory=_make_dataset, **self._flat_structure), None, get_legacy_output_types(self), get_legacy_output_shapes(self), get_legacy_output_classes(self))

    @deprecation.deprecated(None, 'This is a deprecated API that should only be used in TF 1 graph mode and legacy TF 2 graph mode available through `tf.compat.v1`. In all other situations -- namely, eager mode and inside `tf.function` -- you can consume dataset elements using `for elem in dataset: ...` or by explicitly creating iterator via `iterator = iter(dataset)` and fetching its elements via `values = next(iterator)`. Furthermore, this API is not available in TF 2. During the transition from TF 1 to TF 2 you can use `tf.compat.v1.data.make_initializable_iterator(dataset)` to create a TF 1 graph mode style iterator for a dataset created through TF 2 APIs. Note that this should be a transient state of your code base as there are in general no guarantees about the interoperability of TF 1 and TF 2 code.')
    def make_initializable_iterator(self, shared_name=None) -> iterator_ops.Iterator:
        if False:
            i = 10
            return i + 15
        'Creates an iterator for elements of this dataset.\n\n    Note: The returned iterator will be in an uninitialized state,\n    and you must run the `iterator.initializer` operation before using it:\n\n    ```python\n    # Building graph ...\n    dataset = ...\n    iterator = dataset.make_initializable_iterator()\n    next_value = iterator.get_next()  # This is a Tensor.\n\n    # ... from within a session ...\n    sess.run(iterator.initializer)\n    try:\n      while True:\n        value = sess.run(next_value)\n        ...\n    except tf.errors.OutOfRangeError:\n        pass\n    ```\n\n    Args:\n      shared_name: (Optional.) If non-empty, the returned iterator will be\n        shared under the given name across multiple sessions that share the same\n        devices (e.g. when using a remote server).\n\n    Returns:\n      A `tf.data.Iterator` for elements of this dataset.\n\n    Raises:\n      RuntimeError: If eager execution is enabled.\n    '
        return self._make_initializable_iterator(shared_name)

    def _make_initializable_iterator(self, shared_name=None) -> iterator_ops.Iterator:
        if False:
            return 10
        if context.executing_eagerly():
            raise RuntimeError('`make_initializable_iterator()` is not supported in eager mode. Use Python-style iteration instead.')
        _ensure_same_dataset_graph(self)
        dataset = self._apply_debug_options()
        if shared_name is None:
            shared_name = ''
        with ops.colocate_with(self._variant_tensor):
            iterator_resource = gen_dataset_ops.iterator_v2(container='', shared_name=shared_name, **self._flat_structure)
            initializer = gen_dataset_ops.make_iterator(dataset._variant_tensor, iterator_resource)
            return iterator_ops.Iterator(iterator_resource, initializer, get_legacy_output_types(dataset), get_legacy_output_shapes(dataset), get_legacy_output_classes(dataset))

    @property
    @deprecation.deprecated(None, 'Use `tf.compat.v1.data.get_output_classes(dataset)`.')
    def output_classes(self):
        if False:
            print('Hello World!')
        'Returns the class of each component of an element of this dataset.\n\n    Returns:\n      A (nested) structure of Python `type` objects corresponding to each\n      component of an element of this dataset.\n    '
        return nest.map_structure(lambda component_spec: component_spec._to_legacy_output_classes(), self.element_spec)

    @property
    @deprecation.deprecated(None, 'Use `tf.compat.v1.data.get_output_shapes(dataset)`.')
    def output_shapes(self):
        if False:
            return 10
        'Returns the shape of each component of an element of this dataset.\n\n    Returns:\n      A (nested) structure of `tf.TensorShape` objects corresponding to each\n      component of an element of this dataset.\n    '
        return nest.map_structure(lambda component_spec: component_spec._to_legacy_output_shapes(), self.element_spec)

    @property
    @deprecation.deprecated(None, 'Use `tf.compat.v1.data.get_output_types(dataset)`.')
    def output_types(self):
        if False:
            print('Hello World!')
        'Returns the type of each component of an element of this dataset.\n\n    Returns:\n      A (nested) structure of `tf.DType` objects corresponding to each component\n      of an element of this dataset.\n    '
        return nest.map_structure(lambda component_spec: component_spec._to_legacy_output_types(), self.element_spec)

    @property
    def element_spec(self):
        if False:
            for i in range(10):
                print('nop')
        return structure.convert_legacy_structure(self.output_types, self.output_shapes, self.output_classes)

    @staticmethod
    @functools.wraps(DatasetV2.from_tensors)
    def from_tensors(tensors, name=None):
        if False:
            print('Hello World!')
        return DatasetV1Adapter(DatasetV2.from_tensors(tensors, name=name))

    @staticmethod
    @functools.wraps(DatasetV2.from_tensor_slices)
    def from_tensor_slices(tensors, name=None):
        if False:
            i = 10
            return i + 15
        return DatasetV1Adapter(DatasetV2.from_tensor_slices(tensors, name=name))

    @staticmethod
    @deprecation.deprecated(None, 'Use `tf.data.Dataset.from_tensor_slices()`.')
    def from_sparse_tensor_slices(sparse_tensor):
        if False:
            return 10
        'Splits each rank-N `tf.sparse.SparseTensor` in this dataset row-wise.\n\n    Args:\n      sparse_tensor: A `tf.sparse.SparseTensor`.\n\n    Returns:\n      Dataset: A `Dataset` of rank-(N-1) sparse tensors.\n    '
        from tensorflow.python.data.ops import from_sparse_tensor_slices_op
        return from_sparse_tensor_slices_op._from_sparse_tensor_slices(sparse_tensor)

    @staticmethod
    @functools.wraps(DatasetV2.from_generator)
    @deprecation.deprecated_args(None, 'Use output_signature instead', 'output_types', 'output_shapes')
    def from_generator(generator, output_types=None, output_shapes=None, args=None, output_signature=None, name=None):
        if False:
            print('Hello World!')
        with deprecation.silence():
            return DatasetV1Adapter(DatasetV2.from_generator(generator, output_types, output_shapes, args, output_signature, name=name))

    @staticmethod
    @functools.wraps(DatasetV2.range)
    def range(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        return DatasetV1Adapter(DatasetV2.range(*args, **kwargs))

    @staticmethod
    @functools.wraps(DatasetV2.zip)
    def zip(*args, datasets=None, name=None):
        if False:
            while True:
                i = 10
        return DatasetV1Adapter(DatasetV2.zip(*args, datasets=datasets, name=name))

    @functools.wraps(DatasetV2.concatenate)
    def concatenate(self, dataset, name=None):
        if False:
            i = 10
            return i + 15
        return DatasetV1Adapter(super(DatasetV1, self).concatenate(dataset, name=name))

    @functools.wraps(DatasetV2.prefetch)
    def prefetch(self, buffer_size, name=None):
        if False:
            return 10
        return DatasetV1Adapter(super(DatasetV1, self).prefetch(buffer_size, name=name))

    @staticmethod
    @functools.wraps(DatasetV2.list_files)
    def list_files(file_pattern, shuffle=None, seed=None, name=None):
        if False:
            print('Hello World!')
        return DatasetV1Adapter(DatasetV2.list_files(file_pattern, shuffle, seed, name=name))

    @functools.wraps(DatasetV2.repeat)
    def repeat(self, count=None, name=None):
        if False:
            while True:
                i = 10
        return DatasetV1Adapter(super(DatasetV1, self).repeat(count, name=name))

    @functools.wraps(DatasetV2.shuffle)
    def shuffle(self, buffer_size, seed=None, reshuffle_each_iteration=None, name=None):
        if False:
            return 10
        return DatasetV1Adapter(super(DatasetV1, self).shuffle(buffer_size, seed, reshuffle_each_iteration, name=name))

    @functools.wraps(DatasetV2.cache)
    def cache(self, filename='', name=None):
        if False:
            for i in range(10):
                print('nop')
        return DatasetV1Adapter(super(DatasetV1, self).cache(filename, name=name))

    @functools.wraps(DatasetV2.take)
    def take(self, count, name=None):
        if False:
            return 10
        return DatasetV1Adapter(super(DatasetV1, self).take(count, name=name))

    @functools.wraps(DatasetV2.skip)
    def skip(self, count, name=None):
        if False:
            i = 10
            return i + 15
        return DatasetV1Adapter(super(DatasetV1, self).skip(count, name=name))

    @functools.wraps(DatasetV2.shard)
    def shard(self, num_shards, index, name=None):
        if False:
            i = 10
            return i + 15
        return DatasetV1Adapter(super(DatasetV1, self).shard(num_shards, index, name=name))

    @functools.wraps(DatasetV2.batch)
    def batch(self, batch_size, drop_remainder=False, num_parallel_calls=None, deterministic=None, name=None):
        if False:
            i = 10
            return i + 15
        return DatasetV1Adapter(super(DatasetV1, self).batch(batch_size, drop_remainder, num_parallel_calls, deterministic, name=name))

    @functools.wraps(DatasetV2.padded_batch)
    def padded_batch(self, batch_size, padded_shapes=None, padding_values=None, drop_remainder=False, name=None):
        if False:
            for i in range(10):
                print('nop')
        return DatasetV1Adapter(super(DatasetV1, self).padded_batch(batch_size, padded_shapes, padding_values, drop_remainder, name=name))

    @functools.wraps(DatasetV2.map)
    def map(self, map_func, num_parallel_calls=None, deterministic=None, name=None):
        if False:
            return 10
        from tensorflow.python.data.ops import map_op
        return map_op._map_v1(self, map_func, num_parallel_calls=num_parallel_calls, deterministic=deterministic)

    @deprecation.deprecated(None, 'Use `tf.data.Dataset.map()')
    def map_with_legacy_function(self, map_func, num_parallel_calls=None, deterministic=None) -> 'DatasetV1Adapter':
        if False:
            while True:
                i = 10
        'Maps `map_func` across the elements of this dataset.\n\n    Note: This is an escape hatch for existing uses of `map` that do not work\n    with V2 functions. New uses are strongly discouraged and existing uses\n    should migrate to `map` as this method will be removed in V2.\n\n    Args:\n      map_func: A function mapping a (nested) structure of tensors (having\n        shapes and types defined by `self.output_shapes` and\n        `self.output_types`) to another (nested) structure of tensors.\n      num_parallel_calls: (Optional.) A `tf.int32` scalar `tf.Tensor`,\n        representing the number elements to process asynchronously in parallel.\n        If not specified, elements will be processed sequentially. If the value\n        `tf.data.AUTOTUNE` is used, then the number of parallel calls is set\n        dynamically based on available CPU.\n      deterministic: (Optional.) When `num_parallel_calls` is specified, this\n        boolean controls the order in which the transformation produces\n        elements. If set to `False`, the transformation is allowed to yield\n        elements out of order to trade determinism for performance. If not\n        specified, the `tf.data.Options.deterministic` option (`True` by\n        default) controls the behavior.\n\n    Returns:\n      Dataset: A `Dataset`.\n    '
        from tensorflow.python.data.ops import map_op
        return map_op._map_v1_with_legacy_function(self, map_func, num_parallel_calls=num_parallel_calls, deterministic=deterministic)

    @functools.wraps(DatasetV2.flat_map)
    def flat_map(self, map_func, name=None) -> 'DatasetV1Adapter':
        if False:
            print('Hello World!')
        return DatasetV1Adapter(super(DatasetV1, self).flat_map(map_func, name=name))

    @functools.wraps(DatasetV2.interleave)
    def interleave(self, map_func, cycle_length=None, block_length=None, num_parallel_calls=None, deterministic=None, name=None) -> 'DatasetV1Adapter':
        if False:
            while True:
                i = 10
        return DatasetV1Adapter(super(DatasetV1, self).interleave(map_func, cycle_length, block_length, num_parallel_calls, deterministic, name=name))

    @functools.wraps(DatasetV2.filter)
    def filter(self, predicate, name=None) -> 'DatasetV1Adapter':
        if False:
            for i in range(10):
                print('nop')
        return DatasetV1Adapter(super(DatasetV1, self).filter(predicate, name=name))

    @deprecation.deprecated(None, 'Use `tf.data.Dataset.filter()')
    def filter_with_legacy_function(self, predicate) -> 'DatasetV2':
        if False:
            for i in range(10):
                print('nop')
        'Filters this dataset according to `predicate`.\n\n    Note: This is an escape hatch for existing uses of `filter` that do not work\n    with V2 functions. New uses are strongly discouraged and existing uses\n    should migrate to `filter` as this method will be removed in V2.\n\n    Args:\n      predicate: A function mapping a (nested) structure of tensors (having\n        shapes and types defined by `self.output_shapes` and\n        `self.output_types`) to a scalar `tf.bool` tensor.\n\n    Returns:\n      Dataset: The `Dataset` containing the elements of this dataset for which\n          `predicate` is `True`.\n    '
        from tensorflow.python.data.ops import filter_op
        return filter_op._FilterDataset(self, predicate, use_legacy_function=True)

    @functools.wraps(DatasetV2.apply)
    def apply(self, transformation_func) -> 'DatasetV1Adapter':
        if False:
            print('Hello World!')
        return DatasetV1Adapter(super(DatasetV1, self).apply(transformation_func))

    @functools.wraps(DatasetV2.window)
    def window(self, size, shift=None, stride=1, drop_remainder=False, name=None) -> 'DatasetV1Adapter':
        if False:
            i = 10
            return i + 15
        return DatasetV1Adapter(super(DatasetV1, self).window(size, shift, stride, drop_remainder, name=name))

    @functools.wraps(DatasetV2.unbatch)
    def unbatch(self, name=None) -> 'DatasetV1Adapter':
        if False:
            return 10
        return DatasetV1Adapter(super(DatasetV1, self).unbatch(name=name))

    @functools.wraps(DatasetV2.with_options)
    def with_options(self, options, name=None) -> 'DatasetV1Adapter':
        if False:
            for i in range(10):
                print('nop')
        return DatasetV1Adapter(super(DatasetV1, self).with_options(options, name=name))
if tf2.enabled():
    Dataset = DatasetV2
else:
    Dataset = DatasetV1

class DatasetV1Adapter(DatasetV1):
    """Wraps a V2 `Dataset` object in the `tf.compat.v1.data.Dataset` API."""

    def __init__(self, dataset: DatasetV2):
        if False:
            for i in range(10):
                print('nop')
        self._dataset = dataset
        super(DatasetV1Adapter, self).__init__()

    def _as_variant_tensor(self):
        if False:
            i = 10
            return i + 15
        return self._dataset._variant_tensor

    def _inputs(self):
        if False:
            while True:
                i = 10
        return self._dataset._inputs()

    def _functions(self) -> list[StructuredFunctionWrapper]:
        if False:
            return 10
        return self._dataset._functions()

    def options(self):
        if False:
            for i in range(10):
                print('nop')
        return self._dataset.options()

    @property
    def element_spec(self):
        if False:
            i = 10
            return i + 15
        return self._dataset.element_spec

    def __iter__(self):
        if False:
            while True:
                i = 10
        return iter(self._dataset)

def _ensure_same_dataset_graph(dataset):
    if False:
        while True:
            i = 10
    'Walks the dataset graph to ensure all datasets come from the same graph.'
    current_graph = ops.get_default_graph()
    bfs_q = queue.Queue()
    bfs_q.put(dataset)
    visited = []
    while not bfs_q.empty():
        ds = bfs_q.get()
        visited.append(ds)
        ds_graph = ds._graph
        if current_graph != ds_graph:
            raise ValueError(f'The graph {current_graph} of the iterator is different from the graph {ds_graph} the dataset: {ds._variant_tensor} was created in. If you are using the Estimator API, make sure that no part of the dataset returned by the `input_fn` function is defined outside the `input_fn` function. Otherwise, make sure that the dataset is created in the same graph as the iterator.')
        for input_ds in ds._inputs():
            if input_ds not in visited:
                bfs_q.put(input_ds)

@tf_export(v1=['data.make_one_shot_iterator'])
def make_one_shot_iterator(dataset: DatasetV1) -> Union[iterator_ops.Iterator, iterator_ops.OwnedIterator]:
    if False:
        i = 10
        return i + 15
    'Creates an iterator for elements of `dataset`.\n\n  Note: The returned iterator will be initialized automatically.\n  A "one-shot" iterator does not support re-initialization.\n\n  Args:\n    dataset: A `tf.data.Dataset`.\n\n  Returns:\n    A `tf.data.Iterator` for elements of `dataset`.\n\n  @compatibility(TF2)\n  This is a legacy API for consuming dataset elements and should only be used\n  during transition from TF 1 to TF 2. Note that using this API should be\n  a transient state of your code base as there are in general no guarantees\n  about the interoperability of TF 1 and TF 2 code.\n\n  In TF 2 datasets are Python iterables which means you can consume their\n  elements using `for elem in dataset: ...` or by explicitly creating iterator\n  via `iterator = iter(dataset)` and fetching its elements via\n  `values = next(iterator)`.\n  @end_compatibility\n  '
    try:
        return dataset._make_one_shot_iterator()
    except AttributeError:
        return DatasetV1Adapter(dataset)._make_one_shot_iterator()

@tf_export(v1=['data.make_initializable_iterator'])
def make_initializable_iterator(dataset: DatasetV1, shared_name=None) -> iterator_ops.Iterator:
    if False:
        i = 10
        return i + 15
    'Creates an iterator for elements of `dataset`.\n\n  Note: The returned iterator will be in an uninitialized state,\n  and you must run the `iterator.initializer` operation before using it:\n\n  ```python\n  dataset = ...\n  iterator = tf.compat.v1.data.make_initializable_iterator(dataset)\n  # ...\n  sess.run(iterator.initializer)\n  ```\n\n  Args:\n    dataset: A `tf.data.Dataset`.\n    shared_name: (Optional.) If non-empty, the returned iterator will be shared\n      under the given name across multiple sessions that share the same devices\n      (e.g. when using a remote server).\n\n  Returns:\n    A `tf.data.Iterator` for elements of `dataset`.\n\n  Raises:\n    RuntimeError: If eager execution is enabled.\n\n  @compatibility(TF2)\n  This is a legacy API for consuming dataset elements and should only be used\n  during transition from TF 1 to TF 2. Note that using this API should be\n  a transient state of your code base as there are in general no guarantees\n  about the interoperability of TF 1 and TF 2 code.\n\n  In TF 2 datasets are Python iterables which means you can consume their\n  elements using `for elem in dataset: ...` or by explicitly creating iterator\n  via `iterator = iter(dataset)` and fetching its elements via\n  `values = next(iterator)`.\n  @end_compatibility\n  '
    try:
        return dataset._make_initializable_iterator(shared_name)
    except AttributeError:
        return DatasetV1Adapter(dataset)._make_initializable_iterator(shared_name)

@tf_export('data.experimental.get_structure')
def get_structure(dataset_or_iterator):
    if False:
        return 10
    "Returns the type signature for elements of the input dataset / iterator.\n\n  For example, to get the structure of a `tf.data.Dataset`:\n\n  >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])\n  >>> tf.data.experimental.get_structure(dataset)\n  TensorSpec(shape=(), dtype=tf.int32, name=None)\n\n  >>> dataset = tf.data.experimental.from_list([(1, 'a'), (2, 'b'), (3, 'c')])\n  >>> tf.data.experimental.get_structure(dataset)\n  (TensorSpec(shape=(), dtype=tf.int32, name=None),\n   TensorSpec(shape=(), dtype=tf.string, name=None))\n\n  To get the structure of an `tf.data.Iterator`:\n\n  >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])\n  >>> tf.data.experimental.get_structure(iter(dataset))\n  TensorSpec(shape=(), dtype=tf.int32, name=None)\n\n  Args:\n    dataset_or_iterator: A `tf.data.Dataset` or an `tf.data.Iterator`.\n\n  Returns:\n    A (nested) structure of `tf.TypeSpec` objects matching the structure of an\n    element of `dataset_or_iterator` and specifying the type of individual\n    components.\n\n  Raises:\n    TypeError: If input is not a `tf.data.Dataset` or an `tf.data.Iterator`\n      object.\n  "
    try:
        return dataset_or_iterator.element_spec
    except AttributeError:
        raise TypeError(f'Invalid `dataset_or_iterator`. `dataset_or_iterator` must be a `tf.data.Dataset` or tf.data.Iterator object, but got {type(dataset_or_iterator)}.')

@tf_export(v1=['data.get_output_classes'])
def get_legacy_output_classes(dataset_or_iterator):
    if False:
        return 10
    'Returns the output classes for elements of the input dataset / iterator.\n\n  Args:\n    dataset_or_iterator: A `tf.data.Dataset` or `tf.data.Iterator`.\n\n  Returns:\n    A (nested) structure of Python `type` objects matching the structure of the\n    dataset / iterator elements and specifying the class of the individual\n    components.\n\n  @compatibility(TF2)\n  This is a legacy API for inspecting the type signature of dataset elements. In\n  TF 2, you should use the `tf.data.Dataset.element_spec` attribute instead.\n  @end_compatibility\n  '
    return nest.map_structure(lambda component_spec: component_spec._to_legacy_output_classes(), get_structure(dataset_or_iterator))

@tf_export(v1=['data.get_output_shapes'])
def get_legacy_output_shapes(dataset_or_iterator):
    if False:
        i = 10
        return i + 15
    'Returns the output shapes for elements of the input dataset / iterator.\n\n  Args:\n    dataset_or_iterator: A `tf.data.Dataset` or `tf.data.Iterator`.\n\n  Returns:\n    A (nested) structure of `tf.TensorShape` objects matching the structure of\n    the dataset / iterator elements and specifying the shape of the individual\n    components.\n\n  @compatibility(TF2)\n  This is a legacy API for inspecting the type signature of dataset elements. In\n  TF 2, you should use the `tf.data.Dataset.element_spec` attribute instead.\n  @end_compatibility\n  '
    return nest.map_structure(lambda component_spec: component_spec._to_legacy_output_shapes(), get_structure(dataset_or_iterator))

@tf_export(v1=['data.get_output_types'])
def get_legacy_output_types(dataset_or_iterator):
    if False:
        return 10
    'Returns the output shapes for elements of the input dataset / iterator.\n\n  Args:\n    dataset_or_iterator: A `tf.data.Dataset` or `tf.data.Iterator`.\n\n  Returns:\n    A (nested) structure of `tf.DType` objects matching the structure of\n    dataset / iterator elements and specifying the shape of the individual\n    components.\n\n  @compatibility(TF2)\n  This is a legacy API for inspecting the type signature of dataset elements. In\n  TF 2, you should use the `tf.data.Dataset.element_spec` attribute instead.\n  @end_compatibility\n  '
    return nest.map_structure(lambda component_spec: component_spec._to_legacy_output_types(), get_structure(dataset_or_iterator))

class DatasetSource(DatasetV2):
    """Abstract class representing a dataset with no inputs."""

    def _inputs(self):
        if False:
            print('Hello World!')
        return []

class UnaryDataset(DatasetV2):
    """Abstract class representing a dataset with one input."""

    def __init__(self, input_dataset: DatasetV2, variant_tensor):
        if False:
            i = 10
            return i + 15
        self._input_dataset = input_dataset
        super(UnaryDataset, self).__init__(variant_tensor)

    def _inputs(self):
        if False:
            while True:
                i = 10
        return [self._input_dataset]

class UnaryUnchangedStructureDataset(UnaryDataset):
    """Represents a unary dataset with the same input and output structure."""

    def __init__(self, input_dataset: DatasetV2, variant_tensor):
        if False:
            i = 10
            return i + 15
        self._input_dataset = input_dataset
        super(UnaryUnchangedStructureDataset, self).__init__(input_dataset, variant_tensor)

    @property
    def element_spec(self):
        if False:
            while True:
                i = 10
        return self._input_dataset.element_spec

class _VariantDataset(DatasetV2):
    """A Dataset wrapper around a `tf.variant`-typed function argument."""

    def __init__(self, dataset_variant, element_spec):
        if False:
            return 10
        self._element_spec = element_spec
        super(_VariantDataset, self).__init__(dataset_variant)

    def _inputs(self):
        if False:
            print('Hello World!')
        return []

    @property
    def element_spec(self):
        if False:
            i = 10
            return i + 15
        return self._element_spec

class _NestedVariant(composite_tensor.CompositeTensor):

    def __init__(self, variant_tensor, element_spec, dataset_shape):
        if False:
            return 10
        self._variant_tensor = variant_tensor
        self._element_spec = element_spec
        self._dataset_shape = dataset_shape

    @property
    def _type_spec(self):
        if False:
            print('Hello World!')
        return DatasetSpec(self._element_spec, self._dataset_shape)

@tf_export('data.experimental.from_variant')
def from_variant(variant, structure):
    if False:
        return 10
    'Constructs a dataset from the given variant and (nested) structure.\n\n  Args:\n    variant: A scalar `tf.variant` tensor representing a dataset.\n    structure: A (nested) structure of `tf.TypeSpec` objects representing the\n      structure of each element in the dataset.\n\n  Returns:\n    A `tf.data.Dataset` instance.\n  '
    return _VariantDataset(variant, structure)

@tf_export('data.experimental.to_variant')
def to_variant(dataset: DatasetV2):
    if False:
        i = 10
        return i + 15
    'Returns a variant representing the given dataset.\n\n  Args:\n    dataset: A `tf.data.Dataset`.\n\n  Returns:\n    A scalar `tf.variant` tensor representing the given dataset.\n  '
    return dataset._variant_tensor

@tf_export('data.DatasetSpec', v1=['data.DatasetSpec', 'data.experimental.DatasetStructure'])
class DatasetSpec(type_spec.BatchableTypeSpec):
    """Type specification for `tf.data.Dataset`.

  See `tf.TypeSpec` for more information about TensorFlow type specifications.

  >>> dataset = tf.data.Dataset.range(3)
  >>> tf.data.DatasetSpec.from_value(dataset)
  DatasetSpec(TensorSpec(shape=(), dtype=tf.int64, name=None), TensorShape([]))
  """
    __slots__ = ['_element_spec', '_dataset_shape']

    def __init__(self, element_spec, dataset_shape=()):
        if False:
            i = 10
            return i + 15
        self._element_spec = element_spec
        self._dataset_shape = tensor_shape.as_shape(dataset_shape)

    @property
    def value_type(self):
        if False:
            return 10
        return Dataset

    @property
    def element_spec(self):
        if False:
            print('Hello World!')
        'The inner element spec.'
        return self._element_spec

    def is_subtype_of(self, other):
        if False:
            for i in range(10):
                print('nop')
        'See base class.'
        if type(self) is not type(other):
            return False
        try:
            tf_nest.assert_same_structure(self.element_spec, other.element_spec)
        except (TypeError, ValueError):
            return False
        self_elements = tf_nest.flatten(self.element_spec)
        other_elements = tf_nest.flatten(other.element_spec)

        def is_subtype_or_equal(a, b):
            if False:
                return 10
            if isinstance(a, trace.TraceType):
                return a.is_subtype_of(b)
            else:
                return a == b
        for (self_element, other_element) in zip(self_elements, other_elements):
            if not is_subtype_or_equal(self_element, other_element):
                return False
        return self._dataset_shape.is_subtype_of(other._dataset_shape)

    def most_specific_common_supertype(self, others):
        if False:
            for i in range(10):
                print('nop')
        'See base class.'
        if not all((type(self) is type(other) for other in others)):
            return None
        try:
            for other in others:
                tf_nest.assert_same_structure(self.element_spec, other.element_spec)
        except (TypeError, ValueError):
            return None
        self_components = tf_nest.flatten(self.element_spec)
        others_components = [tf_nest.flatten(other.element_spec) for other in others]
        common_components = [None] * len(self_components)

        def common_supertype_or_equal(a, bs):
            if False:
                return 10
            if isinstance(a, trace.TraceType):
                return a.most_specific_common_supertype(bs)
            else:
                return a if all((a == b for b in bs)) else None
        for (i, self_component) in enumerate(self_components):
            common_components[i] = common_supertype_or_equal(self_component, [other_components[i] for other_components in others_components])
            if self_component is not None and common_components[i] is None:
                return None
        common_element_spec = tf_nest.pack_sequence_as(self._element_spec, common_components)
        common_dataset_shape = self._dataset_shape.most_specific_common_supertype([other._dataset_shape for other in others])
        if common_dataset_shape is None:
            return None
        return DatasetSpec(common_element_spec, common_dataset_shape)

    def _serialize(self):
        if False:
            for i in range(10):
                print('nop')
        return (self._element_spec, self._dataset_shape)

    @property
    def _component_specs(self):
        if False:
            for i in range(10):
                print('nop')
        return tensor_spec.TensorSpec(self._dataset_shape, dtypes.variant)

    def _to_components(self, value):
        if False:
            i = 10
            return i + 15
        return value._variant_tensor

    def _from_components(self, components):
        if False:
            i = 10
            return i + 15
        if self._dataset_shape.ndims == 0:
            return _VariantDataset(components, self._element_spec)
        else:
            return _NestedVariant(components, self._element_spec, self._dataset_shape)

    def _to_tensor_list(self, value):
        if False:
            i = 10
            return i + 15
        return [ops.convert_to_tensor(tf_nest.map_structure(lambda x: x._variant_tensor, value))]

    @staticmethod
    def from_value(value):
        if False:
            i = 10
            return i + 15
        'Creates a `DatasetSpec` for the given `tf.data.Dataset` value.'
        return DatasetSpec(value.element_spec)

    def _batch(self, batch_size):
        if False:
            while True:
                i = 10
        return DatasetSpec(self._element_spec, tensor_shape.TensorShape([batch_size]).concatenate(self._dataset_shape))

    def _unbatch(self):
        if False:
            print('Hello World!')
        if self._dataset_shape.ndims == 0:
            raise ValueError('Slicing dataset elements is not supported for rank 0.')
        return DatasetSpec(self._element_spec, self._dataset_shape[1:])

    def _to_batched_tensor_list(self, value):
        if False:
            return 10
        if self._dataset_shape.ndims == 0:
            raise ValueError('Slicing dataset elements is not supported for rank 0.')
        return self._to_tensor_list(value)

    def _to_legacy_output_types(self):
        if False:
            i = 10
            return i + 15
        return self

    def _to_legacy_output_shapes(self):
        if False:
            return 10
        return self

    def _to_legacy_output_classes(self):
        if False:
            i = 10
            return i + 15
        return self

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return hash(DatasetSpec)

    def __eq__(self, other):
        if False:
            return 10
        return isinstance(other, DatasetSpec) and self._element_spec == other._element_spec and (self._dataset_shape == other._dataset_shape)
nested_structure_coder.register_codec(nested_structure_coder.BuiltInTypeSpecCodec(DatasetSpec, struct_pb2.TypeSpecProto.DATA_DATASET_SPEC))

@tf_export('data.NumpyIterator')
class NumpyIterator(tracking_base.Trackable):
    """Iterator over a dataset with elements converted to numpy."""
    __slots__ = ['_iterator']

    def __init__(self, dataset):
        if False:
            print('Hello World!')
        self._iterator = iter(dataset)
        self._dataset = dataset

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def __next__(self):
        if False:
            i = 10
            return i + 15

        def to_numpy(x):
            if False:
                print('Hello World!')
            if hasattr(x, '_numpy'):
                numpy = x._numpy()
            else:
                numpy = x.numpy()
            if isinstance(numpy, np.ndarray):
                numpy.setflags(write=False)
            return numpy
        return nest.map_structure(to_numpy, next(self._iterator))

    def next(self):
        if False:
            i = 10
            return i + 15
        return self.__next__()

    def _serialize_to_tensors(self):
        if False:
            return 10
        return self._iterator._serialize_to_tensors()

    def _restore_from_tensors(self, restored_tensors):
        if False:
            for i in range(10):
                print('nop')
        return self._iterator._restore_from_tensors(restored_tensors)

    def _copy_trackable_to_cpu(self, object_map):
        if False:
            i = 10
            return i + 15
        if self not in object_map:
            object_map[self] = NumpyIterator(self._dataset)
        serialized = self._serialize_to_tensors()
        object_map[self]._restore_from_tensors(serialized)

    def _save(self):
        if False:
            for i in range(10):
                print('nop')
        return self.save()

    def save(self):
        if False:
            print('Hello World!')
        return self._iterator._save()

    def _restore(self, state):
        if False:
            return 10
        return self.restore(state)

    def restore(self, state):
        if False:
            while True:
                i = 10
        return self._iterator._restore(state)
_NumpyIterator = NumpyIterator

class _VariantTracker(resource_lib.CapturableResource):
    """Allows export of functions capturing a Dataset in SavedModels.

  When saving a SavedModel, `tf.saved_model.save` traverses the object
  graph. Since Datasets reference _VariantTracker objects, that traversal will
  find a _VariantTracker for each Dataset and so know how to save and restore
  functions which reference the Dataset's variant Tensor.
  """

    def __init__(self, variant_tensor, resource_creator):
        if False:
            while True:
                i = 10
        "Record that `variant_tensor` is associated with `resource_creator`.\n\n    Args:\n      variant_tensor: The variant-dtype Tensor associated with the Dataset. This\n        Tensor will be a captured input to functions which use the Dataset, and\n        is used by saving code to identify the corresponding _VariantTracker.\n      resource_creator: A zero-argument function which creates a new\n        variant-dtype Tensor. This function will be included in SavedModels and\n        run to re-create the Dataset's variant Tensor on restore.\n    "
        super(_VariantTracker, self).__init__(device='CPU')
        self._resource_handle = variant_tensor
        if not isinstance(resource_creator, def_function.Function):
            raise TypeError('Resource creator should already be a tf.function.')
        self._create_resource = resource_creator

    def _trackable_children(self, save_type=tracking_base.SaveType.CHECKPOINT, **kwargs):
        if False:
            i = 10
            return i + 15
        if save_type != tracking_base.SaveType.SAVEDMODEL:
            return {}
        children = super(_VariantTracker, self)._trackable_children(save_type, **kwargs)
        children['_create_resource'] = self._create_resource
        return children
batch_op = lazy_loader.LazyLoader('batch_op', globals(), 'tensorflow.python.data.ops.batch_op')
BatchDataset = batch_op._BatchDataset
PrefetchDataset = prefetch_op._PrefetchDataset
ShuffleDataset = shuffle_op._ShuffleDataset
repeat_op = lazy_loader.LazyLoader('repeat_op', globals(), 'tensorflow.python.data.ops.repeat_op')
RepeatDataset = repeat_op._RepeatDataset

class _OptionsDataset(UnaryUnchangedStructureDataset):
    """An identity `Dataset` that stores options."""

    def __init__(self, input_dataset, options, name=None):
        if False:
            for i in range(10):
                print('nop')
        self._input_dataset = input_dataset
        options_pb = dataset_options_pb2.Options()
        options_pb.CopyFrom(options._to_proto())
        self._name = name
        with ops.colocate_with(input_dataset._variant_tensor):
            variant_tensor = gen_dataset_ops.options_dataset(input_dataset._variant_tensor, options_pb.SerializeToString(), **self._common_args)
        super(_OptionsDataset, self).__init__(input_dataset, variant_tensor)
        if self._options_attr:
            self._options_attr._set_mutable(True)
            self._options_attr = self._options_attr.merge(options)
        else:
            self._options_attr = options
        self._options_attr._set_mutable(False)

def normalize_to_dense(dataset: Dataset):
    if False:
        i = 10
        return i + 15
    'Normalizes non-tensor components in a dataset to dense representations.\n\n  This is necessary for dataset transformations that slice along the batch\n  dimension and are oblivious to non-tensors, e.g. `unbatch`, `rebatch`.\n\n  Args:\n    dataset: Dataset to normalize.\n\n  Returns:\n    A dataset whose sparse and ragged tensors have been normalized to their\n    dense representations.\n  '
    if structured_function._should_unpack(dataset.element_spec):

        def normalize(*args):
            if False:
                while True:
                    i = 10
            return structure.to_batched_tensor_list(dataset.element_spec, tuple(args))
    else:

        def normalize(arg):
            if False:
                return 10
            return structure.to_batched_tensor_list(dataset.element_spec, arg)
    normalized_dataset = dataset.map(normalize)
    return _RestructuredDataset(normalized_dataset, dataset.element_spec)

class _RestructuredDataset(UnaryDataset):
    """An internal helper for changing the element spec of a dataset."""

    def __init__(self, dataset, element_spec):
        if False:
            while True:
                i = 10
        self._input_dataset = dataset
        self._element_spec = element_spec
        variant_tensor = self._input_dataset._variant_tensor
        super(_RestructuredDataset, self).__init__(dataset, variant_tensor)

    @property
    def element_spec(self):
        if False:
            return 10
        return self._element_spec

def _get_prob_original_static(initial_dist_t, target_dist_t):
    if False:
        while True:
            i = 10
    "Returns the static probability of sampling from the original.\n\n  `tensor_util.constant_value(prob_of_original)` returns `None` if it encounters\n  an Op that it isn't defined for. We have some custom logic to avoid this.\n\n  Args:\n    initial_dist_t: A tensor of the initial distribution.\n    target_dist_t: A tensor of the target distribution.\n\n  Returns:\n    The probability of sampling from the original distribution as a constant,\n    if it is a constant, or `None`.\n  "
    init_static = tensor_util.constant_value(initial_dist_t)
    target_static = tensor_util.constant_value(target_dist_t)
    if init_static is None or target_static is None:
        return None
    else:
        return np.min(target_static / init_static)

def _filter_ds(dataset, acceptance_dist_ds, initial_dist_ds, class_func, seed, name=None) -> DatasetV2:
    if False:
        print('Hello World!')
    'Filters a dataset based on per-class acceptance probabilities.\n\n  Args:\n    dataset: The dataset to be filtered.\n    acceptance_dist_ds: A dataset of acceptance probabilities.\n    initial_dist_ds: A dataset of the initial probability distribution, given or\n      estimated.\n    class_func: A function mapping an element of the input dataset to a scalar\n      `tf.int32` tensor. Values should be in `[0, num_classes)`.\n    seed: (Optional.) Python integer seed for the resampler.\n    name: (Optional.) A name for the tf.data operation.\n\n  Returns:\n    A dataset of (class value, data) after filtering.\n  '

    def maybe_warn_on_large_rejection(accept_dist, initial_dist):
        if False:
            i = 10
            return i + 15
        proportion_rejected = math_ops.reduce_sum((1 - accept_dist) * initial_dist)
        return cond.cond(math_ops.less(proportion_rejected, 0.5), lambda : accept_dist, lambda : logging_ops.Print(accept_dist, [proportion_rejected, initial_dist, accept_dist], message='Proportion of examples rejected by sampler is high: ', summarize=100, first_n=10))
    acceptance_dist_ds = DatasetV2.zip((acceptance_dist_ds, initial_dist_ds), name=name).map(maybe_warn_on_large_rejection, name=name)

    def _gather_and_copy(acceptance_prob, data):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(data, tuple):
            class_val = class_func(*data)
        else:
            class_val = class_func(data)
        return (class_val, array_ops.gather(acceptance_prob, class_val), data)
    current_probabilities_and_class_and_data_ds = DatasetV2.zip((acceptance_dist_ds, dataset), name=name).map(_gather_and_copy, name=name)

    def _reject(unused_class_val, p, unused_data):
        if False:
            for i in range(10):
                print('nop')
        return random_ops.random_uniform([], seed=seed, dtype=p.dtype) < p
    filtered_ds = current_probabilities_and_class_and_data_ds.filter(_reject, name=name)
    return filtered_ds.map(lambda class_value, _, data: (class_value, data), name=name)

def _estimate_initial_dist_ds(target_dist_t, class_values_ds, dist_estimation_batch_size=32, smoothing_constant=10, name=None):
    if False:
        while True:
            i = 10
    num_classes = target_dist_t.shape[0] or array_ops.shape(target_dist_t)[0]
    initial_examples_per_class_seen = array_ops.fill([num_classes], np.int64(smoothing_constant))

    def update_estimate_and_tile(num_examples_per_class_seen, c):
        if False:
            i = 10
            return i + 15
        (updated_examples_per_class_seen, dist) = _estimate_data_distribution(c, num_examples_per_class_seen)
        tiled_dist = array_ops.tile(array_ops.expand_dims(dist, 0), [dist_estimation_batch_size, 1])
        return (updated_examples_per_class_seen, tiled_dist)
    initial_dist_ds = class_values_ds.batch(dist_estimation_batch_size, name=name).scan(initial_examples_per_class_seen, update_estimate_and_tile, name=name).unbatch(name=name)
    return initial_dist_ds

def _get_target_to_initial_ratio(initial_probs, target_probs):
    if False:
        return 10
    denom = initial_probs + np.finfo(initial_probs.dtype.as_numpy_dtype).tiny
    return target_probs / denom

def _estimate_data_distribution(c, num_examples_per_class_seen):
    if False:
        print('Hello World!')
    'Estimate data distribution as labels are seen.\n\n  Args:\n    c: The class labels.  Type `int32`, shape `[batch_size]`.\n    num_examples_per_class_seen: Type `int64`, shape `[num_classes]`, containing\n      counts.\n\n  Returns:\n    num_examples_per_lass_seen: Updated counts.  Type `int64`, shape\n      `[num_classes]`.\n    dist: The updated distribution.  Type `float32`, shape `[num_classes]`.\n  '
    num_classes = num_examples_per_class_seen.get_shape()[0]
    num_examples_per_class_seen = math_ops.add(num_examples_per_class_seen, math_ops.reduce_sum(array_ops.one_hot(c, num_classes, dtype=dtypes.int64), 0))
    init_prob_estimate = math_ops.truediv(num_examples_per_class_seen, math_ops.reduce_sum(num_examples_per_class_seen))
    dist = math_ops.cast(init_prob_estimate, dtypes.float32)
    return (num_examples_per_class_seen, dist)

def _calculate_acceptance_probs_with_mixing(initial_probs, target_probs):
    if False:
        while True:
            i = 10
    'Calculates the acceptance probabilities and mixing ratio.\n\n  In this case, we assume that we can *either* sample from the original data\n  distribution with probability `m`, or sample from a reshaped distribution\n  that comes from rejection sampling on the original distribution. This\n  rejection sampling is done on a per-class basis, with `a_i` representing the\n  probability of accepting data from class `i`.\n\n  This method is based on solving the following analysis for the reshaped\n  distribution:\n\n  Let F be the probability of a rejection (on any example).\n  Let p_i be the proportion of examples in the data in class i (init_probs)\n  Let a_i is the rate the rejection sampler should *accept* class i\n  Let t_i is the target proportion in the minibatches for class i (target_probs)\n\n  ```\n  F = sum_i(p_i * (1-a_i))\n    = 1 - sum_i(p_i * a_i)     using sum_i(p_i) = 1\n  ```\n\n  An example with class `i` will be accepted if `k` rejections occur, then an\n  example with class `i` is seen by the rejector, and it is accepted. This can\n  be written as follows:\n\n  ```\n  t_i = sum_k=0^inf(F^k * p_i * a_i)\n      = p_i * a_j / (1 - F)    using geometric series identity, since 0 <= F < 1\n      = p_i * a_i / sum_j(p_j * a_j)        using F from above\n  ```\n\n  Note that the following constraints hold:\n  ```\n  0 <= p_i <= 1, sum_i(p_i) = 1\n  0 <= a_i <= 1\n  0 <= t_i <= 1, sum_i(t_i) = 1\n  ```\n\n  A solution for a_i in terms of the other variables is the following:\n    ```a_i = (t_i / p_i) / max_i[t_i / p_i]```\n\n  If we try to minimize the amount of data rejected, we get the following:\n\n  M_max = max_i [ t_i / p_i ]\n  M_min = min_i [ t_i / p_i ]\n\n  The desired probability of accepting data if it comes from class `i`:\n\n  a_i = (t_i/p_i - m) / (M_max - m)\n\n  The desired probability of pulling a data element from the original dataset,\n  rather than the filtered one:\n\n  m = M_min\n\n  Args:\n    initial_probs: A Tensor of the initial probability distribution, given or\n      estimated.\n    target_probs: A Tensor of the corresponding classes.\n\n  Returns:\n    (A 1D Tensor with the per-class acceptance probabilities, the desired\n    probability of pull from the original distribution.)\n  '
    ratio_l = _get_target_to_initial_ratio(initial_probs, target_probs)
    max_ratio = math_ops.reduce_max(ratio_l)
    min_ratio = math_ops.reduce_min(ratio_l)
    m = min_ratio
    a_i = (ratio_l - m) / (max_ratio - m)
    return (a_i, m)

def _apply_rewrite(dataset, rewrite):
    if False:
        i = 10
        return i + 15
    return _VariantDataset(gen_dataset_ops.rewrite_dataset(dataset._variant_tensor, rewrite, **dataset._flat_structure), dataset.element_spec)

def _collect_resource_inputs(op):
    if False:
        for i in range(10):
            print('nop')
    'Collects resource inputs for the given ops (and its variant inputs).'

    def _process(op_queue, seen_ops):
        if False:
            i = 10
            return i + 15
        'Processes the next element of the op queue.\n\n    Args:\n      op_queue: Queue of Dataset operations to process.\n      seen_ops: Already processed set of Operations.\n\n    Returns:\n      A 2-tuple containing sets of resource handles. The first tuple entry\n      contains read-only handles and the second entry contains read-write\n      handles.\n    '
        reads = []
        writes = []
        op = op_queue.pop()
        if op in seen_ops:
            return (reads, writes)
        seen_ops.add(op)
        (reads, writes) = acd_utils.get_read_write_resource_inputs(op)
        op_queue.extend((t.op for t in op.inputs if t.dtype == dtypes.variant))
        return (reads, writes)
    op_queue = [op]
    seen_ops = set()
    all_reads = []
    all_writes = []
    while op_queue:
        (reads, writes) = _process(op_queue, seen_ops)
        all_reads.extend(reads)
        all_writes.extend(writes)
    return (all_reads, all_writes)

@auto_control_deps.register_acd_resource_resolver
def _resource_resolver(op, resource_reads, resource_writes):
    if False:
        print('Hello World!')
    'Updates resource inputs for tf.data ops with indirect dependencies.'
    updated = False
    if op.type in ['DatasetToSingleElement', 'DatasetToTFRecord', 'ReduceDataset']:
        (reads, writes) = _collect_resource_inputs(op)
        for inp in reads:
            if inp not in resource_reads:
                updated = True
                resource_reads.add(inp)
        for inp in writes:
            if inp not in resource_writes:
                updated = True
                resource_writes.add(inp)
    if op.type in ['IteratorGetNext', 'IteratorGetNextSync', 'IteratorGetNextAsOptional']:
        iterator_resource = op.inputs[0]
        make_iterator_ops = [op for op in iterator_resource.consumers() if op.type == 'MakeIterator']
        if len(make_iterator_ops) == 1:
            (reads, writes) = _collect_resource_inputs(make_iterator_ops[0])
            for inp in reads:
                if inp not in resource_reads:
                    updated = True
                    resource_reads.add(inp)
            for inp in writes:
                if inp not in resource_writes:
                    updated = True
                    resource_writes.add(inp)
    return updated
dataset_autograph.register_overrides()