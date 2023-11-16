"""Unified high level distribution APIs across backends.

!!!DO NOT USE!!! Currently under development and APIs are not final.

Currently only the JAX backend has been implemented. The TensorFlow backend
will be implemented in the future (via tf.dtensor API).
"""
import collections
import contextlib
import os
import re
import warnings
import numpy as np
from keras.api_export import keras_export
from keras.backend import KerasTensor
from keras.backend import distribution_lib
from keras.backend.common import global_state
DEFAULT_BATCH_DIM_NAME = 'batch'
GLOBAL_ATTRIBUTE_NAME = 'distribution'

@keras_export('keras.distribution.list_devices')
def list_devices(device_type=None):
    if False:
        return 10
    'Return all the available devices based on the device type.\n\n    Note: in a distributed setting, global devices are returned.\n\n    Args:\n        device_type: string, one of `"cpu"`, `"gpu"` or `"tpu"`.\n            Defaults to `"gpu"` or `"tpu"` if available when\n            `device_type` is not provided. Otherwise\n            will return the `"cpu"` devices.\n\n    Return:\n        List of devices that are available for distribute computation.\n    '
    return distribution_lib.list_devices(device_type)

@keras_export('keras.distribution.initialize')
def initialize(job_addresses=None, num_processes=None, proceed_id=None):
    if False:
        while True:
            i = 10
    'Initialize the distribution system for multi-host/process setting.\n\n    Calling `initialize` will prepare the backend for execution on multi-host\n    GPU or TPUs. It should be called before any computations.\n\n    Note that the parameters can also be injected via enviornment variables,\n    which can be better controlled by the launch script at startup time.\n    For certain backend that also rely on the enviornment variables to\n    configure, Keras will properly forward them.\n\n    Args:\n        job_addresses: string. Comma separated IP addresses for all the jobs\n            that will form the whole computation cluster. Note that for JAX\n            backend, only the address for job 0 (coodinator) is needed. For\n            certain runtime like cloud TPU, this value can be `None`, and the\n            backend will figure it out with the TPU enviornment variables. You\n            can also config this value via enviornment variable\n            `KERAS_DISTRIBUTION_JOB_ADDRESSES`.\n        num_processes: int. The number of worker/processes that will form the\n            whole computation cluster. For certain runtime like cloud TPU, this\n            value can be `None`, and the backend will figure it out with the TPU\n            enviornment variables. You can also configure this value via\n            enviornment variable `KERAS_DISTRIBUTION_NUM_PROCESSES`.\n        process_id: int. The ID number of the current worker/process. The value\n            should be ranged from `0` to `num_processes - 1`. `0` will indicate\n            the current worker/process is the master/coordinate job. You can\n            also configure this value via enviornment variable\n            `KERAS_DISTRIBUTION_PROCESS_ID`.\n\n        Example:\n            Suppose there are two GPU processes, and process 0 is running at\n            address `10.0.0.1:1234`, and process 1 is running at address\n            `10.0.0.2:2345`. To configure such cluster, you can run\n\n        On process 0:\n        ```python\n        keras.distribute.initialize(\n            job_addresses="10.0.0.1:1234,10.0.0.2:2345",\n            num_processes=2,\n            process_id=0)\n        ```\n\n        On process 1:\n        ```python\n        keras.distribute.initialize(\n            job_addresses="10.0.0.1:1234,10.0.0.2:2345",\n            num_processes=2,\n            process_id=1)\n        ```\n\n        or via the enviornment variables:\n        On process 0:\n        ```python\n        os.environ[\n            "KERAS_DISTRIBUTION_JOB_ADDRESSES"] = "10.0.0.1:1234,10.0.0.2:2345"\n        os.environ["KERAS_DISTRIBUTION_NUM_PROCESSES"] = "2\n        os.environ["KERAS_DISTRIBUTION_PROCESS_ID"] = "0"\n        keras.distribute.initialize()\n        ```\n\n        On process 1:\n        ```python\n        os.environ[\n            "KERAS_DISTRIBUTION_JOB_ADDRESSES"] = "10.0.0.1:1234,10.0.0.2:2345"\n        os.environ["KERAS_DISTRIBUTION_NUM_PROCESSES"] = "2\n        os.environ["KERAS_DISTRIBUTION_PROCESS_ID"] = "1"\n        keras.distribute.initialize()\n        ```\n\n        Also note that for JAX backend, the `job_addresses` can be further\n        reduced to just the master/coordinator address, which is\n        `10.0.0.1:1234`.\n    '
    if job_addresses is None and 'KERAS_DISTRIBUTION_JOB_ADDRESSES' in os.environ:
        job_addresses = os.environ['KERAS_DISTRIBUTION_JOB_ADDRESSES']
    if num_processes is None and 'KERAS_DISTRIBUTION_NUM_PROCESSES' in os.environ:
        num_processes = int(os.environ['KERAS_DISTRIBUTION_NUM_PROCESSES'])
    if proceed_id is None and 'KERAS_DISTRIBUTION_PROCESS_ID' in os.environ:
        proceed_id = int(os.environ['KERAS_DISTRIBUTION_PROCESS_ID'])
    distribution_lib.initialize(job_addresses, num_processes, proceed_id)

@keras_export('keras.distribution.DeviceMesh')
class DeviceMesh:
    """A cluster of computation devices for distributed computation.

    This API is aligned with `jax.sharding.Mesh` and `tf.dtensor.Mesh`, which
    represents the computation devices in the global context.

    See more details in [jax.sharding.Mesh](
        https://jax.readthedocs.io/en/latest/jax.sharding.html#jax.sharding.Mesh)
    and [tf.dtensor.Mesh](
        https://www.tensorflow.org/api_docs/python/tf/experimental/dtensor/Mesh).

    Args:
        shape: tuple of list of integers. The shape of the overall
            `DeviceMesh`, e.g. `(8,)` for a data parallel only distribution,
            or `(4, 2)` for a model+data parallel distribution.
        axis_names: List of string. The logical name of the each axis for
            the `DeviceMesh`. The length of the `axis_names` should match to
            the rank of the `shape`. The `axis_names` will be used to
            match/create the `TensorLayout` when distribute the data and
            variables.
        devices: Optional list of devices. Defaults to all the available
            devices locally from `keras.distribution.list_devices()`.
    """

    def __init__(self, shape, axis_names, devices=None):
        if False:
            while True:
                i = 10
        if not shape or not axis_names:
            raise ValueError(f'Shape and axis_names cannot be empty. Received: shape={shape}, axis_names={axis_names}')
        if len(shape) != len(axis_names):
            raise ValueError(f'Shape and axis_names should have same size. Received: shape={shape}, axis_names={axis_names}')
        if devices is None:
            devices = list_devices()
        devices = np.array(devices)
        if np.prod(shape) != np.prod(devices.shape):
            raise ValueError(f'Shape does not match the number of devices. Received: shape={shape}; devices.shape={devices.shape}')
        self._shape = shape
        self._axis_names = axis_names
        self._devices = np.reshape(devices, shape)

    @property
    def shape(self):
        if False:
            while True:
                i = 10
        return self._shape

    @property
    def axis_names(self):
        if False:
            while True:
                i = 10
        return self._axis_names

    @property
    def devices(self):
        if False:
            for i in range(10):
                print('nop')
        return self._devices

@keras_export('keras.distribution.TensorLayout')
class TensorLayout:
    """A layout to apply to a tensor.

    This API is aligned with `jax.sharding.NamedSharding`
    and `tf.dtensor.Layout`.

    See more details in [jax.sharding.NamedSharding](
        https://jax.readthedocs.io/en/latest/jax.sharding.html#jax.sharding.NamedSharding)
    and [tf.dtensor.Layout](
        https://www.tensorflow.org/api_docs/python/tf/experimental/dtensor/Layout).

    Args:
        axes: tuple of strings that should map to the `axis_names` in
            a `DeviceMesh`. For any dimentions that doesn't need any sharding,
            A `None` can be used a placeholder.
        device_mesh: Optional `DeviceMesh` that will be used to create
            the layout. The actual mapping of tensor to physical device
            is not known until the mesh is specified.
    """

    def __init__(self, axes, device_mesh=None):
        if False:
            return 10
        self._axes = tuple(axes)
        self._device_mesh = device_mesh
        self._validate_axes()

    @property
    def axes(self):
        if False:
            i = 10
            return i + 15
        return self._axes

    @property
    def device_mesh(self):
        if False:
            return 10
        return self._device_mesh

    @device_mesh.setter
    def device_mesh(self, device_mesh):
        if False:
            return 10
        if self._device_mesh is not None:
            raise ValueError(f'Cannot override device mesh value. Existing value is {self._device_mesh}')
        self._device_mesh = device_mesh
        self._validate_axes()

    def _validate_axes(self):
        if False:
            print('Hello World!')
        if self._device_mesh:
            valid_axis_names = set(self._device_mesh.axis_names)
            axis_names = set(self._axes) - set([None])
            if axis_names - valid_axis_names:
                raise ValueError(f'Invalid axis names for Layout. Valid axis names: {valid_axis_names}, Got {axis_names}')

class Distribution:
    """Base class for variable distribution strategies.

    A `Distribution` has following key functionalities:

    1. Distribute the model variables to a `DeviceMesh`.
    2. Distribute the input data to a `DeviceMesh`.
    3. Distribute an intermediate state tensor in the model.

    It can create a context scope so that the framework to properly detect the
    `Distribution` and distribute the variable/data accordingly.

    Args:
        device_mesh: A `DeviceMesh` instance.
    """

    def __init__(self, device_mesh):
        if False:
            i = 10
            return i + 15
        self._device_mesh = device_mesh

    def get_data_layout(self, data_shape):
        if False:
            for i in range(10):
                print('nop')
        'Retrieve the `TensorLayout` for the input data.\n\n        Args:\n            data_shape: shape for the input data in list or tuple format.\n\n        Returns:\n            The `TensorLayout` for the data, which can be used by\n            `backend.distribute_value()` to redistribute a input data.\n        '
        raise NotImplementedError()

    def get_variable_layout(self, variable):
        if False:
            while True:
                i = 10
        'Retrieve the `TensorLayout` for the variable.\n\n        Args:\n            variable: A `KerasVariable` instance.\n\n        return:\n            The `TensorLayout` for the variable, which can be used by\n            `backend.distribute_value()` to redistribute a variable.\n        '
        raise NotImplementedError()

    def get_tensor_layout(self, path):
        if False:
            return 10
        'Retrieve the `TensorLayout` for the intermediate tensor.\n\n        Args:\n            path: a string path for the correspoding tensor.\n\n        return:\n            The `TensorLayout` for the intermediate tensor, which can be used\n            by `backend.relayout()` to reshard the tensor. Could also return\n            None.\n        '
        raise NotImplementedError()

    @contextlib.contextmanager
    def scope(self):
        if False:
            return 10
        'Context manager to make the `Distribution` current.'
        original_scope = distribution()
        set_distribution(self)
        try:
            yield
        finally:
            set_distribution(original_scope)

    @property
    def device_mesh(self):
        if False:
            for i in range(10):
                print('nop')
        return self._device_mesh

    def distribute_dataset(self, dataset):
        if False:
            for i in range(10):
                print('nop')
        'Create a distributed dataset instance from the original user dataset.\n\n        Args:\n            dataset: the original global dataset instance. Only\n            `tf.data.Dataset` is supported at the moment.\n\n        Returns:\n            a sharded `tf.data.Dataset` instance, which will produce data for\n            the current local worker/process.\n        '
        raise NotImplementedError()

@keras_export('keras.distribution.DataParallel')
class DataParallel(Distribution):
    """Distribution for data parallelism.

    You can choose to create this instance by either specifing
    the `device_mesh` or `devices` arguments (but not both).

    The `device_mesh` argument is expected to be a `DeviceMesh` instance,
    and is expected to be 1D only. In case that the mesh has multiple axes,
    then the first axis will be treated as the data parallel dimension
    (and a warning will be raised).

    When a list of `devices` are provided, they will be used to construct a
    1D mesh.

    When both `mesh` and `devices` are absent, then `list_devices()`
    will be used to detect any available devices and create a 1D mesh from
    them.

    Args:
        device_mesh: Optional `DeviceMesh` instance.
        devices: Optional list of devices.
    """

    def __init__(self, device_mesh=None, devices=None):
        if False:
            print('Hello World!')
        if device_mesh:
            self._initialize_with_device_mesh(device_mesh)
        elif devices:
            self._initialize_mesh_from_devices(devices)
        else:
            self._initialize_mesh_from_list_devices()
        self._batch_dim_name = self.device_mesh.axis_names[0]
        self._num_process = distribution_lib.num_processes()
        self._process_id = distribution_lib.process_id()
        self._is_multi_process = self._num_process > 1

    def _initialize_with_device_mesh(self, device_mesh):
        if False:
            return 10
        if not isinstance(device_mesh, DeviceMesh):
            raise ValueError(f'Expect `mesh` to be an instance of `DeviceMesh`. Received: mesh={device_mesh} (of type {type(device_mesh)})')
        super().__init__(device_mesh)
        if self.device_mesh.devices.ndim != 1:
            warnings.warn('Expect the input mesh to be 1D, but received mesh.devices.ndim=%d. The first axis will be used for data-parallel sharding.', device_mesh.devices.ndim)

    def _initialize_mesh_from_devices(self, devices):
        if False:
            return 10
        devices = np.array(devices)
        device_mesh = DeviceMesh(shape=devices.shape, axis_names=[DEFAULT_BATCH_DIM_NAME], devices=devices)
        super().__init__(device_mesh)

    def _initialize_mesh_from_list_devices(self):
        if False:
            for i in range(10):
                print('nop')
        devices = np.array(list_devices())
        device_mesh = DeviceMesh(shape=devices.shape, axis_names=[DEFAULT_BATCH_DIM_NAME], devices=devices)
        super().__init__(device_mesh)

    def get_data_layout(self, data_shape):
        if False:
            while True:
                i = 10
        data_shard_spec = [None] * len(data_shape)
        data_shard_spec[0] = self._batch_dim_name
        return TensorLayout(data_shard_spec, self.device_mesh)

    def get_variable_layout(self, variable):
        if False:
            i = 10
            return i + 15
        variable_shard_spec = [None] * len(variable.shape)
        return TensorLayout(variable_shard_spec, self.device_mesh)

    def get_tensor_layout(self, path):
        if False:
            for i in range(10):
                print('nop')
        return None

    def distribute_dataset(self, dataset):
        if False:
            return 10
        from tensorflow.python.data.experimental.ops import distribute as tf_data_distribute
        from keras.utils.module_utils import tensorflow as tf
        if not isinstance(dataset, tf.data.Dataset):
            raise ValueError(f'Only `tf.data.Dataset` is supported for sharding, got {type(dataset)}')
        if not self._is_multi_process:
            return dataset
        batch_size = tf_data_distribute.compute_batch_size(dataset)
        if batch_size.numpy() < 0:
            raise ValueError('The batch size of the input dataset is unknown. Please config the batch size for the input dataset, e.g via `dataset.batch(batch_size)`')
        per_worker_batch_size = tf_data_distribute.batch_sizes_for_worker(global_batch_size=batch_size, num_workers=self._num_process, num_replicas_per_worker=1, worker_index=self._process_id)
        distributed_dataset = dataset.rebatch(per_worker_batch_size)
        distributed_dataset = tf_data_distribute._AutoShardDataset(distributed_dataset, num_workers=self._num_process, index=self._process_id, num_replicas=self._num_process)
        return distributed_dataset.prefetch(tf.data.AUTOTUNE)

@keras_export('keras.distribution.ModelParallel')
class ModelParallel(Distribution):
    """Distribution that shards model variables.

    Compare to `DataParallel` which replicates the variables across all devices,
    `ModelParallel` allows you to shard variables in addition to the input data.

    To construct a `ModelParallel` distribution, you need to provide a
    `DeviceMesh` and a `LayoutMap`.

    1. `DeviceMesh` contains physcial device information. The axis names in
        the mesh will be used to map the variable and data layout.
    2. `LayoutMap` contains the mapping between variable paths to their
        corresponding `TensorLayout`.

    Example:

    ```python
    devices = list_devices()    # Assume there are 8 devices.

    # Create a mesh with 2 devices for data parallelism and 4 devices for
    # model parallelism.
    device_mesh = DeviceMesh(shape=(2, 4), axis_names=('batch', 'model'),
                             devices=devices)
    # Create a layout map that shard the `Dense` layer and `Conv2D`
    # layer variables on the last dimension.
    # Based on the `device_mesh`, this means the variables
    # will be split across 4 devices. Any other variable that doesn't
    # match any key in the layout map will be fully replicated.
    layout_map = LayoutMap(device_mesh)
    layout_map['dense.*kernel'] = (None, 'model')
    layout_map['dense.*bias'] = ('model',)
    layout_map['conv2d.*kernel'] = (None, None, None, 'model')
    layout_map['conv2d.*bias'] = ('model',)

    distribution = ModelParallel(device_mesh=device_mesh,
                                 layout_map=layout_map,
                                 batch_dim_name='batch')
    # Set the global distribution, or via `with distribution.scope():`
    set_distribution(distribution)

    model = model_creation()
    model.compile()
    model.fit(data)
    ```

    You can quickly update the device mesh shape to change the sharding factor
    of the variables. E.g.
    ```
    # With only the shape change for the device mesh, the variables will be
    # sharded across 8 devices instead of 4, which further reduces the memory
    # footprint of variables on each of the device.
    device_mesh = DeviceMesh(shape=(1, 8), axis_names=('batch', 'model'),
                             devices=devices)
    ```

    To figure out a proper layout mapping rule for all the model variables, you
    can first list out all the model variable paths, which will be used as the
    key to map the variables to `TensorLayout`.

    e.g.
    ```
    model = create_model()
    for v in model.variables:
        print(v.path)
    ```

    Args:
        device_mesh: `DeviceMesh` instance for physical device and its
            logical mapping.
        layout_map: `LayoutMap` instance which map the variable path to the
            corresponding `TensorLayout`. The axis names of the
            `TensorLayout`s should match to the axis names in the
            device_mesh, or exception will be raised.
        batch_dim_name: optional string, the axis name in the `device_mesh`
            that will be used to distribute data. If unspecified, the
            first axis from the `device_mesh` will be used.
    """

    def __init__(self, device_mesh, layout_map, batch_dim_name=None):
        if False:
            i = 10
            return i + 15
        super().__init__(device_mesh)
        self._layout_map = layout_map
        self._batch_dim_name = batch_dim_name or self.device_mesh.axis_names[0]
        self._num_process = distribution_lib.num_processes()
        self._process_id = distribution_lib.process_id()
        self._is_multi_process = self._num_process > 1

    def get_data_layout(self, data_shape):
        if False:
            while True:
                i = 10
        data_shard_spec = [None] * len(data_shape)
        data_shard_spec[0] = self._batch_dim_name
        return TensorLayout(data_shard_spec, self.device_mesh)

    def get_variable_layout(self, variable):
        if False:
            print('Hello World!')
        variable_layout = self._layout_map[variable.path]
        if variable_layout is not None:
            return variable_layout
        variable_shard_spec = [None] * len(variable.shape)
        return TensorLayout(variable_shard_spec, self.device_mesh)

    def get_tensor_layout(self, path):
        if False:
            i = 10
            return i + 15
        return self._layout_map[path]

    def distribute_dataset(self, dataset):
        if False:
            return 10
        from tensorflow.python.data.experimental.ops import distribute as tf_data_distribute
        from keras.utils.module_utils import tensorflow as tf
        if not isinstance(dataset, tf.data.Dataset):
            raise ValueError(f'Only `tf.data.Dataset` is supported for sharding, got {type(dataset)}')
        if not self._is_multi_process:
            return dataset
        global_batch_size = tf_data_distribute.compute_batch_size(dataset)
        if global_batch_size.numpy() < 0:
            raise ValueError('The batch size of the input dataset is unknown. Please config the batch size for the input dataset, e.g via `dataset.batch(batch_size)`')
        mesh_batch_dim_index = self.device_mesh.axis_names.index(self._batch_dim_name)
        mesh_batch_dim_size = self.device_mesh.shape[mesh_batch_dim_index]
        local_device_count = np.prod(self.device_mesh.shape) // self._num_process
        if mesh_batch_dim_size < local_device_count:
            return dataset.prefetch(tf.data.AUTOTUNE)
        else:
            if mesh_batch_dim_size % local_device_count != 0:
                raise ValueError(f'The Batch dimention of the mesh is not compatible with the local worker device count. Mesh batch dim = {mesh_batch_dim_size} and local device count = {local_device_count}')
            num_shards = mesh_batch_dim_size // local_device_count
            per_worker_batch_size = global_batch_size // num_shards
            distributed_dataset = dataset.rebatch(per_worker_batch_size)
            distributed_dataset = tf_data_distribute._AutoShardDataset(distributed_dataset, num_workers=num_shards, index=self._process_id % num_shards, num_replicas=num_shards)
            return distributed_dataset.prefetch(tf.data.AUTOTUNE)

@keras_export('keras.distribution.LayoutMap')
class LayoutMap(collections.abc.MutableMapping):
    """A dict-like object that maps string to `TensorLayout` instances.

    `LayoutMap` uses a string as key and a `TensorLayout` as value. There is a
    behavior difference between a normal Python dict and this class. The string
    key will be treated as a regex when retrieving the value. See the docstring
    of `get` for more details.

    See below for a usage example. You can define the naming schema
    of the `TensorLayout`, and then retrieve the corresponding
    `TensorLayout` instance.

    In the normal case, the key to query is usually the `variable.path`, which
    is the idenifier of the variable.

    As shortcut, tuple or list of axis names are also allowed when inserting
    as value, and will be converted to `TensorLayout`.

    ```python
    layout_map = LayoutMap(device_mesh=None)
    layout_map['dense.*kernel'] = (None, 'model')         # layout_2d
    layout_map['dense.*bias'] = ('model',)                # layout_1d
    layout_map['conv2d.*kernel'] = TensorLayout((None, None, None, 'model'))
    layout_map['conv2d.*bias'] = TensorLayout(('model',))  # layout_1d

    layout_1 = layout_map['dense_1.kernel']             # layout_1 == layout_2d
    layout_2 = layout_map['dense_1.bias']               # layout_2 == layout_1d
    layout_3 = layout_map['dense_2.kernel']             # layout_3 == layout_2d
    layout_4 = layout_map['dense_2.bias']               # layout_4 == layout_1d
    layout_5 = layout_map['my_model/conv2d_123/kernel'] # layout_5 == layout_4d
    layout_6 = layout_map['my_model/conv2d_123/bias']   # layout_6 == layout_1d
    layout_7 = layout_map['my_model/conv3d_1/kernel']   # layout_7 == None
    layout_8 = layout_map['my_model/conv3d_1/bias']     # layout_8 == None
    ```

    Args:
        device_mesh: An optional `DeviceMesh` that can be used to populate the
            `TensorLayout.device_mesh` if `TensorLayout.device_mesh` is not set.
    """

    def __init__(self, device_mesh=None):
        if False:
            for i in range(10):
                print('nop')
        self._layout_map = collections.OrderedDict()
        self._device_mesh = device_mesh

    def __getitem__(self, key):
        if False:
            for i in range(10):
                print('nop')
        "Retrieves the corresponding layout by the string key.\n\n        When there isn't an exact match, all the existing keys in the layout map\n        will be treated as a regex and map against the input key again. When\n        there are multiple matches for the regex, an `ValueError` will be\n        raised. Returns `None` if there isn't any match found.\n\n        Args:\n            key: String key to query a layout.\n\n        Returns:\n            Corresponding layout based on the query.\n        "
        if key in self._layout_map:
            return self._layout_map[key]
        matching_keys = []
        for k in self._layout_map:
            if re.search(k, key):
                matching_keys.append(k)
        if len(matching_keys) > 1:
            raise ValueError(f"Path '{key}' matches multiple layout specification keys: {matching_keys}. Please make sure each tensor/variable path only matches at most one layout specification key in the LayoutMap.")
        elif len(matching_keys) == 1:
            return self._layout_map[matching_keys[0]]
        return None

    def __setitem__(self, key, layout):
        if False:
            i = 10
            return i + 15
        'Insert TensorLayout to the LayoutMap.\n\n        Args:\n            key: String key for the `TensorLayout`.\n            layout: The `TensorLayout`. As a shortcut, tuple of string and None\n                are also acceptable, and will be converted to `TensorLayout`.\n        '
        if key in self._layout_map:
            raise ValueError(f'{key} already exist in the LayoutMap with value {self._layout_map[key]}. Please make sure to not use duplicated keys.')
        if isinstance(layout, tuple):
            layout = TensorLayout(axes=layout, device_mesh=None)
        if not isinstance(layout, TensorLayout):
            raise ValueError(f'{layout} should be a TensorLayout type, got {type(layout)}')
        self._maybe_populate_device_mesh(layout)
        self._layout_map[key] = layout

    def __delitem__(self, key):
        if False:
            print('Hello World!')
        return self._layout_map.pop(key)

    def __len__(self):
        if False:
            return 10
        return len(self._layout_map)

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return iter(self._layout_map)

    @property
    def device_mesh(self):
        if False:
            while True:
                i = 10
        return self._device_mesh

    def _maybe_populate_device_mesh(self, layout):
        if False:
            while True:
                i = 10
        if layout.device_mesh is None and self.device_mesh is not None:
            layout.device_mesh = self.device_mesh
LayoutMap.get.__doc__ = LayoutMap.__getitem__.__doc__

@keras_export('keras.distribution.distribute_tensor')
def distribute_tensor(tensor, layout):
    if False:
        print('Hello World!')
    'Change the layout of a Tensor value in the jit function execution.\n\n    Args:\n        tensor: a Tensor to change the layout.\n        layout: `TensorLayout` to be applied on the value.\n\n    Returns:\n        a new value with the specified tensor layout.\n    '
    if isinstance(tensor, KerasTensor):
        return tensor
    return distribution_lib.distribute_tensor(tensor, layout)

@keras_export('keras.distribution.distribution')
def distribution():
    if False:
        print('Hello World!')
    'Retrieve the current distribution from global context.'
    return global_state.get_global_attribute(GLOBAL_ATTRIBUTE_NAME)

@keras_export('keras.distribution.set_distribution')
def set_distribution(value):
    if False:
        for i in range(10):
            print('nop')
    'Set the distribution as the global distribution setting.\n\n    Args:\n        value: a `Distribution` instance.\n    '
    global_state.set_global_attribute(GLOBAL_ATTRIBUTE_NAME, value)