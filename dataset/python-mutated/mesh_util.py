"""Utilities to help with mesh creation."""
from typing import Dict, List, Optional, Tuple, Union
from absl import logging
import numpy as np
from tensorflow.dtensor.python import accelerator_util
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import layout
from tensorflow.dtensor.python import tpu_util
from tensorflow.python.eager import context
from tensorflow.python.framework import device as tf_device
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import tf_export

def _print_context(num_global_devices: int, num_clients: int, client_id: int, device_type: str, mesh: layout.Mesh) -> None:
    if False:
        return 10
    logging.info('This is client %d of %d clients', client_id, num_clients)
    logging.info('Number of global %s devices: %d', device_type.upper(), num_global_devices)
    logging.info('Global device IDs: %s', mesh.global_device_ids())
    logging.info('Local device IDs: %s', mesh.local_device_ids())
    logging.info('Local devices: %s', mesh.local_devices())

def _make_device_specs(devices: Optional[List[Union[tf_device.DeviceSpec, str]]]=None, device_type: Optional[str]=None) -> Tuple[List[tf_device.DeviceSpec], str]:
    if False:
        for i in range(10):
            print('nop')
    'Makes device specs for all local devices or from a provided list.'
    if devices is None:
        if device_type is None:
            device_type = 'CPU'
        devices = config.local_devices(device_type)
    else:
        if isinstance(devices[0], str):
            devices = [tf_device.DeviceSpec.from_string(d) for d in devices]
        if device_type is None:
            device_type = devices[0].device_type
        if device_type.upper() != devices[0].device_type.upper():
            raise ValueError(f'Conflicting devices {str(devices)} and device_type {device_type}')
    return (devices, device_type)

@tf_export('experimental.dtensor.create_mesh', v1=[])
def create_mesh(mesh_dims: Optional[Union[List[Tuple[str, int]], Dict[str, int]]]=None, mesh_name: str='', devices: Optional[List[Union[tf_device.DeviceSpec, str]]]=None, device_type: Optional[str]=None, use_xla_spmd: bool=layout.USE_XLA_SPMD) -> layout.Mesh:
    if False:
        while True:
            i = 10
    "Creates a single-client mesh.\n\n  If both `mesh_dims` and `devices` are specified, they must match each otehr.\n  As a special case, when all arguments are missing, this creates a 1D CPU mesh\n  with an empty name, assigning all available devices to that dimension.\n\n  Args:\n    mesh_dims: A dict of dim_name: dim_size, or a list of (dim_name, dim_size)\n      tuples. Defaults to a single batch-parallel dimension called 'x' usin all\n      devices. As a special case, a single-element mesh_dims whose dim_size is\n      -1 also uses all devices.  e.g. `{'x' : 4, 'y' : 1}` or `[('x', 4), ('y',\n      1)]`.\n    mesh_name: Name of the created mesh. Defaults to ''.\n    devices: String representations of devices to use. This is the device part\n      of tf.DeviceSpec, e.g. 'CPU:0'. Defaults to all available logical devices.\n    device_type: If `devices` is missing, the type of devices to use. Defaults\n      to 'CPU'.\n    use_xla_spmd: Boolean when True, will use XLA SPMD instead of DTensor SPMD.\n\n  Returns:\n    A single-client mesh created from specified or default arguments.\n  "
    (device_specs, device_type) = _make_device_specs(devices, device_type)
    local_spec = tf_device.DeviceSpec(job=config.job_name(), replica=0, task=0)
    device_specs = [local_spec.make_merged_spec(d) for d in device_specs]
    if isinstance(mesh_dims, dict):
        mesh_dims = list(mesh_dims.items())
    if mesh_dims is None:
        mesh_dims = [('x', len(device_specs))]
    elif len(mesh_dims) == 1 and mesh_dims[0][1] == -1:
        mesh_dims[0] = (mesh_dims[0][0], len(device_specs))
    dim_names = [d[0] for d in mesh_dims]
    shape = [d[1] for d in mesh_dims]
    if np.prod(shape) != len(device_specs):
        raise ValueError(f'length of devices ({len(device_specs)}) must be equal to total size of the mesh of shape {shape}')
    global_device_ids = np.arange(len(device_specs)).reshape(shape)
    local_device_ids = np.ravel(global_device_ids).tolist()
    mesh = layout.Mesh(dim_names=dim_names, global_device_ids=global_device_ids, local_device_ids=local_device_ids, local_devices=device_specs, mesh_name=mesh_name, use_xla_spmd=use_xla_spmd)
    _print_context(num_global_devices=len(device_specs), num_clients=1, client_id=0, device_type=device_type, mesh=mesh)
    return mesh

@tf_export('experimental.dtensor.create_distributed_mesh', v1=[])
def create_distributed_mesh(mesh_dims: Union[List[Tuple[str, int]], Dict[str, int]], mesh_name: str='', local_devices: Optional[List[Union[tf_device.DeviceSpec, str]]]=None, device_type: Optional[str]=None, use_xla_spmd: bool=layout.USE_XLA_SPMD) -> layout.Mesh:
    if False:
        print('Hello World!')
    "Creates a distributed mesh.\n\n  This is similar to `create_mesh`, but with a different set of arguments to\n  create a mesh that spans evenly across a multi-client DTensor cluster.\n\n  For CPU and GPU meshes, users can choose to use fewer local devices than what\n  is available `local_devices`.\n\n  For TPU, only meshes that uses all TPU cores is supported by the DTensor\n  runtime.\n\n  Args:\n    mesh_dims: A dict of dim_name: dim_size, or a list of (dim_name, dim_size)\n      tuples. e.g. `{'x' : 4, 'y' : 1}` or `[('x', 4), ('y', 1)]`.\n    mesh_name: Name of the created mesh. Defaults to ''.\n    local_devices: String representations of devices to use. This is the device\n      part of tf.DeviceSpec, e.g. 'CPU:0'. Defaults to all available local\n      logical devices.\n    device_type: Type of device to build the mesh for. Defaults to 'CPU'.\n      Supported values are 'CPU', 'GPU', 'TPU'.6\n    use_xla_spmd: Boolean when True, will use XLA SPMD instead of DTensor SPMD.\n\n  Returns:\n    A mesh that spans evenly across all DTensor clients in the cluster.\n  "
    if isinstance(mesh_dims, dict):
        mesh_dims = list(mesh_dims.items())
    (dim_names, shape) = zip(*mesh_dims)
    if not accelerator_util.is_initialized():
        raise ValueError('Accelerators are uninitialized, please run dtensor.initialize_accelerator_system() first.')
    if device_type and device_type.upper() == 'TPU':
        if local_devices is not None:
            raise ValueError(f'Do not specify devices for {device_type.upper()} meshes. Using a partial list of devices for {device_type.upper()} is not supported.')
    (device_specs, device_type) = _make_device_specs(local_devices, device_type)
    if device_type.upper() in ['CPU', 'GPU']:
        local_spec = tf_device.DeviceSpec(job=config.job_name(), replica=0, task=config.client_id())
        device_specs = [local_spec.make_merged_spec(d) for d in device_specs]
        num_global_devices = len(device_specs) * config.num_clients()
        if np.prod(shape) != num_global_devices:
            raise ValueError(f'Global number of devices ({len(device_specs)} per client * {config.num_clients()} clients = {num_global_devices}) must be equal to total size of the mesh of shape {shape}')
        global_device_ids = np.arange(num_global_devices).reshape(shape)
        flattened = np.ravel(global_device_ids).tolist()
        start_idx = len(device_specs) * config.client_id()
        local_device_ids = flattened[start_idx:start_idx + len(device_specs)]
        mesh = layout.Mesh(dim_names=dim_names, global_device_ids=global_device_ids, local_device_ids=local_device_ids, local_devices=device_specs, mesh_name=mesh_name, use_xla_spmd=use_xla_spmd)
        _print_context(num_global_devices, config.num_clients(), config.client_id(), device_type, mesh)
        return mesh
    if device_type.upper() == 'TPU':
        mesh = tpu_util.create_tpu_mesh(mesh_dim_names=dim_names, mesh_shape=shape, mesh_name=mesh_name, use_xla_spmd=use_xla_spmd)
        _print_context(config.num_global_devices(device_type), config.num_clients(), config.client_id(), device_type, mesh)
        return mesh
    raise ValueError(f'Device type {device_type} is not CPU, GPU or TPU')
_BARRIER_DICT = {}

@tf_export('experimental.dtensor.barrier', v1=[])
def barrier(mesh: layout.Mesh, barrier_name: Optional[str]=None, timeout_in_ms: Optional[int]=None):
    if False:
        i = 10
        return i + 15
    "Runs a barrier on the mesh.\n\n  Upon returning from the barrier, all operations run before the barrier\n  would have completed across all clients. Currently we allocate a fully\n  sharded tensor with mesh shape and run an all_reduce on it.\n\n  Example:\n\n  A barrier can be used before application exit to ensure completion of pending\n  ops.\n\n  ```python\n\n  x = [1, 2, 3]\n  x = dtensor.relayout(x, dtensor.Layout.batch_sharded(mesh, 'batch', 1))\n  dtensor.barrier(mesh)\n\n  # At this point all devices on all clients in the mesh have completed\n  # operations before the barrier. Therefore it is OK to tear down the clients.\n  sys.exit()\n  ```\n\n  Args:\n    mesh: The mesh to run the barrier on.\n    barrier_name: The name of the barrier. Mainly used for logging purpose.\n    timeout_in_ms: The timeout of the barrier in ms. If omitted, blocks\n      indefinitely till the barrier is reached from all clients.\n  "
    if barrier_name is None:
        barrier_name = '(barrier)'
    logging.info('entering barrier before op: %s', barrier_name)
    context.async_wait()
    component = array_ops.reshape(1.0, [1] * len(mesh.shape()))
    ones = api.pack([component] * mesh.num_local_devices(), layout.Layout(mesh.dim_names, mesh))
    mesh_size = math_ops.reduce_sum(ones)
    if mesh_size != mesh.size:
        raise ValueError('Global barrier produced wrong mesh size : {0} while mesh has actualsize : {1}'.format(mesh_size, mesh.size))
    context.async_wait()
    if context.context().coordination_service:
        if timeout_in_ms is None:
            timeout_in_ms = 24 * 60 * 60 * 1000
        num_calls = _BARRIER_DICT.setdefault(barrier_name, 0)
        _BARRIER_DICT[barrier_name] = num_calls + 1
        barrier_id = f'{barrier_name}:{num_calls}'
        context.context().wait_at_barrier(barrier_id, timeout_in_ms)
    logging.info('finished running barrier across all clients after op: %s', barrier_name)