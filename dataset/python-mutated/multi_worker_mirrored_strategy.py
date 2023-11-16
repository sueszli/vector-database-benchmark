"""Implement a MultiMirroredStrategy based on the DTensor low level API.

This is an experiment to validate the viability of the DTensor API, and expose
any potential feature gaps between the current API and the need.
"""
import os
from tensorflow.dtensor.python import config as d_config
from tensorflow.dtensor.python import mesh_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute.cluster_resolver import tfconfig_cluster_resolver
from tensorflow.python.distribute.experimental import dtensor_strategy_extended
from tensorflow.python.distribute.experimental import dtensor_util

class MultiWorkerMirroredStrategy(distribute_lib.Strategy):
    """A distribution strategy for synchronous training on multiple workers.

  This strategy implements synchronous distributed training across multiple
  workers, each with potentially multiple GPUs. Similar to
  `tf.distribute.MirroredStrategy`, it replicates all variables and computations
  to each local device. The difference is that it uses a distributed collective
  implementation (e.g. all-reduce), so that multiple workers can work together.
  """

    def __init__(self, cluster_resolver=None, communication_options=None, *, mesh=None):
        if False:
            i = 10
            return i + 15
        'Creates the strategy.\n\n    Args:\n      cluster_resolver: optional\n        `tf.distribute.cluster_resolver.ClusterResolver`. In case neither `mesh`\n        nor `cluster_resolver` are provided,\n        `tf.distribute.cluster_resolver.TFConfigClusterResolver` is used.\n      communication_options: currently ignore.\n      mesh: optional Dtensor global mesh for the computation. Note that either\n        `mesh` or the `cluster_resolver` should be provided. and not both.\n    '
        self._validate_init_args(mesh, cluster_resolver)
        if not mesh:
            if not cluster_resolver:
                cluster_resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()
            dtensor_env_var = _parse_dtensor_env_var_from_cluster_resolver(cluster_resolver)
            _config_dtensor_env_var(dtensor_env_var)
            mesh = _build_distributed_mesh(dtensor_util.DEFAULT_BATCH_MESH_DIM_NAME)
        extended = dtensor_strategy_extended.DTensorStrategyExtended(container_strategy=self, mesh=mesh)
        super().__init__(extended)
        self._mesh = mesh
        self._cluster_resolver = cluster_resolver

    @classmethod
    def _validate_init_args(cls, mesh, cluster_resolver):
        if False:
            i = 10
            return i + 15
        if mesh and cluster_resolver:
            raise ValueError(f'Mesh and cluster_resolver can not be provided at the same time. Received mesh = {mesh}, cluster_resolver = {cluster_resolver}')
        if mesh and len(mesh.shape()) != 1:
            raise ValueError(f'The mesh for MultiWorkerMirroredStrategy must be 1D, received: {len(mesh.shape())}D')

    def reduce(self, reduce_op, value, axis):
        if False:
            for i in range(10):
                print('nop')
        return dtensor_util.dtensor_reduce(self, reduce_op, value, axis)

    @property
    def mesh(self):
        if False:
            return 10
        'Returns the mesh used by the strategy.'
        return self._mesh

def _parse_dtensor_env_var_from_cluster_resolver(cluster_resolver):
    if False:
        while True:
            i = 10
    'Parse the env vars for Dtensor based on the cluster resolver.\n\n  In the multi-client setting, each of the DTensor jobs need to aware of each\n  other, and the interface to setup those values are via the envvars. The\n  value used by dtensor are different from the existing\n  `MultiWorkerMirroredStrategy`. This function will parse the value from\n  cluster resolver, and populate the corresponding value for DTensor jobs in the\n  `os.environ`.\n\n  Args:\n    cluster_resolver: A `tf.distribute.cluster_resolver.ClusterResolver`\n      instance.\n\n  Returns:\n    A dict of {Str:Str} which contains all the env vars needed by DTensor jobs.\n    The value is for verification purpose.\n\n  Raises:\n    The value parsed from existing cluster spec is not valid.\n  '
    result = {}
    cluster_spec = multi_worker_util.normalize_cluster_spec(cluster_resolver.cluster_spec())
    dtensor_jobs = []
    if 'chief' in cluster_spec.jobs:
        dtensor_jobs.extend(cluster_spec.job_tasks('chief'))
    if 'worker' in cluster_spec.jobs:
        dtensor_jobs.extend(cluster_spec.job_tasks('worker'))
    if None in dtensor_jobs:
        raise ValueError(f'Unexpected dtensor job address from cluster spec: {cluster_spec}')
    result['DTENSOR_JOBS'] = ','.join(dtensor_jobs)
    result['DTENSOR_NUM_CLIENTS'] = str(len(dtensor_jobs))
    if cluster_resolver.task_type == 'chief':
        dtensor_client_id = 0
    elif cluster_resolver.task_type == 'worker':
        dtensor_client_id = cluster_resolver.task_id
        if 'chief' in cluster_spec.jobs:
            dtensor_client_id += 1
    result['DTENSOR_CLIENT_ID'] = str(dtensor_client_id)
    result['DTENSOR_JOB_NAME'] = 'worker'
    return result

def _config_dtensor_env_var(dtensor_env_vars):
    if False:
        i = 10
        return i + 15
    for (k, v) in dtensor_env_vars.items():
        os.environ[k] = v

def _build_distributed_mesh(batch_dim_name):
    if False:
        i = 10
        return i + 15
    device_type = d_config.preferred_device_type()
    local_devices = d_config.local_devices(device_type)
    number_clients = d_config.num_clients()
    dtensor_util.initialize_accelerator_system_once(device_type)
    mesh_dims = [(batch_dim_name, len(local_devices) * number_clients)]
    return mesh_util.create_distributed_mesh(mesh_dims, device_type=device_type)