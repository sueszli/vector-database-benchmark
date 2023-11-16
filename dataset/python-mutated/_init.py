import os
from cupy.cuda import nccl
from cupyx.distributed import _store
from cupyx.distributed._nccl_comm import NCCLBackend
_backends = {'nccl': NCCLBackend}

def init_process_group(n_devices, rank, *, backend='nccl', host=None, port=None, use_mpi=False):
    if False:
        return 10
    'Start `cupyx.distributed` and obtain a communicator.\n\n    This call initializes the distributed environment, it needs to be\n    called for every process that is involved in the communications.\n\n    A single device per returned communication is only allowed. It is the user\n    responsibility of setting the appropiated gpu to be used before creating\n    and using the communicator.\n\n    Currently the user needs to specify each process rank and the total\n    number of processes, and start all the processes in different hosts\n    manually.\n\n    The process with rank 0 will spawn a TCP server using a\n    subprocess that listens in the port indicated by\n    the env var `CUPYX_DISTRIBUTED_PORT`, the rank 0 must be executed\n    in the host determined by the env var `CUPYX_DISTRIBUTED_HOST`.\n    In case their values are not specified, `\'127.0.0.1\'` and `13333` will be\n    used by default.\n\n    Note that this feature is expected to be used within a trusted cluster\n    environment.\n\n    Example:\n\n        >>> import cupy\n        >>> def process_0():\n        ...     import cupyx.distributed\n        ...     cupy.cuda.Device(0).use()\n        ...     comm = cupyx.distributed.init_process_group(2, 0)\n        ...     array = cupy.ones(1)\n        ...     comm.broadcast(array, 0)\n        ...\n        >>> def process_1():\n        ...     import cupyx.distributed\n        ...     cupy.cuda.Device(1).use()\n        ...     comm = cupyx.distributed.init_process_group(2, 1)\n        ...     array = cupy.zeros(1)\n        ...     comm.broadcast(array, 0)\n        ...     cupy.equal(array, cupy.ones(1))\n\n    Args:\n        n_devices (int): Total number of devices that will be used in the\n            distributed execution.\n        rank (int): Unique id of the GPU that the communicator is associated to\n            its value needs to be `0 <= rank < n_devices`.\n        backend (str): Backend to use for the communications. Optional,\n            defaults to `"nccl"`.\n        host (str): host address for the process rendezvous on initialization\n            defaults to `None`.\n        port (int): port for the process rendezvous on initialization\n            defaults to `None`.\n        use_mpi (bool): if ``False``, it avoids using MPI for synchronization\n            and uses the provided TCP server for exchanging CPU only\n            information.\n            defaults to `False`.\n    Returns:\n        Backend: object used to perform communications, adheres to the\n            :class:`~cupyx.distributed.Backend` specification:\n    '
    if n_devices <= 0:
        raise ValueError(f'Invalid number of devices {n_devices}')
    if not 0 <= rank < n_devices:
        raise ValueError(f'Invalid number of rank {rank} {n_devices}')
    if backend not in _backends:
        raise ValueError(f'{backend} is not supported')
    if backend == 'nccl' and (not nccl.available):
        raise RuntimeError('NCCL is not available')
    if host is None:
        host = os.environ.get('CUPYX_DISTRIBUTED_HOST', _store._DEFAULT_HOST)
    if port is None:
        port = int(os.environ.get('CUPYX_DISTRIBUTED_PORT', _store._DEFAULT_PORT))
    return _backends[backend](n_devices, rank, host, port, use_mpi)