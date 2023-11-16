"""Library of TPU helper functions."""
import enum
import math
from typing import List, Optional, Tuple
import numpy as np
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu.topology import Topology
from tensorflow.python.util.tf_export import tf_export
SINGLE_CORE_ASSIGNMENT = [[[0, 0, 0, 0]]]

def _compute_task_and_cores_to_replicas(core_assignment, topology):
    if False:
        while True:
            i = 10
    'Computes a nested dict which maps task and logical core to replicas.'
    task_and_cores_to_replicas = {}
    for replica in range(core_assignment.shape[0]):
        for logical_core in range(core_assignment.shape[1]):
            coordinates = core_assignment[replica, logical_core, :]
            task_id = topology.task_ordinal_at_coordinates(coordinates)
            if task_id not in task_and_cores_to_replicas:
                task_and_cores_to_replicas[task_id] = {}
            if logical_core not in task_and_cores_to_replicas[task_id]:
                task_and_cores_to_replicas[task_id][logical_core] = set()
            task_and_cores_to_replicas[task_id][logical_core].add(replica)
    task_to_sorted_replica_id = {}
    for (task, core_to_replicas) in task_and_cores_to_replicas.items():
        core_to_sorted_replicas = {}
        for (core, replicas) in core_to_replicas.items():
            core_to_sorted_replicas[core] = sorted(replicas)
        task_to_sorted_replica_id[task] = core_to_sorted_replicas
    return task_to_sorted_replica_id

@tf_export('tpu.experimental.DeviceOrderMode')
class DeviceOrderMode(enum.IntEnum):
    """The way of determining device orders when computing device assignment."""
    AUTO = 0
    RING = 1
    MESH = 2

@tf_export('tpu.experimental.DeviceAssignment')
class DeviceAssignment(object):
    """Mapping from logical cores in a computation to the physical TPU topology.

  Prefer to use the `DeviceAssignment.build()` helper to construct a
  `DeviceAssignment`; it is easier if less flexible than constructing a
  `DeviceAssignment` directly.
  """

    def __init__(self, topology: Topology, core_assignment: np.ndarray):
        if False:
            print('Hello World!')
        'Constructs a `DeviceAssignment` object.\n\n    Args:\n      topology: A `Topology` object that describes the physical TPU topology.\n      core_assignment: A logical to physical core mapping, represented as a\n        rank 3 numpy array. See the description of the `core_assignment`\n        property for more details.\n\n    Raises:\n      ValueError: If `topology` is not `Topology` object.\n      ValueError: If `core_assignment` is not a rank 3 numpy array.\n    '
        if not isinstance(topology, Topology):
            raise ValueError('topology must be a Topology object, got {}'.format(type(topology)))
        core_assignment = np.asarray(core_assignment, dtype=np.int32)
        self._topology = topology
        if core_assignment.ndim != 3:
            raise ValueError(f'core_assignment must be a rank 3 numpy array, got shape {core_assignment.shape}')
        self._num_replicas = core_assignment.shape[0]
        self._num_cores_per_replica = core_assignment.shape[1]
        if core_assignment.shape[-1] != topology.mesh_rank:
            raise ValueError(f'core_assignment.shape[-1] must have size equal to topology rank ({topology.mesh_rank}), got core_assignment.shape={core_assignment.shape}')
        self._core_assignment = core_assignment
        self._task_and_cores_to_replicas = _compute_task_and_cores_to_replicas(self._core_assignment, topology)

    @property
    def topology(self) -> Topology:
        if False:
            return 10
        'A `Topology` that describes the TPU topology.'
        return self._topology

    @property
    def num_cores_per_replica(self) -> int:
        if False:
            print('Hello World!')
        'The number of cores per replica.'
        return self._num_cores_per_replica

    @property
    def num_replicas(self) -> int:
        if False:
            i = 10
            return i + 15
        'The number of replicas of the computation.'
        return self._num_replicas

    @property
    def core_assignment(self) -> np.ndarray:
        if False:
            print('Hello World!')
        'The logical to physical core mapping.\n\n    Returns:\n      An integer numpy array of rank 3, with shape\n      `[num_replicas, num_cores_per_replica, topology_rank]`. Maps\n      (replica, logical core) pairs to physical topology coordinates.\n    '
        return self._core_assignment

    def coordinates(self, replica: int, logical_core: int) -> Tuple:
        if False:
            return 10
        'Returns the physical topology coordinates of a logical core.'
        return tuple(self.core_assignment[replica, logical_core, :])

    def lookup_replicas(self, task_id: int, logical_core: int) -> List[int]:
        if False:
            print('Hello World!')
        'Lookup replica ids by task number and logical core.\n\n    Args:\n      task_id: TensorFlow task number.\n      logical_core: An integer, identifying a logical core.\n    Returns:\n      A sorted list of the replicas that are attached to that task and\n      logical_core.\n    Raises:\n      ValueError: If no replica exists in the task which contains the logical\n      core.\n    '
        try:
            return self._task_and_cores_to_replicas[task_id][logical_core]
        except KeyError:
            raise ValueError('Can not find any replica in task: {} contains logical_core: {} '.format(task_id, logical_core))

    def tpu_ordinal(self, replica: int=0, logical_core: int=0) -> int:
        if False:
            while True:
                i = 10
        'Returns the ordinal of the TPU device assigned to a logical core.'
        coordinates = self.coordinates(replica, logical_core)
        return self._topology.tpu_device_ordinal_at_coordinates(coordinates)

    def host_device(self, replica: int=0, logical_core: int=0, job: Optional[str]=None) -> str:
        if False:
            while True:
                i = 10
        'Returns the CPU device attached to a logical core.'
        coordinates = self.coordinates(replica, logical_core)
        return self._topology.cpu_device_name_at_coordinates(coordinates, job=job)

    def tpu_device(self, replica: int=0, logical_core: int=0, job: Optional[str]=None) -> str:
        if False:
            return 10
        'Returns the name of the TPU device assigned to a logical core.'
        coordinates = self.coordinates(replica, logical_core)
        return self._topology.tpu_device_name_at_coordinates(coordinates, job=job)

    @classmethod
    def build(cls, topology: Topology, computation_shape: Optional[np.ndarray]=None, computation_stride: Optional[np.ndarray]=None, num_replicas: int=1, device_order_mode: DeviceOrderMode=DeviceOrderMode.AUTO) -> 'DeviceAssignment':
        if False:
            i = 10
            return i + 15
        return device_assignment(topology=topology, computation_shape=computation_shape, computation_stride=computation_stride, num_replicas=num_replicas, device_order_mode=device_order_mode)

def _open_ring_2d(x_size: int, y_size: int, z_coord: int) -> List[Tuple[int, int, int]]:
    if False:
        print('Hello World!')
    'Ring-order of a X by Y mesh, with a fixed Z coordinate.\n\n  For example, in a 4x4 mesh, this returns the following order.\n    0 -- 1 -- 2 -- 3\n    |    |    |    |\n    15-- 6 -- 5 -- 4\n    |    |    |    |\n    14-- 7 -- 8 -- 9\n    |    |    |    |\n    13-- 12-- 11-- 10\n\n  Note that chip 0 is not included in the output.\n\n  Args:\n    x_size: An integer represents the mesh size in the x-dimension. Must be\n      larger than 1.\n    y_size: An integer represents the mesh size in the y-dimension. Must be\n      larger than 1.\n    z_coord: An integer represents the z-coordinate to use for the chips in the\n      ring.\n\n  Returns:\n    A list of (x,y,z) triples in ring order.\n  '
    ret = []
    for i in range(y_size // 2):
        for j in range(1, x_size):
            ret.append((j, 2 * i, z_coord))
        for j in range(x_size - 1, 0, -1):
            ret.append((j, 2 * i + 1, z_coord))
    for i in range(y_size - 1, 0, -1):
        ret.append((0, i, z_coord))
    return ret

def _ring_3d(x_size: int, y_size: int, z_size: int) -> List[Tuple[int, int, int]]:
    if False:
        return 10
    'Ring-order of a X by Y by Z mesh.\n\n  Constructs the 3d ring from 2d rings that are stacked in the Z dimension and\n  joined in one corner.\n\n  z == 0:\n    0 -- 1 -- 2 -- 3\n    |    |    |    |\n    15 - 6 -- 5 -- 4\n    |    |    |    |\n    14 - 7 -- 8 -- 9\n    |    |    |    |\n    13 - 12 - 11 - 10\n  z == 1:\n    63 - 30 - 29 - 28\n    |    |    |    |\n    16 - 25 - 26 - 27\n    |    |    |    |\n    17 - 24 - 23 - 22\n    |    |    |    |\n    18 - 19 - 20 - 21\n  z == 2:\n    62 - 31 - 32 - 33\n    |    |    |    |\n    45 - 36 - 35 - 34\n    |    |    |    |\n    44 - 37 - 38 - 39\n    |    |    |    |\n    43 - 42 - 41 - 40\n  z == 3:\n    61 - 60 - 59 - 58\n    |    |    |    |\n    46 - 55 - 56 - 57\n    |    |    |    |\n    47 - 54 - 53 - 52\n    |    |    |    |\n    48 - 49 - 50 - 51\n\n  Args:\n    x_size: An integer represents the mesh size in the x-dimension. Must be\n      larger than 1.\n    y_size: An integer represents the mesh size in the y-dimension. Must be\n      larger than 1.\n    z_size: An integer represents the mesh size in the z-dimension. Must be\n      larger than 1.  For example, in a 4x4x4 mesh, this returns the following\n      order.\n\n  Returns:\n    A list of (x,y,z) triples in ring order.\n  '
    if x_size == 1 and y_size == 1:
        return [(0, 0, i) for i in range(z_size)]
    if x_size == 1 and z_size == 1:
        return [(0, i, 0) for i in range(y_size)]
    if y_size == 1 and z_size == 1:
        return [(i, 0, 0) for i in range(x_size)]
    if x_size > 1 and x_size % 2 != 0 or (y_size > 1 and y_size % 2 != 0) or (z_size > 1 and z_size % 2 != 0):
        logging.warning('Odd dimension')
        ret = []
        for z in range(z_size):
            for y in range(y_size):
                ret.extend(((x, y, z) for x in range(x_size)))
        return ret
    ret = [(0, 0, 0)]
    if z_size == 1:
        ret.extend(_open_ring_2d(x_size, y_size, 0))
        return ret
    if y_size == 1:
        ret = [(0, 0, 0)]
        ret.extend(((x, y, z) for (x, z, y) in _open_ring_2d(x_size, z_size, 0)))
        return ret
    if x_size == 1:
        ret = [(0, 0, 0)]
        ret.extend(((x, y, z) for (y, z, x) in _open_ring_2d(y_size, z_size, 0)))
        return ret
    ret = [(0, 0, 0)]
    for i in range(0, z_size):
        r = _open_ring_2d(x_size, y_size, i)
        if i % 2 == 0:
            ret.extend(r)
        else:
            ret.extend(reversed(r))
    for i in range(z_size - 1, 0, -1):
        ret.append((0, 0, i))
    return ret

def device_assignment(topology: Topology, computation_shape: Optional[np.ndarray]=None, computation_stride: Optional[np.ndarray]=None, num_replicas: int=1, device_order_mode: DeviceOrderMode=DeviceOrderMode.AUTO) -> DeviceAssignment:
    if False:
        print('Hello World!')
    "Computes a device_assignment of a computation across a TPU topology.\n\n  Attempts to choose a compact grid of cores for locality.\n\n  Returns a `DeviceAssignment` that describes the cores in the topology assigned\n  to each core of each replica.\n\n  `computation_shape` and `computation_stride` values should be powers of 2 for\n  optimal packing.\n\n  Args:\n    topology: A `Topology` object that describes the TPU cluster topology. To\n      obtain a TPU topology, evaluate the `Tensor` returned by\n      `initialize_system` using `Session.run`. Either a serialized\n      `TopologyProto` or a `Topology` object may be passed. Note: you must\n        evaluate the `Tensor` first; you cannot pass an unevaluated `Tensor`\n        here.\n    computation_shape: A rank 1 int32 numpy array with size equal to the\n      topology rank, describing the shape of the computation's block of cores.\n      If None, the `computation_shape` is `[1] * topology_rank`.\n    computation_stride: A rank 1 int32 numpy array of size `topology_rank`,\n      describing the inter-core spacing of the `computation_shape` cores in the\n      TPU topology. If None, the `computation_stride` is `[1] * topology_rank`.\n    num_replicas: The number of computation replicas to run. The replicas will\n      be packed into the free spaces of the topology.\n    device_order_mode: An enum of `DeviceOrderMode` class which indicates\n      whether to assign devices to form rings or meshes, or let the library to\n      choose.\n\n  Returns:\n    A DeviceAssignment object, which describes the mapping between the logical\n    cores in each computation replica and the physical cores in the TPU\n    topology.\n\n  Raises:\n    ValueError: If `topology` is not a valid `Topology` object.\n    ValueError: If `computation_shape` or `computation_stride` are not 1D int32\n      numpy arrays with shape [3] where all values are positive.\n    ValueError: If computation's replicas cannot fit into the TPU topology.\n  "
    if isinstance(topology, bytes):
        topology = Topology(serialized=topology)
    if not isinstance(topology, Topology):
        raise ValueError(f'`topology` is not a Topology object; got {type(topology)}')
    topology_rank = len(topology.mesh_shape)
    mesh_shape = topology.mesh_shape
    if computation_shape is None:
        computation_shape = np.array([1] * topology_rank, dtype=np.int32)
    else:
        computation_shape = np.asarray(computation_shape, dtype=np.int32)
    if computation_stride is None:
        computation_stride = np.array([1] * topology_rank, dtype=np.int32)
    else:
        computation_stride = np.asarray(computation_stride, dtype=np.int32)
    if computation_shape.shape != (topology_rank,):
        raise ValueError(f'computation_shape must have shape [{topology_rank}]; got {computation_shape.shape}')
    if computation_stride.shape != (topology_rank,):
        raise ValueError(f'computation_stride must have shape [{topology_rank}]; got {computation_stride.shape}')
    if any(computation_shape < 1):
        raise ValueError('computation_shape must be positive; got computation_shape={}'.format(computation_shape))
    if any(computation_stride < 1):
        raise ValueError('computation_stride must be positive; got computation_stride={}'.format(computation_stride))
    computation_footprint = computation_shape * computation_stride
    if any(computation_footprint > mesh_shape):
        raise ValueError('computation footprint {} does not fit in TPU topology shape {}'.format(computation_footprint, mesh_shape))
    block_counts = mesh_shape // computation_footprint
    replica_counts = block_counts * computation_stride
    max_replicas = np.prod(replica_counts)
    if num_replicas > max_replicas:
        raise ValueError('requested {} replicas but only {} replicas with shape {} and computation_stride {} fit in a TPU mesh of shape {}'.format(num_replicas, max_replicas, computation_shape, computation_stride, mesh_shape))

    def ceil_of_ratio(n, m):
        if False:
            for i in range(10):
                print('nop')
        return (n + m - 1) // m
    if topology.missing_devices.size == 0:
        replica_shape = [0] * topology_rank
        if num_replicas > 0:
            remaining_replicas = num_replicas
            remaining_dims = topology_rank
            for (x, ni) in sorted(((x, (i + 1) % topology_rank) for (i, x) in enumerate(replica_counts))):
                i = (ni + topology_rank - 1) % topology_rank
                target_size = int(math.ceil(remaining_replicas ** (1.0 / remaining_dims)))
                replica_shape[i] = min(target_size, x)
                remaining_replicas = ceil_of_ratio(remaining_replicas, replica_shape[i])
                remaining_dims -= 1
            assert remaining_replicas == 1 and remaining_dims == 0
        replica_offsets = np.full([num_replicas, topology_rank], -1, dtype=np.int32)
        enable_3d_tiling = topology_rank == 4 and computation_shape[-1] == mesh_shape[-1] and (np.prod(computation_stride) == 1) and (num_replicas == max_replicas)
        if device_order_mode != DeviceOrderMode.AUTO:
            if device_order_mode == DeviceOrderMode.RING and (not enable_3d_tiling):
                raise ValueError('device_order_mode=DeviceOrderMode.RING is not compatible with the 3D tiling current topology.  Try setting device_order_mode=DeviceOrderMode.AUTO')
            enable_3d_tiling = device_order_mode == DeviceOrderMode.RING
        if enable_3d_tiling:
            assignment = []
            inner_ring = _ring_3d(computation_shape[0], computation_shape[1], computation_shape[2])
            outer_ring = _ring_3d(replica_shape[0], replica_shape[1], replica_shape[2])
            for replica in range(num_replicas):
                (outer_x, outer_y, outer_z) = outer_ring[replica]
                per_replica_assignment = []
                for index in range(np.prod(computation_shape)):
                    (inner_x, inner_y, inner_z) = inner_ring[index // mesh_shape[-1]]
                    px = outer_x * computation_shape[0] + inner_x
                    py = outer_y * computation_shape[1] + inner_y
                    pz = outer_z * computation_shape[2] + inner_z
                    pi = index % mesh_shape[-1]
                    per_replica_assignment.append([px, py, pz, pi])
                assignment.append(per_replica_assignment)
        else:
            for replica in range(num_replicas):
                t = replica
                pos = []
                for dim in np.concatenate([[replica_shape[-1]], replica_shape[:-1]]):
                    pos.append(t % dim)
                    t //= dim
                replica_pos = np.concatenate([pos[1:], [pos[0]]])
                outer = replica_pos // computation_stride
                inner = replica_pos % computation_stride
                replica_offsets[replica, :] = outer * computation_footprint + inner
            indices = [np.arange(0, computation_shape[i] * computation_stride[i], computation_stride[i]) for i in range(topology_rank)]
            indices = np.concatenate([i[..., np.newaxis] for i in np.meshgrid(*indices, indexing='ij')], axis=-1)
            indices = indices.reshape((-1, topology_rank))
            assignment = indices + replica_offsets[:, np.newaxis, :]
    else:
        assert np.prod(computation_stride) == 1
        assert num_replicas * np.prod(computation_shape) <= topology.num_tasks * topology.num_tpus_per_task
        device_coordinates = topology.device_coordinates
        assignment = []
        devices_per_replica = np.prod(computation_shape)
        for rindex in range(num_replicas):
            replica_assignment = []
            for index in range(devices_per_replica):
                logical_id = rindex * devices_per_replica + index
                task = logical_id // topology.num_tpus_per_task
                device = logical_id % topology.num_tpus_per_task
                replica_assignment.append(device_coordinates[task, device, :])
            assignment.append(replica_assignment)
    return DeviceAssignment(topology, core_assignment=assignment)