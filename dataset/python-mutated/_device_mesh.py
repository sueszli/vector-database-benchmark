import logging
import math
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union
import torch
from torch.distributed.distributed_c10d import _find_pg_by_ranks_and_tag, _get_default_group, _get_group_tag, get_rank, get_world_size, init_process_group, is_initialized, new_group, ProcessGroup
logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    try:
        from numpy.typing import ArrayLike
    except ImportError:
        logger.warning('DeviceMesh requires numpy >= 1.21 to be installed for type checking')

class _MeshEnv:

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self.mesh_stack: List[DeviceMesh] = []
        self.child_to_parent_mapping: Dict[DeviceMesh, DeviceMesh] = {}

    def get_current_mesh(self) -> 'DeviceMesh':
        if False:
            return 10
        if len(self.mesh_stack) == 0:
            raise RuntimeError('No device mesh is currently active!')
        return self.mesh_stack[-1]

    def create_child_mesh(self, device_mesh: 'DeviceMesh', mesh_dim: int, mesh_dim_name: str) -> 'DeviceMesh':
        if False:
            i = 10
            return i + 15
        cur_rank = device_mesh.get_rank()
        pg_ranks_by_dim = device_mesh.mesh.swapdims(-1, mesh_dim).reshape(-1, device_mesh.mesh.size(mesh_dim))
        for mesh_1d in pg_ranks_by_dim:
            sub_mesh = DeviceMesh(device_mesh.device_type, mesh_1d, mesh_dim_names=(mesh_dim_name,), _init_process_groups=False)
            if cur_rank in mesh_1d:
                res_sub_mesh = sub_mesh
        res_sub_mesh._dim_group_infos = [device_mesh._dim_group_infos[mesh_dim]]
        self.child_to_parent_mapping[res_sub_mesh] = device_mesh
        return res_sub_mesh

    def get_parent_mesh(self, device_mesh: 'DeviceMesh') -> Optional['DeviceMesh']:
        if False:
            return 10
        return self.child_to_parent_mapping.get(device_mesh, None)

    def get_parent_mesh_dim(self, device_mesh: 'DeviceMesh') -> Optional[int]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the index of the mesh dim in the parent mesh.\n        The device_mesh passed in needs to be sliced out from a parent mesh.\n        '
        parent_mesh = self.get_parent_mesh(device_mesh)
        child_mesh_dim_names = device_mesh.mesh_dim_names
        if parent_mesh and child_mesh_dim_names:
            assert len(child_mesh_dim_names) == 1, 'The child mesh can only be a 1D mesh.'
            child_mesh_dim_name = child_mesh_dim_names[0]
            if parent_mesh.mesh_dim_names:
                return parent_mesh.mesh_dim_names.index(child_mesh_dim_name)
        return None

    @staticmethod
    def num_devices_per_host(device_type: str) -> int:
        if False:
            for i in range(10):
                print('nop')
        return _get_device_handle(device_type).device_count()

    @staticmethod
    def num_hosts(device_type: str) -> int:
        if False:
            return 10
        return get_world_size() // _MeshEnv.num_devices_per_host(device_type)
_mesh_resources: _MeshEnv = _MeshEnv()

def _get_device_handle(device_type: str='cuda'):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get the module corresponding to the device_type which is cuda or cuda-like device.\n    For example, when the device_type is cuda, the module `torch.cuda` is returned.\n    Return None when there is no corresponding module for device_type, otherwise\n    return the corresponding module.\n    '
    return getattr(torch, device_type, None)

class DeviceMesh:
    """
    DeviceMesh represents a mesh of devices, where layout of devices could be
    represented as a n-d dimension array, and each value of the n-d dimensional
    array is the global id of the default process group ranks.

    DeviceMesh could be used to describe the layout of devices across the cluster,
    and serves as a proxy for communication among the device lists within the cluster.

    We use the default ProcessGroup in this DeviceMesh class to implement proper
    communications. Note that we also add collective wrappers in this class. This is
    used to decouple detailed communication backend with the underlying
    DTensor implementation.

    DeviceMesh can be used as a context manager.

    .. note::
        DeviceMesh follows SPMD programming model, which means the same PyTorch Python program
        is running on all processes/ranks in the cluster. Therefore, users need to make sure the
        `mesh` array (which describes the layout of devices) should be identical across all ranks.
        Inconsistent `mesh` will lead to silent hang.

    Args:
        device_type (str): device type of the mesh. Currently supports: cpu, cuda/cuda-like.
        mesh (ndarray): could be a multi-dimension array or an integer tensor that
            describes the layout of devices, the ids are global ids of the
            default process group.

    Returns:
        A :class:`DeviceMesh` object

    Example (2 host with 4 GPUs each):
        ```
        # The following program runs on each process/rank in SPMD manner.
        # initialize device mesh as (2, 4) to represent the topology
        # of cross-host(dim 0), and within-host (dim 1)
        mesh = DeviceMesh(device_type="cuda",
                          mesh=[
                            [0, 1, 2, 3],
                            [4, 5, 6, 7]
                          ])
        ```
        A reduction over the first dimension of mesh will reduce across
        columns (0, 4), .. and (3, 7), a reduction over the second dimension
        of mesh reduces across rows (0, 1, 2, 3) and (4, 5, 6, 7)

    """
    device_type: str
    mesh: torch.Tensor
    mesh_dim_names: Optional[Tuple[str, ...]]

    def __init__(self, device_type: str, mesh: Union[torch.Tensor, 'ArrayLike'], *, mesh_dim_names: Optional[Tuple[str, ...]]=None, _init_process_groups: bool=True) -> None:
        if False:
            while True:
                i = 10
        self.device_type = device_type
        self.mesh = mesh.detach() if isinstance(mesh, torch.Tensor) else torch.tensor(mesh, dtype=torch.int)
        self.mesh_dim_names = mesh_dim_names
        self._flatten_mesh_list = tuple(self.mesh.flatten().tolist())
        self._hash = hash((self._flatten_mesh_list, self.mesh.shape))
        if device_type != 'xla':
            self._get_or_create_default_group()
            if _init_process_groups:
                self._init_process_groups()

    def _get_or_create_default_group(self):
        if False:
            for i in range(10):
                print('nop')
        default_initialized = is_initialized()
        if not default_initialized:
            init_process_group()
        world_size = get_world_size()
        if self.mesh.numel() > world_size:
            raise RuntimeError(f'Mesh should not be bigger than default world size, but found {self.mesh.numel()} ranks!')
        device_handle = _get_device_handle(self.device_type)
        if not default_initialized and device_handle:
            num_devices_per_host = device_handle.device_count()
            if world_size > num_devices_per_host and world_size % num_devices_per_host != 0:
                raise RuntimeError(f'DeviceMesh only support homogeneous hardware, but found {world_size} ranks and {num_devices_per_host} {self.device_type} devices!')
            device_handle.set_device(get_rank() % num_devices_per_host)
        rank_coords = (self.mesh == get_rank()).nonzero()
        assert rank_coords.size(0) in (0, 1)
        self._coordinate_on_dim: Optional[List[int]] = rank_coords[0].tolist() if rank_coords.size(0) > 0 else None
        return _get_default_group()

    def _init_process_groups(self):
        if False:
            return 10
        dim_group_infos: List[Tuple[str, List[int]]] = []
        if self.mesh.ndim == 1 and self.mesh.numel() == get_world_size():
            dim_group_infos.append((_get_group_tag(_get_default_group()), list(range(get_world_size()))))
        else:
            for dim in range(self.mesh.ndim):
                pg_ranks_by_dim = self.mesh.swapdims(-1, dim).reshape(-1, self.mesh.size(dim))
                for dim_mesh in pg_ranks_by_dim:
                    subgroup_ranks = dim_mesh.tolist()
                    dim_group = new_group(ranks=subgroup_ranks)
                    if self.get_rank() in subgroup_ranks:
                        if len(dim_group_infos) > dim:
                            raise RuntimeError(f'Each device mesh dimension should get only one process group, but got {self.get_rank} in {subgroup_ranks}!')
                        dim_group_infos.append((_get_group_tag(dim_group), subgroup_ranks))
        self._dim_group_infos = dim_group_infos

    def __enter__(self) -> 'DeviceMesh':
        if False:
            print('Hello World!')
        _mesh_resources.mesh_stack.append(self)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        if False:
            for i in range(10):
                print('nop')
        _mesh_resources.mesh_stack.pop()

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'DeviceMesh({self.mesh.tolist()})'

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return self._hash

    def __eq__(self, other: object) -> bool:
        if False:
            i = 10
            return i + 15
        if not isinstance(other, DeviceMesh):
            return False
        if id(self.mesh) == id(other.mesh):
            return True
        return self.mesh.shape == other.mesh.shape and self._flatten_mesh_list == other._flatten_mesh_list

    def __getitem__(self, mesh_dim_name: str) -> 'DeviceMesh':
        if False:
            while True:
                i = 10
        '\n        Slice the current DeviceMesh based on the mesh_dim_name given to create a child\n        DeviceMesh.\n\n        Args:\n            mesh_dim_name (str): the name of the mesh dimension of the parent DeviceMesh\n            to create a child DeviceMesh for.\n        Returns:\n            A :class:`DeviceMesh` object\n\n        Example (2 host with 4 GPUs each):\n        ```\n        # Below is a DeviceMesh with mesh_shape of (2, 4) and mesh_dim_name of ("dp", "tp")\n        mesh = DeviceMesh(device_type="cuda",\n                          mesh=[\n                            [0, 1, 2, 3],\n                            [4, 5, 6, 7]\n                          ],\n                          mesh_dim_names=["dp", "tp"])\n                          )\n        ```\n        Calling mesh["tp"] on rank 0, 1, 2, 3 would return a 1D child DeviceMesh:([0, 1, 2, 3]).\n        Calling mesh["tp"] on rank 4, 5, 6, 7 would return a 1D child DeviceMesh:([4, 5, 6, 7]).\n        Calling mesh["dp"] on rank 0, 4 would return a 1D child DeviceMesh:([0, 4]).\n        Calling mesh["dp"] on rank 1, 5 would return a 1D child DeviceMesh:([1, 5]).\n        Calling mesh["dp"] on rank 2, 6 would return a 1D child DeviceMesh:([2, 6]).\n        Calling mesh["dp"] on rank 3, 7 would return a 1D child DeviceMesh:([3, 7]).\n        '
        if self.mesh.ndim <= 1:
            raise RuntimeError(f'Cannot slice a DeviceMesh with {self.mesh.ndim} dimension.')
        if self.mesh_dim_names is None:
            raise KeyError('No `mesh_dim_names` found.', 'To slice the device mesh, please call `init_device_mesh` with `mesh_dim_names`.')
        if mesh_dim_name not in self.mesh_dim_names:
            raise KeyError(f"Mesh dimension '{mesh_dim_name}' does not exist.", f'Available mesh dimensions are: {self.mesh_dim_names}')
        mesh_dim = self.mesh_dim_names.index(mesh_dim_name)
        submesh = _mesh_resources.create_child_mesh(self, mesh_dim, mesh_dim_name)
        return submesh

    def get_dim_groups(self, mesh_dim: Optional[int]=None) -> Union[ProcessGroup, List[ProcessGroup]]:
        if False:
            while True:
                i = 10
        if not hasattr(self, '_dim_group_infos'):
            raise RuntimeError('DeviceMesh process groups not initialized!')
        if mesh_dim is not None:
            return _find_pg_by_ranks_and_tag(*self._dim_group_infos[mesh_dim])
        else:
            dim_groups = []
            for mesh_dim in range(self.mesh.ndim):
                dim_groups.append(_find_pg_by_ranks_and_tag(*self._dim_group_infos[mesh_dim]))
            return dim_groups

    def size(self, dim: Optional[int]=None) -> int:
        if False:
            return 10
        return self.mesh.numel() if dim is None else self.mesh.size(dim)

    @property
    def ndim(self) -> int:
        if False:
            return 10
        return self.mesh.ndim

    @property
    def shape(self) -> Tuple[int, ...]:
        if False:
            print('Hello World!')
        return tuple(self.mesh.shape)

    def get_rank(self) -> int:
        if False:
            i = 10
            return i + 15
        return get_rank()

    def get_coordinate(self) -> Optional[List[int]]:
        if False:
            i = 10
            return i + 15
        '\n        Return the relative indices of this rank relative to all\n        dimensions of the mesh. If this rank is not part of the mesh, return None.\n        '
        return self._coordinate_on_dim if self._coordinate_on_dim else None

def init_device_mesh(device_type: str, mesh_shape: Tuple[int, ...], *, mesh_dim_names: Optional[Tuple[str, ...]]=None) -> DeviceMesh:
    if False:
        print('Hello World!')
    '\n    Initializes a `DeviceMesh` based on `device_type`, `mesh_shape`, and `mesh_dim_names` parameters.\n    This creates a DeviceMesh with a mesh layout of n-d dimensional array, n being the len(mesh_shape)\n    and ith dimension being in size mesh_shape[i]. If mesh_dim_names is provided, each dimension is\n    labeled as mesh_dim_names[i].\n\n    .. note::\n        `init_device_mesh` follows SPMD programming model, which means the same PyTorch Python program\n        is running on all processes/ranks in the cluster. Therefore, users need to make sure the `mesh_shape`\n        tuple (the dimension of the nD array that describes the layout of devices) should be identical across\n        all ranks. Inconsistent `mesh_shape` will lead to silent hang.\n\n    Args:\n        device_type (str): device type of the mesh. Currently supports: cpu, cuda/cuda-like.\n        mesh_shape: Tuple[int]: A tuple defines the dimension of the multi-dimesnion array\n        that describes the layout of devices.\n    Kwargs:\n        mesh_dim_names: Optional[Tuple[str]]: A tuple of mesh dim names to be assigned to each dimension\n        of the multi-dimensional array that describes the layout of devices. Its length must match the length\n        of `mesh_shape`. Each string in mesh_dim_names must be unique.\n\n    Returns:\n        A :class:`DeviceMesh` object\n\n    .. note: If no process group is found, init_device_mesh will initialize distributed process group/groups\n    behind the scene, which are required for distributed communications.\n\n    Example:\n        >>> # xdoctest: +SKIP\n        >>> from torch.distributed._tensor.device_mesh import init_device_mesh\n        >>>\n        >>> mesh_1d = init_device_mesh("cuda", mesh_shape=(8,))\n        >>> mesh_2d = init_device_mesh("cuda", mesh_shape=(2, 8), mesh_dim_names=("dp", "tp"))\n    '
    if mesh_dim_names is not None:
        if len(set(mesh_dim_names)) != len(mesh_dim_names):
            raise RuntimeError('Each mesh_dim_name must be unique.', f'Found repeated mesh_dim_name in mesh_dim_names {mesh_dim_names}')
        if len(mesh_shape) != len(mesh_dim_names):
            raise RuntimeError('mesh_shape and mesh_dim_names should have same length!', f'Found len(mesh_dim_names): {len(mesh_dim_names)} and len(mesh_shape):{len(mesh_shape)}.')
    mesh = torch.arange(math.prod(mesh_shape)).view(mesh_shape)
    device_mesh = DeviceMesh(device_type=device_type, mesh=mesh, mesh_dim_names=mesh_dim_names)
    return device_mesh