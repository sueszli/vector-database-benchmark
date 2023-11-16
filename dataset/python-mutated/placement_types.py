from dataclasses import dataclass
from typing import cast, List, NamedTuple, Optional, Tuple
import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torch.distributed._tensor._collective_utils import mesh_broadcast, mesh_scatter
from torch.distributed._tensor.device_mesh import DeviceMesh

class Placement:

    def is_shard(self, dim: Optional[int]=None) -> bool:
        if False:
            print('Hello World!')
        if dim is not None and isinstance(self, Shard):
            return self.dim == dim
        else:
            return isinstance(self, Shard)

    def is_replicate(self) -> bool:
        if False:
            return 10
        return isinstance(self, Replicate)

    def is_partial(self) -> bool:
        if False:
            while True:
                i = 10
        return isinstance(self, _Partial)

class Shard(Placement):

    def __init__(self, dim):
        if False:
            print('Hello World!')
        self.dim = dim

    def _split_tensor(self, tensor: torch.Tensor, num_chunks: int, *, with_padding: bool=True, contiguous: bool=True) -> Tuple[List[torch.Tensor], List[int]]:
        if False:
            i = 10
            return i + 15
        '\n        This function uses torch.chunk to split a tensor into num_chunks shards along\n        the Shard placement dimension, and return a list of shards with their pad sizes.\n\n        Keyword args:\n            with_padding (bool, optional): when True, we pad the tensor on the last\n            few ranks before calling the collectives (i.e. scatter/all_gather, etc.).\n            This is because collectives usually require equal size tensor inputs\n        '
        assert self.dim <= tensor.ndim, f'Sharding dim {self.dim} greater than tensor ndim {tensor.ndim}'
        assert tensor.size(self.dim) > 0, f'Tensor size along dim{self.dim} is 0. There is nothing to be sharded.'
        tensor_list = list(torch.chunk(tensor, num_chunks, dim=self.dim))
        full_chunk_size = (tensor.size(self.dim) + num_chunks - 1) // num_chunks
        chunk_sizes = [tensor_list[idx].size(self.dim) if idx < len(tensor_list) else 0 for idx in range(num_chunks)]
        pad_sizes = [full_chunk_size - chunk_size for chunk_size in chunk_sizes]
        num_empty_tensors = num_chunks - len(tensor_list)
        tensor_size = list(tensor_list[0].size())
        tensor_size = [size if idx != self.dim else 0 for (idx, size) in enumerate(tensor_size)]
        tensor = tensor.new_zeros(tensor_size)
        for _ in range(num_empty_tensors):
            tensor_list.append(tensor)
        if with_padding or contiguous:
            shard_list = []
            for (shard, pad_size) in zip(tensor_list, pad_sizes):
                if with_padding and pad_size > 0:
                    shard = self._pad_tensor(shard, pad_size)
                shard = shard.contiguous() if contiguous else shard
                shard_list.append(shard)
            return (shard_list, pad_sizes)
        else:
            return (tensor_list, pad_sizes)

    def _pad_tensor(self, tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        pad = [0, 0] * (tensor.ndim - self.dim)
        pad[-1] = pad_size
        return torch.nn.functional.pad(tensor, pad)

    def _unpad_tensor(self, tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
        if False:
            while True:
                i = 10
        return tensor.narrow(self.dim, start=0, length=tensor.size(self.dim) - pad_size)

    def _local_shard_size_on_dim(self, size_on_dim: int, num_chunks: int, rank: int, return_offset: bool=False) -> Tuple[int, int]:
        if False:
            i = 10
            return i + 15
        '\n        returns the local shard size and offset on a given tensor dim\n        '
        assert size_on_dim >= num_chunks, f'Size to be sharded on dim {self.dim} must be at least as large as the number of devices in that dimension {num_chunks}'
        full_chunk_size = (size_on_dim + num_chunks - 1) // num_chunks
        chunk_sizes = [max(min(size_on_dim, full_chunk_size * (idx + 1)) - full_chunk_size * idx, 0) for idx in range(num_chunks)]
        local_shard_size = chunk_sizes[rank]
        local_offset_on_dim = -1
        if return_offset:
            local_offset_on_dim = sum(chunk_sizes[:rank])
        return (local_shard_size, local_offset_on_dim)

    def _shard_tensor(self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int) -> torch.Tensor:
        if False:
            while True:
                i = 10
        '\n        shard and scatter a tensor on a mesh dimension (use coordinate\n        0 on the mesh dimension as source of truth)\n        '
        my_coordinate = mesh.get_coordinate()
        num_chunks = mesh.size(dim=mesh_dim)
        if my_coordinate is None:
            return tensor.new_empty(0, requires_grad=tensor.requires_grad)
        (scatter_list, pad_sizes) = self._split_tensor(tensor, num_chunks, with_padding=True, contiguous=True)
        output = torch.empty_like(scatter_list[my_coordinate[mesh_dim]])
        mesh_scatter(output, scatter_list, mesh, mesh_dim=mesh_dim)
        pad_size = pad_sizes[my_coordinate[mesh_dim]]
        if pad_size > 0:
            output = self._unpad_tensor(output, pad_size)
        return output

    def _reduce_shard_tensor(self, tensor: torch.Tensor, mesh: DeviceMesh, reduce_op: c10d.ReduceOp.RedOpType, mesh_dim: int) -> torch.Tensor:
        if False:
            return 10
        '\n        reduce and scatter a tensor on a mesh dimension\n        '
        my_coordinate = mesh.get_coordinate()
        num_chunks = mesh.size(dim=mesh_dim)
        if my_coordinate is None:
            return tensor
        is_padded = tensor.size(self.dim) % num_chunks != 0
        if is_padded:
            (scattered_list, pad_sizes) = self._split_tensor(tensor, num_chunks, with_padding=True, contiguous=True)
            tensor = torch.cat(scattered_list, dim=self.dim)
        output = funcol.reduce_scatter_tensor(tensor, reduce_op.name, scatter_dim=self.dim, group=(mesh, mesh_dim))
        if is_padded:
            output = self._unpad_tensor(output, pad_sizes[my_coordinate[mesh_dim]])
        return output

    def _to_replicate_tensor(self, local_tensor: torch.Tensor, size: torch.Size, mesh: DeviceMesh, mesh_dim: int) -> torch.Tensor:
        if False:
            return 10
        '\n        This function all_gather all shards and return a tensor that\n        is replicated on the previously sharded mesh dimension\n        '
        my_coordinate = mesh.get_coordinate()
        num_chunks = mesh.size(dim=mesh_dim)
        if my_coordinate is None:
            return local_tensor
        full_chunk_size = (size[self.dim] + num_chunks - 1) // num_chunks
        chunk_sizes = [max(min(size[self.dim], full_chunk_size * (idx + 1)) - full_chunk_size * idx, 0) for idx in range(num_chunks)]
        pad_sizes = [full_chunk_size - chunk_size for chunk_size in chunk_sizes]
        is_padded = size[self.dim] % num_chunks != 0
        pad_size = pad_sizes[my_coordinate[mesh_dim]]
        if pad_size > 0:
            local_tensor = self._pad_tensor(local_tensor, pad_size)
        local_tensor = local_tensor.contiguous()
        result = funcol.all_gather_tensor(local_tensor, gather_dim=self.dim, group=(mesh, mesh_dim))
        if is_padded:
            full_pad_size = sum(pad_sizes)
            result = self._unpad_tensor(result, full_pad_size)
        return result

    def __eq__(self, other: object) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(other, Shard):
            return False
        return self.dim == other.dim

    def __hash__(self) -> int:
        if False:
            while True:
                i = 10
        return hash(self.dim)

    def __repr__(self) -> str:
        if False:
            return 10
        '\n        machine readable representation of the Shard placement\n        '
        return f'Shard(dim={self.dim})'

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'human readable representation of the Shard placement'
        return f'S({self.dim})'

class Replicate(Placement):

    def __eq__(self, other: object) -> bool:
        if False:
            while True:
                i = 10
        if not isinstance(other, Replicate):
            return False
        return True

    def __hash__(self) -> int:
        if False:
            print('Hello World!')
        return -1

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        '\n        machine readable representation of the Replicate placement\n        '
        return 'Replicate()'

    def __str__(self) -> str:
        if False:
            return 10
        '\n        human readable representation of the Replicate placement\n        '
        return 'R'

    def _replicate_tensor(self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int) -> torch.Tensor:
        if False:
            print('Hello World!')
        '\n        Replicate (broadcast) a torch.Tensor on a mesh dimension (use\n        the first coordinate on the mesh dimension as source of truth)\n        '
        my_coordinate = mesh.get_coordinate()
        if my_coordinate is None:
            return tensor.new_empty(0, requires_grad=tensor.requires_grad)
        tensor = tensor.contiguous()
        mesh_broadcast(tensor, mesh, mesh_dim=mesh_dim)
        return tensor

class _Partial(Placement):

    def __init__(self, reduce_op: c10d.ReduceOp.RedOpType=c10d.ReduceOp.SUM):
        if False:
            return 10
        self.reduce_op: c10d.ReduceOp.RedOpType = reduce_op

    def _to_replicate(self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int) -> torch.Tensor:
        if False:
            while True:
                i = 10
        return funcol.all_reduce(tensor, reduceOp=self.reduce_op.name, group=(mesh, mesh_dim))

    def _to_shard(self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int, shard_spec: Placement) -> torch.Tensor:
        if False:
            print('Hello World!')
        shard_spec = cast(Shard, shard_spec)
        return shard_spec._reduce_shard_tensor(tensor, mesh, self.reduce_op, mesh_dim)

    def __eq__(self, other: object) -> bool:
        if False:
            i = 10
            return i + 15
        if not isinstance(other, _Partial):
            return False
        return self.reduce_op == other.reduce_op

    def __hash__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return 1 + hash(self.reduce_op)

    def __repr__(self) -> str:
        if False:
            return 10
        '\n        machine readable representation of the Partial placement\n        '
        return f'_Partial(reduce_op={self.reduce_op})'

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        '\n        human readable representation of the Partial placement\n        '
        return 'P'

class TensorMeta(NamedTuple):
    shape: torch.Size
    stride: Tuple[int, ...]
    dtype: torch.dtype

@dataclass
class DTensorSpec:
    mesh: DeviceMesh
    placements: Tuple[Placement, ...]
    tensor_meta: Optional[TensorMeta] = None

    def __hash__(self) -> int:
        if False:
            return 10
        if self.tensor_meta is not None:
            return hash((self.mesh, self.placements, self.tensor_meta.shape, self.tensor_meta.stride, self.tensor_meta.dtype))
        else:
            return hash((self.mesh, self.placements))

    def __eq__(self, __o: object) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if not (isinstance(__o, DTensorSpec) and self.mesh == __o.mesh and (self.placements == __o.placements)):
            return False
        if self.tensor_meta is None or __o.tensor_meta is None:
            return self.tensor_meta == __o.tensor_meta
        return self.tensor_meta.shape == __o.tensor_meta.shape and self.tensor_meta.stride == __o.tensor_meta.stride and (self.tensor_meta.dtype == __o.tensor_meta.dtype)

    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        '\n        human readable representation of the DTensorSpec\n        '
        if len(self.placements) == 1:
            placement_str = str(self.placements[0])
        else:
            placement_str = str(self.placements)
        if self.tensor_meta is not None:
            tensor_shape = str(tuple(self.tensor_meta.shape))
        else:
            tensor_shape = 'unknown shape'
        return f'Spec({placement_str} on {tensor_shape})'

    @property
    def shape(self) -> torch.Size:
        if False:
            print('Hello World!')
        if self.tensor_meta is None:
            raise ValueError('tensor_meta is not set')
        return self.tensor_meta.shape

    @property
    def ndim(self) -> int:
        if False:
            while True:
                i = 10
        if self.tensor_meta is None:
            raise ValueError('tensor_meta is not set')
        return len(self.tensor_meta.shape)

    @property
    def num_shards(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        num_shards = 1
        for (i, placement) in enumerate(self.placements):
            if placement.is_shard():
                num_shards *= self.mesh.size(i)
        return num_shards

    @property
    def dim_map(self) -> List[int]:
        if False:
            return 10
        '\n        dim_map is a property we derive from `placements` of\n        the distributed tensor. It simply return a list of ints\n        where dim_map[i] denotes the sharding mapping to the mesh\n        dimension, and len(dim_map) == dist_tensor.ndim\n        dim_map[i] = -1: means tensor dim i replicate on mesh\n        dim_map[i] = j: means tensor dim i shard on mesh dim j\n\n        For example, we have a dist tensor that have the shape of\n        [18, 20, 30], and device_mesh([0, 1, 2, 3]), placements:\n        [Shard(1)], the dim_map of this placement would be:\n        [-1, 0, -1]. This representation is pretty helpful during\n        sharding propagation where we could know exactly each\n        tensor dimension is sharded or not.\n\n        Note that if placements contains `_Partial`, we have to\n        explicitly deal with it, so that when we create a DTensorSpec\n        with dim_map, we could properly record the pending sums.\n        '
        r = [-1] * self.ndim
        for (i, placement) in enumerate(self.placements):
            if placement.is_shard():
                shard_dim = cast(Shard, placement).dim
                if r[shard_dim] > -1:
                    raise ValueError(f'Tensor dim {shard_dim} is already sharded on mesh dim {r[shard_dim]}, DTensor operator implementation does not support things like hybrid sharding strategies yet (i.e. [Shard(0), Shard(0)])')
                r[shard_dim] = i
        return r

    @property
    def sums(self) -> List[int]:
        if False:
            i = 10
            return i + 15
        '\n        sums is a property we derive from `placements` of the\n        distributed tensor. It simply return a list of ints where\n        sums[i] denotes the pending sum (partial) on mesh dim i\n        '
        return [idx for (idx, placement) in enumerate(self.placements) if placement.is_partial()]

    @classmethod
    def from_dim_map(cls, mesh: DeviceMesh, dim_map: List[int], sums: List[int], tensor_meta: Optional[TensorMeta]=None) -> 'DTensorSpec':
        if False:
            print('Hello World!')
        '\n        Construct a DTensorSpec from dim_map list and pending sum.\n\n        Args:\n            mesh (class:`DeviceMesh`): device mesh to be used in the DTensorSpec\n            dim_map (List[int]): a list of integer that represents sharding on each\n                tensor dimension, see `dim_map` property doc for details\n            sums (List[int]): a list of integer that represents the dist tensor have\n                pending sum on which device mesh dimension.\n            tensor meta (TensorMeta): DTensor metadata\n\n        Return:\n            a class:`DTensorSpec` object\n        '
        placements: List[Placement] = [Replicate() for _ in range(mesh.ndim)]
        for s in sums:
            placements[s] = _Partial()
        for (i, m) in enumerate(dim_map):
            if m >= 0:
                placement = placements[m]
                if placement.is_shard():
                    placement = cast(Shard, placement)
                    raise RuntimeError(f"DeviceMesh dimension cann't be mapped to two dimension of the same tensor: {i} and {placement.dim}")
                elif placement.is_partial():
                    raise RuntimeError(f'DeviceMesh dimension {m} cannot be both shard and partial!')
                placements[m] = Shard(i)
        return cls(mesh, tuple(placements), tensor_meta=tensor_meta)

    def is_replicated(self):
        if False:
            while True:
                i = 10
        '\n        return True if the current DTensorSpec replicates on all mesh dims (devices)\n        '
        return all((placement.is_replicate() for placement in self.placements))