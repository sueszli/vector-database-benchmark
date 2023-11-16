import datetime
import os
from typing import Any, List, Optional, Union
import torch
import torch.distributed as dist
from torch import Tensor
from typing_extensions import Self
from lightning.fabric.plugins.collectives.collective import Collective
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_1_13
from lightning.fabric.utilities.types import CollectibleGroup, RedOpType, ReduceOp
if dist.is_available():
    from torch.distributed.constants import default_pg_timeout
else:
    default_pg_timeout = datetime.timedelta(seconds=1800)

class TorchCollective(Collective):
    """Collective operations using `torch.distributed <https://pytorch.org/docs/stable/distributed.html>`__.

    .. warning:: This is an :ref:`experimental <versioning:Experimental API>` feature which is still in development.

    """
    manages_default_group = False

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        if not dist.is_available():
            raise RuntimeError('Torch distributed is not available.')
        super().__init__()

    @property
    def group(self) -> CollectibleGroup:
        if False:
            while True:
                i = 10
        if self._group is None:
            self._group = dist.GroupMember.WORLD
        return super().group

    @property
    def rank(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return dist.get_rank(self.group)

    @property
    def world_size(self) -> int:
        if False:
            print('Hello World!')
        return dist.get_world_size(self.group)

    def broadcast(self, tensor: Tensor, src: int) -> Tensor:
        if False:
            i = 10
            return i + 15
        dist.broadcast(tensor, src, group=self.group)
        return tensor

    def all_reduce(self, tensor: Tensor, op: Union[str, ReduceOp, RedOpType]='sum') -> Tensor:
        if False:
            print('Hello World!')
        op = self._convert_to_native_op(op)
        dist.all_reduce(tensor, op=op, group=self.group)
        return tensor

    def reduce(self, tensor: Tensor, dst: int, op: Union[str, ReduceOp, RedOpType]='sum') -> Tensor:
        if False:
            return 10
        op = self._convert_to_native_op(op)
        dist.reduce(tensor, dst, op=op, group=self.group)
        return tensor

    def all_gather(self, tensor_list: List[Tensor], tensor: Tensor) -> List[Tensor]:
        if False:
            while True:
                i = 10
        dist.all_gather(tensor_list, tensor, group=self.group)
        return tensor_list

    def gather(self, tensor: Tensor, gather_list: List[Tensor], dst: int=0) -> List[Tensor]:
        if False:
            for i in range(10):
                print('nop')
        dist.gather(tensor, gather_list, dst, group=self.group)
        return gather_list

    def scatter(self, tensor: Tensor, scatter_list: List[Tensor], src: int=0) -> Tensor:
        if False:
            while True:
                i = 10
        dist.scatter(tensor, scatter_list, src, group=self.group)
        return tensor

    def reduce_scatter(self, output: Tensor, input_list: List[Tensor], op: Union[str, ReduceOp, RedOpType]='sum') -> Tensor:
        if False:
            print('Hello World!')
        op = self._convert_to_native_op(op)
        dist.reduce_scatter(output, input_list, op=op, group=self.group)
        return output

    def all_to_all(self, output_tensor_list: List[Tensor], input_tensor_list: List[Tensor]) -> List[Tensor]:
        if False:
            for i in range(10):
                print('nop')
        dist.all_to_all(output_tensor_list, input_tensor_list, group=self.group)
        return output_tensor_list

    def send(self, tensor: Tensor, dst: int, tag: int=0) -> None:
        if False:
            while True:
                i = 10
        dist.send(tensor, dst, tag=tag, group=self.group)

    def recv(self, tensor: Tensor, src: Optional[int]=None, tag: int=0) -> Tensor:
        if False:
            while True:
                i = 10
        dist.recv(tensor, src, tag=tag, group=self.group)
        return tensor

    def all_gather_object(self, object_list: List[Any], obj: Any) -> List[Any]:
        if False:
            for i in range(10):
                print('nop')
        dist.all_gather_object(object_list, obj, group=self.group)
        return object_list

    def broadcast_object_list(self, object_list: List[Any], src: int, device: Optional[torch.device]=None) -> List[Any]:
        if False:
            while True:
                i = 10
        dist.broadcast_object_list(object_list, src, group=self.group, device=device)
        return object_list

    def gather_object(self, obj: Any, object_gather_list: List[Any], dst: int=0) -> List[Any]:
        if False:
            print('Hello World!')
        dist.gather_object(obj, object_gather_list, dst, group=self.group)
        return object_gather_list

    def scatter_object_list(self, scatter_object_output_list: List[Any], scatter_object_input_list: List[Any], src: int=0) -> List[Any]:
        if False:
            return 10
        dist.scatter_object_list(scatter_object_output_list, scatter_object_input_list, src, group=self.group)
        return scatter_object_output_list

    def barrier(self, device_ids: Optional[List[int]]=None) -> None:
        if False:
            return 10
        if self.group == dist.GroupMember.NON_GROUP_MEMBER:
            return
        dist.barrier(group=self.group, device_ids=device_ids)

    def monitored_barrier(self, timeout: Optional[datetime.timedelta]=None, wait_all_ranks: bool=False) -> None:
        if False:
            while True:
                i = 10
        dist.monitored_barrier(group=self.group, timeout=timeout, wait_all_ranks=wait_all_ranks)

    def setup(self, main_address: Optional[str]=None, main_port: Optional[str]=None, **kwargs: Any) -> Self:
        if False:
            for i in range(10):
                print('nop')
        if self.is_initialized():
            return self
        set_addr = False
        addr_key = 'MASTER_ADDR'
        if main_address is not None and addr_key not in os.environ:
            os.environ[addr_key] = main_address
            set_addr = True
        set_port = False
        port_key = 'MASTER_PORT'
        if main_port is not None and port_key not in os.environ:
            os.environ[port_key] = str(main_port)
            set_port = True
        super().setup(**kwargs)
        TorchCollective.manages_default_group = True
        if set_addr:
            os.environ.pop('MASTER_ADDR', None)
        if set_port:
            os.environ.pop('MASTER_PORT', None)
        return self

    def teardown(self) -> Self:
        if False:
            while True:
                i = 10
        group_member = self.group != dist.GroupMember.NON_GROUP_MEMBER
        super().teardown()
        if group_member and TorchCollective.manages_default_group and ((default_group := dist.GroupMember.WORLD) is not None) and (len(dist.distributed_c10d._pg_map) == 1):
            self.destroy_group(default_group)
            TorchCollective.manages_default_group = False
        elif TorchCollective.manages_default_group and dist.GroupMember.WORLD is None:
            TorchCollective.manages_default_group = False
        return self

    @classmethod
    def is_available(cls) -> bool:
        if False:
            print('Hello World!')
        return dist.is_available()

    @classmethod
    def is_initialized(cls) -> bool:
        if False:
            while True:
                i = 10
        return cls.is_available() and dist.is_initialized()

    @classmethod
    def init_group(cls, **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        dist.init_process_group(**kwargs)

    @classmethod
    def new_group(cls, **kwargs: Any) -> CollectibleGroup:
        if False:
            i = 10
            return i + 15
        return dist.new_group(**kwargs)

    @classmethod
    def destroy_group(cls, group: CollectibleGroup) -> None:
        if False:
            for i in range(10):
                print('nop')
        if group in dist.distributed_c10d._pg_map:
            dist.destroy_process_group(group)

    @classmethod
    def _convert_to_native_op(cls, op: Union[str, ReduceOp, RedOpType]) -> Union[ReduceOp, RedOpType]:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(op, ReduceOp) or (_TORCH_GREATER_EQUAL_1_13 and isinstance(op, RedOpType)):
            return op
        if not isinstance(op, str):
            raise ValueError(f'Unsupported op {op!r} of type {type(op).__name__}')
        op = op.upper()
        value = getattr(ReduceOp, op, None)
        if value is None:
            raise ValueError(f'op {op!r} is not a member of `ReduceOp`')
        return value