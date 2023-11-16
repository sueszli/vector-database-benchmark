import torch.distributed as dist
from torch._C._distributed_c10d import _create_work_from_future, AllgatherOptions, AllreduceOptions, BarrierOptions, ReduceScatterOptions, BroadcastOptions, ScatterOptions, AllToAllOptions
from torch.futures import Future
from typing import List
from torch import Tensor

def ret_work(ret):
    if False:
        i = 10
        return i + 15
    fut = Future()
    fut.set_result(ret)
    return _create_work_from_future(fut)

class FakeProcessGroup(dist.ProcessGroup):
    """
    A fake process group (not related to FakeTensor) is a process group which
    doesn't actually do any communication, it just hallucinates some
    communication.  You can run a single rank with a fake process group
    without needing multiple processes (simulates per-rank behavior)

    NOTE: This is not a real process group, and it would produce wrong results
    for every collective. It should be used as a convinient tool when playing
    with distributed but don't care about the actual data.
    """

    def __init__(self, rank, world_size):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(rank, world_size)
        self._rank = rank
        self._world_size = world_size

    def allreduce(self, tensor_list, opts=AllreduceOptions()):
        if False:
            return 10
        return ret_work(tensor_list)

    def allreduce_coalesced(self, tensor_list, opts=AllreduceOptions()):
        if False:
            return 10
        return ret_work(tensor_list)

    def allgather(self, output_tensors, input_tensor, opts=AllgatherOptions()):
        if False:
            return 10
        for chunk in output_tensors[0]:
            chunk.copy_(input_tensor[0])
        return ret_work(output_tensors)

    def reduce_scatter(self, output_tensor, scatter_list, opts=ReduceScatterOptions()):
        if False:
            while True:
                i = 10
        return ret_work(output_tensor)

    def _allgather_base(self, output_tensor, input_tensor, opts=AllgatherOptions()):
        if False:
            while True:
                i = 10
        chunks = output_tensor.chunk(self._world_size)
        for chunk in chunks:
            chunk.copy_(input_tensor)
        return ret_work(output_tensor)

    def _reduce_scatter_base(self, output_tensor, input_tensor, opts=ReduceScatterOptions()):
        if False:
            print('Hello World!')
        return ret_work(output_tensor)

    def barrier(self, opts=BarrierOptions()):
        if False:
            return 10
        pass

    def broadcast(self, tensors: List[Tensor], opts=BroadcastOptions()):
        if False:
            print('Hello World!')
        return ret_work(tensors)

    def scatter(self, output_tensors: List[Tensor], input_tensors: List[List[Tensor]], opts=ScatterOptions()):
        if False:
            return 10
        return ret_work(output_tensors)

    def alltoall(self, output_tensors: List[Tensor], input_tensors: List[Tensor], opts=AllToAllOptions()):
        if False:
            print('Hello World!')
        return ret_work(output_tensors)

    def alltoall_base(self, output_tensor: Tensor, input_tensor: Tensor, output_split_sizes: List[int], input_split_sizes: List[int], opts=AllToAllOptions()):
        if False:
            while True:
                i = 10
        return ret_work(output_tensor)

    def send(self, tensors: List[Tensor], dstRank: int, tag: int):
        if False:
            for i in range(10):
                print('nop')
        return ret_work(None)

    def recv(self, tensors: List[Tensor], srcRank: int, tag: int):
        if False:
            i = 10
            return i + 15
        return ret_work(tensors)

    def getBackendName(self):
        if False:
            for i in range(10):
                print('nop')
        return 'fake'

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'FakePG world_size:{self._world_size} rank:{self._rank}'

class FakeStore(dist.Store):
    """
    A fake store is a fake Key-Value store simply for initialization usage
    the of fake process group, one can either use FakeStore or HashStore.
    """
    pass

def _create_fake_pg(prefix_store, rank, world_size, timeout):
    if False:
        i = 10
        return i + 15
    return FakeProcessGroup(rank, world_size)
dist.Backend.register_backend('fake', _create_fake_pg, devices=['cpu', 'cuda'])