"""Abstract class for collective groups."""
from abc import ABCMeta
from abc import abstractmethod
from ray.util.collective.types import AllReduceOptions, BarrierOptions, ReduceOptions, AllGatherOptions, BroadcastOptions, ReduceScatterOptions

class BaseGroup(metaclass=ABCMeta):

    def __init__(self, world_size, rank, group_name):
        if False:
            return 10
        'Init the process group with basic information.\n\n        Args:\n            world_size: The total number of processes in the group.\n            rank: The rank of the current process.\n            group_name: The group name.\n        '
        self._world_size = world_size
        self._rank = rank
        self._group_name = group_name

    @property
    def rank(self):
        if False:
            while True:
                i = 10
        'Return the rank of the current process.'
        return self._rank

    @property
    def world_size(self):
        if False:
            print('Hello World!')
        'Return the number of processes in this group.'
        return self._world_size

    @property
    def group_name(self):
        if False:
            return 10
        'Return the group name of this group.'
        return self._group_name

    def destroy_group(self):
        if False:
            for i in range(10):
                print('nop')
        'GC the communicators.'
        pass

    @classmethod
    def backend(cls):
        if False:
            while True:
                i = 10
        'The backend of this collective group.'
        raise NotImplementedError()

    @abstractmethod
    def allreduce(self, tensor, allreduce_options=AllReduceOptions()):
        if False:
            print('Hello World!')
        raise NotImplementedError()

    @abstractmethod
    def barrier(self, barrier_options=BarrierOptions()):
        if False:
            return 10
        raise NotImplementedError()

    @abstractmethod
    def reduce(self, tensor, reduce_options=ReduceOptions()):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    @abstractmethod
    def allgather(self, tensor_list, tensor, allgather_options=AllGatherOptions()):
        if False:
            print('Hello World!')
        raise NotImplementedError()

    @abstractmethod
    def broadcast(self, tensor, broadcast_options=BroadcastOptions()):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    @abstractmethod
    def reducescatter(self, tensor, tensor_list, reducescatter_options=ReduceScatterOptions()):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    @abstractmethod
    def send(self, tensor, dst_rank):
        if False:
            print('Hello World!')
        raise NotImplementedError()

    @abstractmethod
    def recv(self, tensor, src_rank):
        if False:
            print('Hello World!')
        raise NotImplementedError()