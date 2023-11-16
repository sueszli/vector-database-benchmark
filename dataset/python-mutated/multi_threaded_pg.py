import sys
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from functools import partial, reduce
import torch
import torch.distributed as dist
import weakref
from torch._C._distributed_c10d import _create_work_from_future, AllgatherOptions, AllreduceOptions, AllToAllOptions, BarrierOptions, BroadcastOptions, ReduceScatterOptions, ScatterOptions, Store, ReduceOp
from torch.distributed.distributed_c10d import _CollOp, _store_based_barrier, P2POp
from torch.futures import Future
from torch.utils import _pytree as pytree
"\nTODO:\nLots of missing collectives.\nCollectives validation.\nMake timeout robust by making collectives respect the test deadline.\nMake tests robust by making collectives interruptible.\nWe need some synchronization around cleanup to ensure that timedout ranks don't cause spurious failures.\n\n"

def flatten_list(lst):
    if False:
        print('Hello World!')
    return pytree.tree_leaves(lst)

def ret_work(ret):
    if False:
        i = 10
        return i + 15
    fut = Future()
    fut.set_result(ret)
    return _create_work_from_future(fut)

def binop_reduce(tensors, op):
    if False:
        print('Hello World!')
    res = op(torch.stack(tensors), dim=0)
    if isinstance(res, torch.Tensor):
        return res
    return res.values

def bitwise_reduce(tensors, op):
    if False:
        for i in range(10):
            print('nop')
    return reduce(op, tensors)
_reduce_ops = {ReduceOp.SUM: partial(binop_reduce, op=torch.sum), ReduceOp.AVG: partial(binop_reduce, op=torch.mean), ReduceOp.PRODUCT: partial(binop_reduce, op=torch.prod), ReduceOp.MIN: partial(binop_reduce, op=torch.min), ReduceOp.MAX: partial(binop_reduce, op=torch.max), ReduceOp.BAND: partial(bitwise_reduce, op=torch.bitwise_and), ReduceOp.BOR: partial(bitwise_reduce, op=torch.bitwise_or), ReduceOp.BXOR: partial(bitwise_reduce, op=torch.bitwise_xor)}

class AllToAll:

    @torch.no_grad()
    def work(self, data):
        if False:
            i = 10
            return i + 15
        world_size = len(data)
        for dest_rank in range(world_size):
            (output_tensor_list, _) = data[dest_rank]
            for src_rank in range(world_size):
                (_, input_tensor_list) = data[src_rank]
                output_tensor_list[src_rank].copy_(input_tensor_list[dest_rank])

class AllReduce:

    def __init__(self, op):
        if False:
            i = 10
            return i + 15
        if op.op not in _reduce_ops:
            raise NotImplementedError(f'AllReduce op {op.op} not supported on multithreaded pg for now.')
        self.op = op.op

    @torch.no_grad()
    def work(self, data):
        if False:
            return 10
        for i in range(len(data[0])):
            tensors = []
            rank_0_device = data[0][i].device
            for src_rank in range(0, len(data)):
                tensors.append(data[src_rank][i].to(rank_0_device))
            res = _reduce_ops[self.op](tensors)
            for src_rank in range(len(data)):
                data[src_rank][i].copy_(res.to(data[src_rank][i].device))

class AllGather:

    @torch.no_grad()
    def work(self, data):
        if False:
            while True:
                i = 10
        for src_rank in range(len(data)):
            in_tensor_list = data[src_rank][1]
            assert len(in_tensor_list) == 1
            src_tensor = in_tensor_list[0]
            for dest in data:
                dest_tensor = dest[0][0][src_rank]
                dest_tensor.copy_(src_tensor)

class Scatter:

    def __init__(self, src):
        if False:
            print('Hello World!')
        self.src = src

    @torch.no_grad()
    def work(self, data):
        if False:
            i = 10
            return i + 15
        src_in_tensor_list = data[self.src][1]
        assert len(src_in_tensor_list) == 1
        src_in_tensors = src_in_tensor_list[0]
        for (rank, each_rank_data) in enumerate(data):
            out_tensor_list = each_rank_data[0]
            assert len(out_tensor_list) == 1
            dest_tensor = out_tensor_list[0]
            dest_tensor.copy_(src_in_tensors[rank])

class Gather:

    def __init__(self, dst):
        if False:
            print('Hello World!')
        self.dst = dst

    @torch.no_grad()
    def work(self, data):
        if False:
            while True:
                i = 10
        assert len(data[self.dst][0]) == 1
        out_tensor_list = data[self.dst][0][0]
        for (rank, each_rank_data) in enumerate(data):
            src_in_tensor_list = each_rank_data[1]
            assert len(src_in_tensor_list) == 1
            dest_tensor = out_tensor_list[rank]
            dest_tensor.copy_(src_in_tensor_list[0])

class ReduceScatter:

    def __init__(self, op):
        if False:
            for i in range(10):
                print('nop')
        if op != dist.ReduceOp.SUM:
            raise NotImplementedError('ReduceScatter only supports SUM on threaded pg for now.')
        self.op = op

    @torch.no_grad()
    def work(self, data):
        if False:
            i = 10
            return i + 15
        start_reduction = [False for _ in range(len(data))]
        for each_rank_data in data:
            assert len(each_rank_data[1]) == 1
            to_scatter = each_rank_data[1][0]
            for i in range(len(to_scatter)):
                dest_tensor_on_rank_i = data[i][0]
                assert len(dest_tensor_on_rank_i) == 1
                dst_tensor_device = dest_tensor_on_rank_i[0].device
                if not start_reduction[i]:
                    dest_tensor_on_rank_i[0].copy_(to_scatter[i].to(dst_tensor_device))
                    start_reduction[i] = True
                else:
                    dest_tensor_on_rank_i[0].add_(to_scatter[i].to(dst_tensor_device))

class Broadcast:

    def __init__(self, src):
        if False:
            while True:
                i = 10
        self.src = src

    @torch.no_grad()
    def work(self, data):
        if False:
            i = 10
            return i + 15
        in_tensor_list = flatten_list(data[self.src])
        for i in range(len(data)):
            out_tensor_list = flatten_list(data[i])
            for j in range(len(in_tensor_list)):
                out_tensor_list[j].copy_(in_tensor_list[j])

class Collective:

    def __init__(self, world_size, collective, pg):
        if False:
            i = 10
            return i + 15
        self._world_size = world_size
        self._collective = collective
        self._start_cond = threading.Condition()
        self._done_cond = threading.Condition()
        self._data = [None] * world_size
        self._count = 0
        self._done = False
        self._pg = pg

    def join(self, rank, data):
        if False:
            i = 10
            return i + 15
        with self._start_cond:
            self._data[rank] = data
            self._count += 1
            if self._count == self._world_size:
                if rank > 0:
                    self._start_cond.notify()
            if rank == 0:
                self._start_cond.wait_for(lambda : self._count == self._world_size or self._pg._terminate.is_set())
                if self._pg._terminate.is_set():
                    sys.exit('Test termination event occurs.')
        with self._done_cond:
            if rank > 0:
                self._done_cond.wait_for(lambda : self._done or self._pg._terminate.is_set())
                if self._pg._terminate.is_set():
                    sys.exit('Test termination event occurs.')
            else:
                self._collective.work(self._data)
                self._done = True
                self._done_cond.notify_all()
        return ret_work(data)

class ProcessLocalGroup(dist.ProcessGroup):
    _coll_lock = threading.Lock()
    _cur_coll_on_pgs = {}
    _terminate = threading.Event()

    @classmethod
    def _start_coll(cls, collective, pg):
        if False:
            for i in range(10):
                print('nop')
        with cls._coll_lock:
            if pg.pg_name not in cls._cur_coll_on_pgs:
                cls._cur_coll_on_pgs[pg.pg_name] = Collective(pg.size(), collective, cls)
            return cls._cur_coll_on_pgs[pg.pg_name]

    @classmethod
    def _end_coll(cls, collective, pg):
        if False:
            print('Hello World!')
        with cls._coll_lock:
            if pg.pg_name in cls._cur_coll_on_pgs and cls._cur_coll_on_pgs[pg.pg_name] == collective:
                cls._cur_coll_on_pgs.pop(pg.pg_name)

    @classmethod
    def exception_handle(cls, exc):
        if False:
            i = 10
            return i + 15
        cls._terminate.set()
        for coll in cls._cur_coll_on_pgs.values():
            with coll._start_cond:
                coll._start_cond.notify()
            with coll._done_cond:
                coll._done_cond.notify_all()

    @classmethod
    def reset(cls):
        if False:
            return 10
        with cls._coll_lock:
            cls._cur_coll_on_pgs = {}
            cls._terminate.clear()

    def alltoall(self, output_tensor_list, input_tensor_list, opts=AllToAllOptions()):
        if False:
            i = 10
            return i + 15
        coll = ProcessLocalGroup._start_coll(AllToAll(), self)
        res = coll.join(self._rank, (output_tensor_list, input_tensor_list))
        ProcessLocalGroup._end_coll(coll, self)
        return res

    def allreduce(self, tensor_list, opts=AllreduceOptions()):
        if False:
            for i in range(10):
                print('nop')
        coll = ProcessLocalGroup._start_coll(AllReduce(opts.reduceOp), self)
        res = coll.join(self._rank, tensor_list)
        ProcessLocalGroup._end_coll(coll, self)
        return res

    def allreduce_coalesced(self, tensor_list, opts=AllreduceOptions()):
        if False:
            while True:
                i = 10
        coll = ProcessLocalGroup._start_coll(AllReduce(opts.reduceOp), self)
        res = coll.join(self._rank, tensor_list)
        ProcessLocalGroup._end_coll(coll, self)
        return res

    def barrier(self, opts=BarrierOptions()):
        if False:
            print('Hello World!')
        return self.allreduce(tensor_list=[torch.ones(1)])

    def allgather(self, output_tensors, input_tensor, opts=AllgatherOptions()):
        if False:
            return 10
        coll = ProcessLocalGroup._start_coll(AllGather(), self)
        res = coll.join(self._rank, (output_tensors, input_tensor))
        ProcessLocalGroup._end_coll(coll, self)
        return res

    def _allgather_base(self, output_tensor, input_tensor, opts=AllgatherOptions()):
        if False:
            for i in range(10):
                print('nop')
        tensor_list = list(torch.chunk(output_tensor, self._world_size))
        return self.allgather([tensor_list], [input_tensor], opts)

    def broadcast(self, tensor_list, opts=BroadcastOptions()):
        if False:
            while True:
                i = 10
        coll = ProcessLocalGroup._start_coll(Broadcast(opts.rootRank), self)
        res = coll.join(self._rank, tensor_list)
        ProcessLocalGroup._end_coll(coll, self)
        return res

    def scatter(self, output_tensors, input_tensors, opts=ScatterOptions()):
        if False:
            for i in range(10):
                print('nop')
        coll = ProcessLocalGroup._start_coll(Scatter(opts.rootRank), self)
        res = coll.join(self._rank, (output_tensors, input_tensors))
        ProcessLocalGroup._end_coll(coll, self)
        return res

    def gather(self, output_tensors, input_tensors, opts=ScatterOptions()):
        if False:
            i = 10
            return i + 15
        coll = ProcessLocalGroup._start_coll(Gather(opts.rootRank), self)
        res = coll.join(self._rank, (output_tensors, input_tensors))
        ProcessLocalGroup._end_coll(coll, self)
        return res

    def reduce_scatter(self, output_tensor, scatter_list, opts=ReduceScatterOptions()):
        if False:
            print('Hello World!')
        coll = ProcessLocalGroup._start_coll(ReduceScatter(opts.reduceOp), self)
        res = coll.join(self._rank, (output_tensor, scatter_list))
        ProcessLocalGroup._end_coll(coll, self)
        return res

    def _reduce_scatter_base(self, output_tensor, input_tensor, opts=AllgatherOptions()):
        if False:
            while True:
                i = 10
        tensor_list = list(torch.chunk(input_tensor, self._world_size))
        return self.reduce_scatter([output_tensor], [tensor_list], opts)

    def allgather_into_tensor_coalesced(self, output_tensor_list, input_tensor_list):
        if False:
            i = 10
            return i + 15
        res = None
        for (o_t, i_t) in zip(output_tensor_list, input_tensor_list):
            res = self._allgather_base(o_t, i_t)
        return res

    def __init__(self, rank, world_size):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(rank, world_size)
        self._rank = rank
        self._world_size = world_size
        world = dist.distributed_c10d._world
        if isinstance(world, ThreadLocalWorld):
            world = world._get_world()
        self._world = weakref.ref(world)
        self._ctx = torch.autograd.set_multithreading_enabled(False)

    def size(self):
        if False:
            print('Hello World!')
        return self._world_size

    @property
    def pg_name(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        return the global registered name of the current pg in the world\n        '
        return self._world().pg_names[self]

    def getBackendName(self):
        if False:
            return 10
        return 'threaded'

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'ThreadedPG world_size:{self._world_size} rank:{self._rank}'

def _create_threaded_pg(prefix_store, rank, world_size, timeout):
    if False:
        print('Hello World!')
    pg = ProcessLocalGroup(rank, world_size)
    _store_based_barrier(rank, prefix_store, '', world_size, timeout)
    return pg
dist.Backend.register_backend('threaded', _create_threaded_pg)

@dataclass
class WorldData:
    default_pg: dist.ProcessGroup
    pg_map: Dict[dist.ProcessGroup, Tuple[str, Optional[Store]]]
    pg_names: Dict[dist.ProcessGroup, str]
    pg_group_ranks: Dict[dist.ProcessGroup, Dict[int, int]]
    pg_backend_config: Dict[dist.ProcessGroup, str]
    group_count: int
    tags_to_pg: Dict[str, List[dist.ProcessGroup]]
    pg_to_tag: Dict[dist.ProcessGroup, str]
    pg_coalesce_state: Dict[dist.ProcessGroup, List[Union[_CollOp, P2POp]]]
    pg_default_device: Dict[dist.ProcessGroup, torch.device]

class ThreadLocalWorld:
    _world = threading.local()

    def _get_world(self) -> WorldData:
        if False:
            while True:
                i = 10
        if not hasattr(ThreadLocalWorld._world, 'world'):
            ThreadLocalWorld._world.world = WorldData(None, {}, {}, {}, {}, 0, {}, {}, {}, {})
        return ThreadLocalWorld._world.world

    @property
    def default_pg(self):
        if False:
            i = 10
            return i + 15
        return self._get_world().default_pg

    @default_pg.setter
    def default_pg(self, value):
        if False:
            i = 10
            return i + 15
        self._get_world().default_pg = value

    @property
    def pg_map(self):
        if False:
            i = 10
            return i + 15
        return self._get_world().pg_map

    @property
    def pg_names(self):
        if False:
            while True:
                i = 10
        return self._get_world().pg_names

    @property
    def pg_group_ranks(self):
        if False:
            for i in range(10):
                print('nop')
        return self._get_world().pg_group_ranks

    @property
    def pg_backend_config(self):
        if False:
            return 10
        return self._get_world().pg_backend_config

    @property
    def group_count(self) -> int:
        if False:
            return 10
        return self._get_world().group_count

    @group_count.setter
    def group_count(self, value):
        if False:
            i = 10
            return i + 15
        self._get_world().group_count = value

    @property
    def tags_to_pg(self):
        if False:
            print('Hello World!')
        return self._get_world().tags_to_pg

    @property
    def pg_to_tag(self):
        if False:
            return 10
        return self._get_world().pg_to_tag

    @property
    def pg_coalesce_state(self) -> Dict[dist.ProcessGroup, List[Union[_CollOp, P2POp]]]:
        if False:
            while True:
                i = 10
        return self._get_world().pg_coalesce_state

    @property
    def pg_default_device(self) -> Dict[dist.ProcessGroup, torch.device]:
        if False:
            while True:
                i = 10
        return self._get_world().pg_default_device
_old_pg_world = None
_ctx_manager = None

def _install_threaded_pg():
    if False:
        return 10
    global _old_pg_world
    global _ctx_manager
    _old_pg_world = dist.distributed_c10d._world
    dist.distributed_c10d._world = ThreadLocalWorld()
    _ctx_manager = torch.autograd.set_multithreading_enabled(False)
    return dist.distributed_c10d._world

def _uninstall_threaded_pg():
    if False:
        i = 10
        return i + 15
    dist.distributed_c10d._world = _old_pg_world