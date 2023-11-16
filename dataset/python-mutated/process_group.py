import hashlib
from collections import OrderedDict
import paddle
from paddle.framework import core
from ...collective import _get_global_env, _new_ring_id
from ...utils.log_utils import get_logger
from .utils import dygraph_guard
logger = get_logger('INFO', __name__)

def get_all_process_groups():
    if False:
        while True:
            i = 10
    global _g_process_group_map
    return _g_process_group_map.values()

def get_process_group(group_id, g_process_group_map=None):
    if False:
        i = 10
        return i + 15
    global _g_process_group_map
    return _g_process_group_map.get(group_id, None) if g_process_group_map is None else g_process_group_map.get(group_id, None)

def get_world_process_group():
    if False:
        for i in range(10):
            print('nop')
    global _g_process_group_map
    return _g_process_group_map[0]

def clear_all_process_groups():
    if False:
        print('Hello World!')
    global _g_process_group_map
    _g_process_group_map = {}
    _g_process_group_map[0] = ProcessGroup(0, [])

def remove_process_group(ring_id):
    if False:
        return 10
    global _g_process_group_map
    if ring_id in _g_process_group_map:
        _g_process_group_map.pop(ring_id)

def new_process_group(ranks, group_id=None, force_new_group=False, group_type=None):
    if False:
        while True:
            i = 10
    global _g_process_group_map
    if not force_new_group:
        new_key = '_'.join(map(str, ranks))
        for (pg_id, pg) in _g_process_group_map.items():
            cur_key = '_'.join(map(str, pg.ranks))
            if pg_id != 0 and new_key == cur_key:
                return pg
    num_groups = len(_g_process_group_map)
    if group_id is None:
        group_id = _new_ring_id() + num_groups + 1
    new_pg = ProcessGroup(group_id, ranks, group_type)
    _g_process_group_map[group_id] = new_pg
    return new_pg

class ProcessGroup:

    def __init__(self, group_id, ranks, group_type=None):
        if False:
            while True:
                i = 10
        if group_id == 0 and get_process_group(0) is not None:
            assert group_id != 0, 'Process group id 0 is reserved for all ranks.'
        self._group_id = group_id
        self._ranks = ranks
        if group_id != 0:
            global _g_process_group_map
            _g_process_group_map[0].add_ranks(ranks)
        self._is_instantiate = False
        self._group_type = group_type

    @property
    def id(self):
        if False:
            while True:
                i = 10
        return self._group_id

    @property
    def ranks(self):
        if False:
            return 10
        return self._ranks

    @property
    def nranks(self):
        if False:
            while True:
                i = 10
        return len(self._ranks)

    @property
    def group_type(self):
        if False:
            print('Hello World!')
        return self._group_type

    def add_ranks(self, new_ranks):
        if False:
            i = 10
            return i + 15
        if set(new_ranks) <= set(self.ranks):
            return
        else:
            assert not self.is_instantiate(), 'Cannot add new ranks after instantiating the process group'
        self._ranks.extend(new_ranks)
        self._ranks = list(set(self.ranks))

    def local_rank(self, global_rank):
        if False:
            return 10
        if global_rank in self.ranks:
            return self.ranks.index(global_rank)
        else:
            raise AssertionError(f"Rank {global_rank} doesn't belong to this group")

    def is_instantiate(self):
        if False:
            i = 10
            return i + 15
        return self._is_instantiate

    @dygraph_guard
    def instantiate(self):
        if False:
            print('Hello World!')
        if self._is_instantiate:
            return
        ring_id = self.id
        genv = _get_global_env()
        global_rank = genv.rank
        if self.nranks >= 2 and global_rank in self.ranks:
            strategy = core.ParallelStrategy()
            strategy.nranks = self.nranks
            strategy.local_rank = self.local_rank(global_rank)
            strategy.trainer_endpoints = [genv.trainer_endpoints[i] for i in self.ranks]
            strategy.current_endpoint = genv.current_endpoint
            strategy.nrings = 1
            if core.is_compiled_with_cuda():
                place = core.CUDAPlace(genv.device_id)
                use_new_comm = paddle.get_flags('FLAGS_dynamic_static_unified_comm')['FLAGS_dynamic_static_unified_comm']
                if use_new_comm:
                    store = core.create_or_get_global_tcp_store()
                    endpoints_str = ''
                    for endpoint in strategy.trainer_endpoints:
                        endpoints_str += endpoint
                    endpoints_str += f'ring_id:{ring_id}'
                    endpoints_str_hash = hashlib.md5(endpoints_str.encode(encoding='UTF-8')).hexdigest()
                    core.CommContextManager.set_device_id(genv.device_id)
                    core.CommContextManager.create_nccl_comm_context(store, str(ring_id), strategy.local_rank, strategy.nranks, endpoints_str_hash)
                else:
                    core.NCCLParallelContext(strategy, place).init_with_ring_id(ring_id)
            elif core.is_compiled_with_xpu():
                place = core.XPUPlace(genv.device_id)
                core.BKCLParallelContext(strategy, place).init_with_ring_id(ring_id)
            elif genv.device_type in core.get_all_custom_device_type():
                place = core.CustomPlace(genv.device_type, genv.device_id)
                core.XCCLParallelContext(strategy, place).init_with_ring_id(ring_id)
            else:
                raise AssertionError('No CUDA device found')
            if core.is_compiled_with_cuda():
                paddle.set_device('gpu:%d' % paddle.distributed.ParallelEnv().dev_id)
            elif core.is_compiled_with_xpu():
                paddle.set_device('xpu:%d' % paddle.distributed.ParallelEnv().dev_id)
            elif genv.device_type in core.get_all_custom_device_type():
                paddle.set_device('%s:%d' % (paddle.distributed.ParallelEnv().device_type, paddle.distributed.ParallelEnv().dev_id))
            barrier_tensor = paddle.full([1], 1, dtype='int32')
            paddle._legacy_C_ops.barrier(barrier_tensor, barrier_tensor, 'ring_id', ring_id)
            if self._group_type == 'p2p':
                alltoall_tmp = paddle.empty(shape=[self.nranks, self.nranks], dtype='int32')
                paddle._legacy_C_ops.alltoall(alltoall_tmp, 'use_calc_stream', True, 'ring_id', ring_id)
                paddle.device.cuda.synchronize()
        if self.nranks > 1:
            barrier_tensor = paddle.full([1], 1, dtype='int32')
            paddle._legacy_C_ops.barrier(barrier_tensor, barrier_tensor, 'ring_id', 0)
        self._is_instantiate = True

    def is_member(self):
        if False:
            print('Hello World!')
        return True

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        if not isinstance(other, ProcessGroup):
            return False
        if self.id != other.id:
            return False
        return True

    def __ne__(self, other):
        if False:
            i = 10
            return i + 15
        return not self.__eq__(other)

    def __str__(self):
        if False:
            while True:
                i = 10
        string = 'id: {}, nranks: {}, ranks: {}.'.format(self.id, self.nranks, ', '.join(map(str, self.ranks)))
        return string

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash(self.__str__())
_g_process_group_map = OrderedDict()
_g_process_group_map[0] = ProcessGroup(0, [])