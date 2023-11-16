import re
import sys
import time
from functools import partial, wraps
from typing import Tuple
import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch.distributed.rpc import _rref_context_get_debug_info
from torch.testing._internal.common_utils import FILE_SCHEMA, TEST_WITH_TSAN
if not dist.is_available():
    print('c10d not available, skipping tests', file=sys.stderr)
    sys.exit(0)
INIT_METHOD_TEMPLATE = FILE_SCHEMA + '{file_name}'

def dist_init(old_test_method=None, setup_rpc: bool=True, clean_shutdown: bool=True, faulty_messages=None, messages_to_delay=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    We use this decorator for setting up and tearing down state since\n    MultiProcessTestCase runs each `test*` method in a separate process and\n    each process just runs the `test*` method without actually calling\n    \'setUp\' and \'tearDown\' methods of unittest.\n\n    Note: pass the string representation of MessageTypes that should be used\n    with the faulty agent\'s send function. By default, all retriable messages\n    ("RREF_FORK_REQUEST", "RREF_CHILD_ACCEPT", "RREF_USER_DELETE",\n    "CLEANUP_AUTOGRAD_CONTEXT_REQ") will use the faulty send (this default is\n    set from faulty_rpc_agent_test_fixture.py).\n    '
    if old_test_method is None:
        return partial(dist_init, setup_rpc=setup_rpc, clean_shutdown=clean_shutdown, faulty_messages=faulty_messages, messages_to_delay=messages_to_delay)

    @wraps(old_test_method)
    def new_test_method(self, *arg, **kwargs):
        if False:
            print('Hello World!')
        import torch.distributed.rpc.api as api
        api._ignore_rref_leak = False
        self.worker_id = self.rank
        self.setup_fault_injection(faulty_messages, messages_to_delay)
        rpc_backend_options = self.rpc_backend_options
        if setup_rpc:
            if TEST_WITH_TSAN:
                rpc_backend_options.rpc_timeout = rpc.constants.DEFAULT_RPC_TIMEOUT_SEC * 5
                rpc.constants.DEFAULT_SHUTDOWN_TIMEOUT = 60
            rpc.init_rpc(name='worker%d' % self.rank, backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=rpc_backend_options)
        return_value = old_test_method(self, *arg, **kwargs)
        if setup_rpc:
            rpc.shutdown(graceful=clean_shutdown)
        return return_value
    return new_test_method

def noop() -> None:
    if False:
        return 10
    pass

def wait_until_node_failure(rank: int, expected_error_regex: str='.*') -> str:
    if False:
        i = 10
        return i + 15
    '\n    Loops until an RPC to the given rank fails. This is used to\n    indicate that the node has failed in unit tests.\n    Args:\n    rank (int): Rank of the node expected to fail\n    expected_error_regex (optional, str): Regex of exception message expected. Useful to ensure a specific failure\n    occurs, not just any.\n    '
    while True:
        try:
            rpc.rpc_sync(f'worker{rank}', noop, args=())
            time.sleep(0.1)
        except Exception as e:
            if re.search(pattern=expected_error_regex, string=str(e)):
                return str(e)

def wait_until_pending_futures_and_users_flushed(timeout: int=20) -> None:
    if False:
        for i in range(10):
            print('nop')
    "\n    The RRef protocol holds forkIds of rrefs in a map until those forks are\n    confirmed by the owner. The message confirming the fork may arrive after\n    our tests check whether this map is empty, which leads to failures and\n    flaky tests. to_here also does not guarantee that we have finished\n    processind the owner's confirmation message for the RRef. This function\n    loops until the map is empty, which means the messages have been received\n    as processed. Call this function before asserting the map returned by\n    _get_debug_info is empty.\n    "
    start = time.time()
    while True:
        debug_info = _rref_context_get_debug_info()
        num_pending_futures = int(debug_info['num_pending_futures'])
        num_pending_users = int(debug_info['num_pending_users'])
        if num_pending_futures == 0 and num_pending_users == 0:
            break
        time.sleep(0.1)
        if time.time() - start > timeout:
            raise ValueError('Timed out waiting to flush pending futures and users, had {} pending futures and {} pending users'.format(num_pending_futures, num_pending_users))

def get_num_owners_and_forks() -> Tuple[str, str]:
    if False:
        print('Hello World!')
    '\n    Retrieves number of OwnerRRefs and forks on this node from\n    _rref_context_get_debug_info.\n    '
    rref_dbg_info = _rref_context_get_debug_info()
    num_owners = rref_dbg_info['num_owner_rrefs']
    num_forks = rref_dbg_info['num_forks']
    return (num_owners, num_forks)

def wait_until_owners_and_forks_on_rank(num_owners: int, num_forks: int, rank: int, timeout: int=20) -> None:
    if False:
        while True:
            i = 10
    '\n    Waits until timeout for num_forks and num_owners to exist on the rank. Used\n    to ensure proper deletion of RRefs in tests.\n    '
    start = time.time()
    while True:
        (num_owners_on_rank, num_forks_on_rank) = rpc.rpc_sync(worker_name(rank), get_num_owners_and_forks, args=(), timeout=5)
        num_owners_on_rank = int(num_owners_on_rank)
        num_forks_on_rank = int(num_forks_on_rank)
        if num_owners_on_rank == num_owners and num_forks_on_rank == num_forks:
            return
        time.sleep(1)
        if time.time() - start > timeout:
            raise ValueError('Timed out waiting {} sec for {} owners and {} forks on rank, had {} owners and {} forks'.format(timeout, num_owners, num_forks, num_owners_on_rank, num_forks_on_rank))

def initialize_pg(init_method, rank: int, world_size: int) -> None:
    if False:
        for i in range(10):
            print('nop')
    if not dist.is_initialized():
        dist.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=world_size)

def worker_name(rank: int) -> str:
    if False:
        i = 10
        return i + 15
    return f'worker{rank}'

def get_function_event(function_events, partial_event_name):
    if False:
        return 10
    '\n    Returns the first event that matches partial_event_name in the provided\n    function_events. These function_events should be the output of\n    torch.autograd.profiler.function_events().\n\n    Args:\n    function_events: function_events returned by the profiler.\n    event_name (str): partial key that the event was profiled with.\n    '
    event = [event for event in function_events if partial_event_name in event.name][0]
    return event