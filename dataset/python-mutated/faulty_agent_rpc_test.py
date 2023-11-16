import torch
import time
import torch.distributed.rpc as rpc
from torch.distributed.rpc.api import _delete_all_user_and_unforked_owner_rrefs
from torch.testing._internal.dist_utils import dist_init, wait_until_pending_futures_and_users_flushed, wait_until_owners_and_forks_on_rank, worker_name
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import RpcAgentTestFixture

def my_sleep_func(seconds=1):
    if False:
        while True:
            i = 10
    time.sleep(seconds)
    return torch.mul(torch.tensor(1), torch.tensor(1))

@torch.jit.script
def my_script_func(tensor):
    if False:
        i = 10
        return i + 15
    return torch.add(tensor, tensor)

def add_rref_to_value(rref, value):
    if False:
        return 10
    return rref.to_here() + value

class FaultyAgentRpcTest(RpcAgentTestFixture):

    @dist_init(messages_to_delay={})
    def test_check_failed_messages(self):
        if False:
            i = 10
            return i + 15
        if self.rank == 0:
            dst_worker_b = worker_name((self.rank + 1) % self.world_size)
            dst_worker_c = worker_name((self.rank + 2) % self.world_size)
            rref = rpc.remote(dst_worker_b, torch.add, args=(torch.ones(2, 2), torch.ones(2, 2)))
            rpc.remote(dst_worker_c, add_rref_to_value, args=(rref, torch.ones(2, 2)))
            self.assertEqual(rref.to_here(), torch.add(torch.ones(2, 2), torch.ones(2, 2)))
        _delete_all_user_and_unforked_owner_rrefs()

    @dist_init
    def test_verify_backend_options(self):
        if False:
            return 10
        self.assertEqual(self.rpc_backend, rpc.backend_registry.BackendType.FAULTY_TENSORPIPE)
        self.assertEqual(self.rpc_backend_options.num_worker_threads, 8)
        self.assertEqual(self.rpc_backend_options.num_fail_sends, 3)
        self.assertEqual(len(self.rpc_backend_options.messages_to_fail), 4)
        self.assertEqual(len(self.rpc_backend_options.messages_to_delay), 2)
        self.assertEqual(self.rpc_backend_options.rpc_timeout, rpc.constants.DEFAULT_RPC_TIMEOUT_SEC)

    @dist_init(faulty_messages=['RREF_FORK_REQUEST', 'RREF_CHILD_ACCEPT'])
    def test_custom_faulty_messages(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual({'RREF_FORK_REQUEST', 'RREF_CHILD_ACCEPT'}, set(self.rpc_backend_options.messages_to_fail))

    @dist_init(faulty_messages=[])
    def test_no_faulty_messages(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(len(self.rpc_backend_options.messages_to_fail), 0)

    @dist_init(messages_to_delay={'SCRIPT_CALL': 1.5})
    def test_custom_messages_to_delay(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rpc_backend_options.messages_to_delay, {'SCRIPT_CALL': 1.5})

    def _test_remote_message_dropped_pickle(self, dst=None):
        if False:
            i = 10
            return i + 15
        if self.rank != 0:
            return
        dst_rank = dst if dst is not None else (self.rank + 1) % self.world_size
        dst_worker = f'worker{dst_rank}'
        rref = rpc.remote(dst_worker, my_sleep_func, args=(1,))
        wait_until_pending_futures_and_users_flushed()
        with self.assertRaisesRegex(RuntimeError, 'RRef creation'):
            rref._serialize()
        with self.assertRaisesRegex(RuntimeError, 'RRef creation'):
            rpc.rpc_async(dst_worker, add_rref_to_value, args=(rref, 1))

    @dist_init(faulty_messages=['PYTHON_REMOTE_CALL'])
    def test_remote_message_dropped_pickle(self):
        if False:
            print('Hello World!')
        self._test_remote_message_dropped_pickle()

    @dist_init(faulty_messages=['PYTHON_REMOTE_CALL'])
    def test_remote_message_dropped_pickle_to_self(self):
        if False:
            while True:
                i = 10
        self._test_remote_message_dropped_pickle(self.rank)

    def _test_remote_message_dropped_timeout(self, func, args, dst=None):
        if False:
            print('Hello World!')
        if self.rank != 0:
            return
        dst_rank = dst if dst is not None else (self.rank + 1) % self.world_size
        dst_worker = f'worker{dst_rank}'
        rref = rpc.remote(dst_worker, func, args=args)
        wait_until_pending_futures_and_users_flushed()
        with self.assertRaisesRegex(RuntimeError, 'RRef creation'):
            rref.to_here()

    @dist_init(faulty_messages=['SCRIPT_REMOTE_CALL'])
    def test_builtin_remote_message_dropped_timeout(self):
        if False:
            print('Hello World!')
        func = torch.add
        args = (torch.tensor(1), torch.tensor(1))
        self._test_remote_message_dropped_timeout(func, args)

    @dist_init(faulty_messages=['SCRIPT_REMOTE_CALL'])
    def test_builtin_remote_message_dropped_timeout_to_self(self):
        if False:
            for i in range(10):
                print('nop')
        func = torch.add
        args = (torch.tensor(1), torch.tensor(1))
        self._test_remote_message_dropped_timeout(func, args, dst=0)

    @dist_init(faulty_messages=['PYTHON_REMOTE_CALL'])
    def test_udf_remote_message_dropped_timeout(self):
        if False:
            print('Hello World!')
        func = my_sleep_func
        args = (2,)
        self._test_remote_message_dropped_timeout(func, args)

    @dist_init(faulty_messages=['PYTHON_REMOTE_CALL'])
    def test_udf_remote_message_dropped_timeout_to_self(self):
        if False:
            return 10
        func = my_sleep_func
        args = (2,)
        self._test_remote_message_dropped_timeout(func, args, dst=0)

    def _test_remote_message_delay_timeout(self, func, args, dst=None):
        if False:
            while True:
                i = 10
        if self.rank != 0:
            return
        dst_rank = dst if dst is not None else (self.rank + 1) % self.world_size
        dst_worker = f'worker{dst_rank}'
        rref = rpc.remote(dst_worker, func, args=args, timeout=0.001)
        expected_error = self.get_timeout_error_regex()
        with self.assertRaisesRegex(RuntimeError, expected_error):
            rref._get_future().wait()
        wait_until_pending_futures_and_users_flushed()
        with self.assertRaisesRegex(RuntimeError, 'RRef creation'):
            rref.to_here()
        if dst_rank != self.rank:
            slow_rref = rpc.remote(dst_worker, func, args=args, timeout=2)
            with self.assertRaisesRegex(RuntimeError, expected_error):
                slow_rref.to_here(0.001)
        if dst_rank != self.rank:
            wait_until_owners_and_forks_on_rank(2, 2, rank=dst_rank)

    @dist_init(faulty_messages=[], messages_to_delay={'PYTHON_REMOTE_CALL': 2})
    def test_udf_remote_message_delay_timeout(self):
        if False:
            while True:
                i = 10
        func = my_sleep_func
        args = (2,)
        self._test_remote_message_delay_timeout(func, args)

    @dist_init(faulty_messages=[], messages_to_delay={'PYTHON_REMOTE_CALL': 2})
    def test_udf_remote_message_delay_timeout_to_self(self):
        if False:
            while True:
                i = 10
        func = my_sleep_func
        args = (1,)
        self._test_remote_message_delay_timeout(func, args, dst=0)

    @dist_init(faulty_messages=[], messages_to_delay={'SCRIPT_REMOTE_CALL': 2, 'SCRIPT_RREF_FETCH_CALL': 1})
    def test_remote_message_builtin_delay_timeout(self):
        if False:
            print('Hello World!')
        func = torch.add
        args = (torch.tensor(1), torch.tensor(1))
        self._test_remote_message_delay_timeout(func, args)

    @dist_init(faulty_messages=[], messages_to_delay={'SCRIPT_REMOTE_CALL': 2, 'SCRIPT_RREF_FETCH_CALL': 1})
    def test_remote_message_builtin_delay_timeout_to_self(self):
        if False:
            while True:
                i = 10
        func = torch.add
        args = (torch.tensor(1), torch.tensor(1))
        self._test_remote_message_delay_timeout(func, args, dst=0)

    @dist_init(faulty_messages=[], messages_to_delay={'SCRIPT_REMOTE_CALL': 2, 'SCRIPT_RREF_FETCH_CALL': 1})
    def test_remote_message_script_delay_timeout(self):
        if False:
            print('Hello World!')
        func = my_script_func
        args = (torch.tensor(1),)
        self._test_remote_message_delay_timeout(func, args)

    @dist_init(faulty_messages=[], messages_to_delay={'SCRIPT_REMOTE_CALL': 2, 'SCRIPT_RREF_FETCH_CALL': 1})
    def test_remote_message_script_delay_timeout_to_self(self):
        if False:
            for i in range(10):
                print('nop')
        func = my_script_func
        args = (torch.tensor(1),)
        self._test_remote_message_delay_timeout(func, args, dst=0)

    @dist_init(faulty_messages=[], messages_to_delay={'SCRIPT_RREF_FETCH_CALL': 1})
    def test_rref_to_here_timeout(self):
        if False:
            print('Hello World!')
        if self.rank != 0:
            return
        dst_rank = (self.rank + 1) % self.world_size
        dst_worker = f'worker{dst_rank}'
        rref = rpc.remote(dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1)))
        expected_error = self.get_timeout_error_regex()
        with self.assertRaisesRegex(RuntimeError, expected_error):
            rref.to_here(0.01)
        rref.to_here()

    @dist_init(faulty_messages=[])
    def test_rpc_builtin_timeout(self):
        if False:
            print('Hello World!')
        next_rank = (self.rank + 1) % self.world_size
        dst_worker = worker_name(next_rank)
        expected_error = self.get_timeout_error_regex()
        with self.assertRaisesRegex(RuntimeError, expected_error):
            rpc.rpc_sync(dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1)), timeout=1)
        fut = rpc.rpc_async(dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1)), timeout=1)
        with self.assertRaisesRegex(RuntimeError, expected_error):
            fut.wait()
        fut = rpc.rpc_async(dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1)))
        fut.wait()
        rpc._set_rpc_timeout(0.001)
        fut = rpc.rpc_async(dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1)))
        with self.assertRaisesRegex(RuntimeError, expected_error):
            fut.wait()
        fut = rpc.rpc_async(dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1)), timeout=0)
        fut.wait()
        rpc._set_rpc_timeout(rpc.constants.DEFAULT_RPC_TIMEOUT_SEC)

    @dist_init(faulty_messages=[], messages_to_delay={'SCRIPT_CALL': 1.5})
    def test_rpc_script_timeout(self):
        if False:
            i = 10
            return i + 15
        next_rank = (self.rank + 1) % self.world_size
        dst_worker = worker_name(next_rank)
        expected_error = self.get_timeout_error_regex()
        with self.assertRaisesRegex(RuntimeError, expected_error):
            rpc.rpc_sync(dst_worker, my_script_func, args=(torch.tensor(1),), timeout=1)
        fut = rpc.rpc_async(dst_worker, my_script_func, args=(torch.tensor(1),), timeout=1)
        with self.assertRaisesRegex(RuntimeError, expected_error):
            fut.wait()
        fut = rpc.rpc_async(dst_worker, my_script_func, args=(torch.tensor(1),))
        fut.wait()
        rpc._set_rpc_timeout(0.001)
        fut = rpc.rpc_async(dst_worker, my_script_func, args=(torch.tensor(1),))
        with self.assertRaisesRegex(RuntimeError, expected_error):
            fut.wait()
        rpc._set_rpc_timeout(0.001)
        fut = rpc.rpc_async(dst_worker, my_script_func, args=(torch.tensor(1),), timeout=0)
        fut.wait()
        rpc._set_rpc_timeout(rpc.constants.DEFAULT_RPC_TIMEOUT_SEC)