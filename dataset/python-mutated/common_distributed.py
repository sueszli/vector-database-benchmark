import faulthandler
import logging
import multiprocessing
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import types
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from functools import partial, reduce, wraps
from io import StringIO
from typing import Dict, NamedTuple, Optional, Union
from unittest.mock import patch
import torch
import torch._dynamo.test_case
import torch.cuda.nccl
import torch.distributed as c10d
import torch.nn as nn
from torch.testing._internal.common_utils import FILE_SCHEMA, find_free_port, IS_SANDCASTLE, retry_on_connect_failures, skip_but_pass_in_sandcastle, skip_but_pass_in_sandcastle_if, TEST_WITH_ROCM, TEST_WITH_TSAN, TestCase
from torch.testing._internal.distributed.multi_threaded_pg import _install_threaded_pg, _uninstall_threaded_pg, ProcessLocalGroup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestSkip(NamedTuple):
    exit_code: int
    message: str
TEST_SKIPS = {'backend_unavailable': TestSkip(72, 'Skipped because distributed backend is not available.'), 'small_worldsize': TestSkip(73, 'Skipped due to small world size.'), 'odd_worldsize': TestSkip(87, 'Skipped due to odd world size.'), 'no_cuda': TestSkip(74, 'CUDA is not available.'), 'multi-gpu-1': TestSkip(75, 'Need at least 1 CUDA device'), 'multi-gpu-2': TestSkip(77, 'Need at least 2 CUDA devices'), 'multi-gpu-3': TestSkip(80, 'Need at least 3 CUDA devices'), 'multi-gpu-4': TestSkip(81, 'Need at least 4 CUDA devices'), 'multi-gpu-5': TestSkip(82, 'Need at least 5 CUDA devices'), 'multi-gpu-6': TestSkip(83, 'Need at least 6 CUDA devices'), 'multi-gpu-7': TestSkip(84, 'Need at least 7 CUDA devices'), 'multi-gpu-8': TestSkip(85, 'Need at least 8 CUDA devices'), 'nccl': TestSkip(76, 'c10d not compiled with NCCL support'), 'skipIfRocm': TestSkip(78, 'Test skipped for ROCm'), 'no_peer_access': TestSkip(79, 'Test skipped because no GPU peer access'), 'generic': TestSkip(86, 'Test skipped at subprocess level, look at subprocess log for skip reason'), 'importerror': TestSkip(88, 'Test skipped due to missing import')}

@dataclass
class DistTestCases:
    skip_collective = {}
    skip_collective['allgather_coalesced'] = {'nccl', 'mpi', 'ucc'}
    skip_collective['reduce'] = set()
    skip_collective['sendrecv anysource'] = {'nccl', 'ucc'}
    skip_collective['cpu barrier'] = {'nccl', 'ucc'}
    backend_feature = {}
    backend_feature['gpu'] = {'nccl', 'gloo', 'ucc'}
    backend_feature['cuda'] = {'nccl', 'gloo', 'ucc'}
    backend_feature['ddp'] = {'nccl', 'gloo', 'ucc'}
    backend_feature['subgroup'] = {'nccl', 'gloo', 'ucc'}
    backend_feature['plugin'] = set()

def skip_if_no_gpu(func):
    if False:
        for i in range(10):
            print('nop')
    'Skips if the world size exceeds the number of GPUs, ensuring that if the\n    test is run, each rank has its own GPU via ``torch.cuda.device(rank)``.'

    @wraps(func)
    def wrapper(*args, **kwargs):
        if False:
            return 10
        if not torch.cuda.is_available():
            sys.exit(TEST_SKIPS['no_cuda'].exit_code)
        world_size = int(os.environ['WORLD_SIZE'])
        if torch.cuda.device_count() < world_size:
            sys.exit(TEST_SKIPS[f'multi-gpu-{world_size}'].exit_code)
        return func(*args, **kwargs)
    return wrapper

def skip_if_small_worldsize(func):
    if False:
        for i in range(10):
            print('nop')

    @wraps(func)
    def wrapper(*args, **kwargs):
        if False:
            while True:
                i = 10
        if os.environ['BACKEND'] != 'mpi' and int(os.environ['WORLD_SIZE']) <= 2:
            sys.exit(TEST_SKIPS['small_worldsize'].exit_code)
        return func(*args, **kwargs)
    return wrapper

def skip_if_odd_worldsize(func):
    if False:
        return 10

    @wraps(func)
    def wrapper(*args, **kwargs):
        if False:
            while True:
                i = 10
        if os.environ['BACKEND'] != 'mpi' and int(os.environ['WORLD_SIZE']) % 2 == 1:
            sys.exit(TEST_SKIPS['odd_worldsize'].exit_code)
        return func(*args, **kwargs)
    return wrapper

def require_n_gpus_for_nccl_backend(n, backend):
    if False:
        while True:
            i = 10

    def decorator(func):
        if False:
            print('Hello World!')

        @wraps(func)
        def wrapper(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            if backend == 'nccl' and torch.cuda.device_count() < n:
                sys.exit(TEST_SKIPS[f'multi-gpu-{n}'].exit_code)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator

def import_transformers_or_skip():
    if False:
        return 10

    def decorator(func):
        if False:
            print('Hello World!')

        @wraps(func)
        def wrapper(*args, **kwargs):
            if False:
                print('Hello World!')
            try:
                from transformers import AutoModelForMaskedLM, BertConfig
                return func(*args, **kwargs)
            except ImportError:
                sys.exit(TEST_SKIPS['importerror'].exit_code)
        return wrapper
    return decorator

def skip_if_lt_x_gpu(x):
    if False:
        for i in range(10):
            print('nop')

    def decorator(func):
        if False:
            return 10

        @wraps(func)
        def wrapper(*args, **kwargs):
            if False:
                return 10
            if torch.cuda.is_available() and torch.cuda.device_count() >= x:
                return func(*args, **kwargs)
            sys.exit(TEST_SKIPS[f'multi-gpu-{x}'].exit_code)
        return wrapper
    return decorator

def nccl_skip_if_lt_x_gpu(backend, x):
    if False:
        return 10

    def decorator(func):
        if False:
            while True:
                i = 10

        @wraps(func)
        def wrapper(*args, **kwargs):
            if False:
                return 10
            if backend != 'nccl':
                return func(*args, **kwargs)
            if torch.cuda.is_available() and torch.cuda.device_count() >= x:
                return func(*args, **kwargs)
            sys.exit(TEST_SKIPS[f'multi-gpu-{x}'].exit_code)
        return wrapper
    return decorator

def verify_ddp_error_logged(model_DDP, err_substr):
    if False:
        for i in range(10):
            print('nop')
    ddp_logging_data = model_DDP._get_ddp_logging_data()
    assert 'iteration' in ddp_logging_data
    assert 'has_error' in ddp_logging_data
    assert 'error' in ddp_logging_data
    logging_err = ddp_logging_data['error']
    actual = err_substr if err_substr.find('\nException raised from ') == -1 else err_substr.split('\nException raised from ')[0]
    assert actual in logging_err, f'Did not find expected {actual} in ddp logging data error: {logging_err}'

def with_nccl_blocking_wait(func):
    if False:
        while True:
            i = 10
    '\n    Convenience decorator to set/unset NCCL_BLOCKING_WAIT flag. Note that use of\n    this decorator will override the setting of NCCL_ASYNC_ERROR_HANDLING for\n    the particular test. After the test, both NCCL_BLOCKING_WAIT and\n    NCCL_ASYNC_ERROR_HANDLING will be restored to their original values.\n    '

    @wraps(func)
    def wrapper(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        try:
            cached_nccl_async_error_handling: Union[str, None] = os.environ['NCCL_ASYNC_ERROR_HANDLING']
            del os.environ['NCCL_ASYNC_ERROR_HANDLING']
        except KeyError:
            cached_nccl_async_error_handling = None
        try:
            cached_nccl_blocking_wait: Union[str, None] = os.environ['NCCL_BLOCKING_WAIT']
        except KeyError:
            cached_nccl_blocking_wait = None
        finally:
            os.environ['NCCL_BLOCKING_WAIT'] = '1'
        try:
            ret = func(*args, **kwargs)
            return ret
        finally:
            if cached_nccl_async_error_handling is not None:
                os.environ['NCCL_ASYNC_ERROR_HANDLING'] = cached_nccl_async_error_handling
            if cached_nccl_blocking_wait is not None:
                os.environ['NCCL_BLOCKING_WAIT'] = cached_nccl_blocking_wait
    return wrapper

def with_dist_debug_levels(levels):
    if False:
        i = 10
        return i + 15
    '\n    Runs a test for each distributed debug level specified in levels.\n    '

    def decorator(func):
        if False:
            return 10

        @wraps(func)
        def wrapper(*args, **kwargs):
            if False:
                while True:
                    i = 10
            old_level = os.environ.get('TORCH_DISTRIBUTED_DEBUG', None)
            for level in levels:
                os.environ['TORCH_DISTRIBUTED_DEBUG'] = level
                c10d.set_debug_level_from_env()
                ret = func(*args, **kwargs)
                c10d.barrier()
                if old_level is not None:
                    os.environ['TORCH_DISTRIBUTED_DEBUG'] = old_level
            return ret
        return wrapper
    return decorator

def requires_gloo():
    if False:
        print('Hello World!')
    return skip_but_pass_in_sandcastle_if(not c10d.is_gloo_available(), 'c10d was not compiled with the Gloo backend')

def requires_nccl_version(version, msg):
    if False:
        while True:
            i = 10
    if not c10d.is_nccl_available():
        return skip_but_pass_in_sandcastle('c10d was not compiled with the NCCL backend')
    else:
        return skip_but_pass_in_sandcastle_if(torch.cuda.nccl.version() < version, 'Requires NCCL version greater than or equal to: {}, found: {}, reason: {}'.format(version, torch.cuda.nccl.version(), msg))

def requires_nccl():
    if False:
        i = 10
        return i + 15
    return skip_but_pass_in_sandcastle_if(not c10d.is_nccl_available(), 'c10d was not compiled with the NCCL backend')

def requires_ucc():
    if False:
        print('Hello World!')
    return skip_but_pass_in_sandcastle_if(not c10d.is_ucc_available(), 'c10d was not compiled with the UCC backend')

def requires_mpi():
    if False:
        for i in range(10):
            print('nop')
    return skip_but_pass_in_sandcastle_if(not c10d.is_mpi_available(), 'c10d was not compiled with the MPI backend')

def skip_if_rocm(func):
    if False:
        print('Hello World!')
    'Skips a test for ROCm'
    func.skip_if_rocm = True

    @wraps(func)
    def wrapper(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if not TEST_WITH_ROCM:
            return func(*args, **kwargs)
        sys.exit(TEST_SKIPS['skipIfRocm'].exit_code)
    return wrapper

def skip_if_win32():
    if False:
        while True:
            i = 10
    return skip_but_pass_in_sandcastle_if(sys.platform == 'win32', 'This unit test case is not supported on Windows platform')

@retry_on_connect_failures
def create_tcp_store(addr='localhost', world_size=1, is_master=True, timeout=timedelta(minutes=5), wait_for_workers=True, jit_class=False, use_libuv=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Creates a TCP store. Retries if the chosen port is already in use.\n    '
    port = find_free_port()
    if jit_class:
        timeout_millisecond = int(timeout / timedelta(milliseconds=1))
        return torch.classes.dist_c10d.TCPStore(addr, port, world_size, is_master, timeout_millisecond)
    else:
        return c10d.TCPStore(addr, port, world_size, is_master, wait_for_workers=wait_for_workers, use_libuv=use_libuv)
if TEST_WITH_TSAN:
    TIMEOUT_DEFAULT = 500
else:
    TIMEOUT_DEFAULT = int(os.getenv('DISTRIBUTED_TESTS_DEFAULT_TIMEOUT', '300'))
TIMEOUT_OVERRIDE = {'test_ddp_uneven_inputs': 400}
if TEST_WITH_ROCM:
    TIMEOUT_OVERRIDE['test_join_kwargs'] = 200

def create_device(interface=None):
    if False:
        i = 10
        return i + 15
    if sys.platform == 'win32' or interface is None:
        return c10d.ProcessGroupGloo.create_device(hostname='127.0.0.1')
    else:
        return c10d.ProcessGroupGloo.create_device(interface=interface)

def get_timeout(test_id) -> int:
    if False:
        print('Hello World!')
    return TIMEOUT_OVERRIDE.get(test_id.split('.')[-1], TIMEOUT_DEFAULT)

@contextmanager
def captured_output():
    if False:
        return 10
    (new_out, new_err) = (StringIO(), StringIO())
    (old_out, old_err) = (sys.stdout, sys.stderr)
    try:
        (sys.stdout, sys.stderr) = (new_out, new_err)
        yield (sys.stdout, sys.stderr)
    finally:
        (sys.stdout, sys.stderr) = (old_out, old_err)

def simple_sparse_reduce_tests(rank: int, world_size: int, num_inputs: int=1):
    if False:
        return 10
    '\n    Generate a number of basic test cases for sparse reduction.\n    These cover tensors with a varying number of sparse dimensions and a varying\n    number of dense dimensions. The only reduction operation we support is sum.\n    '

    def generate(rank: int, world_size: int, sparse_dims: int=1, dense_dims: int=0):
        if False:
            return 10
        indices = torch.reshape(torch.arange(rank + 1), (1, rank + 1))
        shape = [world_size] + [2 for _ in range(dense_dims)]
        for _ in range(sparse_dims - 1):
            indices = torch.cat((indices, torch.zeros(1, rank + 1)))
            shape.append(world_size)
        values = torch.ones([rank + 1] + [2 for _ in range(dense_dims)])
        return torch.sparse_coo_tensor(indices, values, shape)

    def compute_sum(fn, world_size: int):
        if False:
            print('Hello World!')
        return reduce(lambda a, b: a + b, [fn(rank, world_size) for rank in range(world_size)])
    return [([fn(num_inputs * rank + i, num_inputs * world_size) for i in range(num_inputs)], [compute_sum(fn, num_inputs * world_size) for i in range(num_inputs)]) for fn in [partial(generate, sparse_dims=1), partial(generate, sparse_dims=2), partial(generate, sparse_dims=3), partial(generate, dense_dims=1), partial(generate, dense_dims=2), partial(generate, dense_dims=3)]]

def init_multigpu_helper(world_size: int, backend: str):
    if False:
        print('Hello World!')
    'Multigpu tests are designed to simulate the multi nodes with multi\n    GPUs on each node. Nccl backend requires equal #GPUs in each process.\n    On a single node, all visible GPUs are evenly\n    divided to subsets, each process only uses a subset.\n    '
    nGPUs = torch.cuda.device_count()
    visible_devices = range(nGPUs)
    if backend == 'nccl':
        os.environ['NCCL_MAX_NRINGS'] = '1'
    nGPUs_per_process = 1
    if world_size > nGPUs:
        nGPUs_per_process = nGPUs // world_size
    rank_to_GPU = {i: list(visible_devices[i * nGPUs_per_process:(i + 1) * nGPUs_per_process]) for i in range(world_size)}
    return rank_to_GPU
tmp_dir: Optional[tempfile.TemporaryDirectory] = None

def initialize_temp_directories(init_method: Optional[str]=None) -> None:
    if False:
        print('Hello World!')
    global tmp_dir
    tmp_dir = tempfile.TemporaryDirectory()
    os.environ['TEMP_DIR'] = tmp_dir.name
    os.mkdir(os.path.join(tmp_dir.name, 'barrier'))
    os.mkdir(os.path.join(tmp_dir.name, 'test_dir'))
    init_dir_path = os.path.join(tmp_dir.name, 'init_dir')
    os.mkdir(init_dir_path)
    if init_method is not None:
        os.environ['INIT_METHOD'] = init_method
    else:
        os.environ['INIT_METHOD'] = FILE_SCHEMA + os.path.join(init_dir_path, 'shared_init_file')

def cleanup_temp_dir() -> None:
    if False:
        for i in range(10):
            print('nop')
    if tmp_dir is not None:
        tmp_dir.cleanup()
DEFAULT_WORLD_SIZE = 4

class MultiProcessTestCase(TestCase):
    MAIN_PROCESS_RANK = -1
    TEST_ERROR_EXIT_CODE = 10

    def _should_stop_test_suite(self) -> bool:
        if False:
            i = 10
            return i + 15
        return False

    @property
    def world_size(self) -> int:
        if False:
            i = 10
            return i + 15
        return DEFAULT_WORLD_SIZE

    def join_or_run(self, fn):
        if False:
            while True:
                i = 10

        @wraps(fn)
        def wrapper(self):
            if False:
                i = 10
                return i + 15
            if self.rank == self.MAIN_PROCESS_RANK:
                self._join_processes(fn)
            else:
                fn()
        return types.MethodType(wrapper, self)

    def __init__(self, method_name: str='runTest') -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(method_name)
        fn = getattr(self, method_name)
        setattr(self, method_name, self.join_or_run(fn))

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        super().setUp()
        self.skip_return_code_checks = []
        self.processes = []
        self.rank = self.MAIN_PROCESS_RANK
        self.file_name = tempfile.NamedTemporaryFile(delete=False).name
        self.pid_to_pipe = {}

    def tearDown(self) -> None:
        if False:
            print('Hello World!')
        super().tearDown()
        for p in self.processes:
            p.terminate()
        self.processes = []

    def _current_test_name(self) -> str:
        if False:
            i = 10
            return i + 15
        return self.id().split('.')[-1]

    def _start_processes(self, proc) -> None:
        if False:
            print('Hello World!')
        self.processes = []
        for rank in range(int(self.world_size)):
            (parent_conn, child_conn) = torch.multiprocessing.Pipe()
            process = proc(target=self.__class__._run, name='process ' + str(rank), args=(rank, self._current_test_name(), self.file_name, child_conn))
            process.start()
            logger.info('Started process %s with pid %s', rank, process.pid)
            self.pid_to_pipe[process.pid] = parent_conn
            self.processes.append(process)

    def _spawn_processes(self) -> None:
        if False:
            while True:
                i = 10
        proc = torch.multiprocessing.get_context('spawn').Process
        self._start_processes(proc)

    class Event(Enum):
        GET_TRACEBACK = 1

    @staticmethod
    def _event_listener(parent_pipe, signal_pipe, rank: int):
        if False:
            i = 10
            return i + 15
        logger.info('Starting event listener thread for rank %s', rank)
        while True:
            ready_pipes = multiprocessing.connection.wait([parent_pipe, signal_pipe])
            if parent_pipe in ready_pipes:
                if parent_pipe.closed:
                    logger.info('Pipe closed for process %s, stopping event listener thread', rank)
                    return
                event = parent_pipe.recv()
                logger.info('Received event %s on process %s', event, rank)
                if event == MultiProcessTestCase.Event.GET_TRACEBACK:
                    with tempfile.NamedTemporaryFile(mode='r+') as tmp_file:
                        faulthandler.dump_traceback(tmp_file)
                        tmp_file.flush()
                        tmp_file.seek(0)
                        parent_pipe.send(tmp_file.read())
                        logger.info('Process %s sent traceback', rank)
            if signal_pipe in ready_pipes:
                return

    @classmethod
    def _run(cls, rank: int, test_name: str, file_name: str, parent_pipe) -> None:
        if False:
            return 10
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name
        self.run_test(test_name, parent_pipe)

    def run_test(self, test_name: str, parent_pipe) -> None:
        if False:
            while True:
                i = 10
        (signal_recv_pipe, signal_send_pipe) = torch.multiprocessing.Pipe(duplex=False)
        event_listener_thread = threading.Thread(target=MultiProcessTestCase._event_listener, args=(parent_pipe, signal_recv_pipe, self.rank), daemon=True)
        event_listener_thread.start()
        if sys.platform != 'win32' and sys.platform != 'darwin':
            torch._C._set_print_stack_traces_on_fatal_signal(True)
        os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '1'
        try:
            getattr(self, test_name)()
        except unittest.SkipTest as se:
            logger.info('Process %s skipping test %s for following reason: %s', self.rank, test_name, str(se))
            sys.exit(TEST_SKIPS['generic'].exit_code)
        except Exception as e:
            logger.error('Caught exception: \n%s exiting process %s with exit code: %s', traceback.format_exc(), self.rank, MultiProcessTestCase.TEST_ERROR_EXIT_CODE)
            parent_pipe.send(traceback.format_exc())
            sys.exit(MultiProcessTestCase.TEST_ERROR_EXIT_CODE)
        finally:
            if signal_send_pipe is not None:
                signal_send_pipe.send(None)
            assert event_listener_thread is not None
            event_listener_thread.join()
            parent_pipe.close()

    def _get_timedout_process_traceback(self) -> None:
        if False:
            print('Hello World!')
        pipes = []
        for (i, process) in enumerate(self.processes):
            if process.exitcode is None:
                pipe = self.pid_to_pipe[process.pid]
                try:
                    pipe.send(MultiProcessTestCase.Event.GET_TRACEBACK)
                    pipes.append((i, pipe))
                except ConnectionError as e:
                    logger.error('Encountered error while trying to get traceback for process %s: %s', i, e)
        for (rank, pipe) in pipes:
            try:
                if pipe.poll(5):
                    if pipe.closed:
                        logger.info('Pipe closed for process %s, cannot retrieve traceback', rank)
                        continue
                    traceback = pipe.recv()
                    logger.error('Process %s timed out with traceback: \n\n%s', rank, traceback)
                else:
                    logger.error('Could not retrieve traceback for timed out process: %s', rank)
            except ConnectionError as e:
                logger.error('Encountered error while trying to get traceback for process %s: %s', rank, e)

    def _join_processes(self, fn) -> None:
        if False:
            return 10
        timeout = get_timeout(self.id())
        start_time = time.time()
        subprocess_error = False
        try:
            while True:
                for (i, p) in enumerate(self.processes):
                    if p.exitcode == MultiProcessTestCase.TEST_ERROR_EXIT_CODE:
                        print(f'Process {i} terminated with exit code {p.exitcode}, terminating remaining processes.')
                        active_children = torch.multiprocessing.active_children()
                        for ac in active_children:
                            ac.terminate()
                        subprocess_error = True
                        break
                if subprocess_error:
                    break
                if all((p.exitcode is not None for p in self.processes)):
                    break
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    self._get_timedout_process_traceback()
                    print(f'Timing out after {timeout} seconds and killing subprocesses.')
                    for p in self.processes:
                        p.terminate()
                    break
                time.sleep(0.1)
            elapsed_time = time.time() - start_time
            if fn in self.skip_return_code_checks:
                self._check_no_test_errors(elapsed_time)
            else:
                self._check_return_codes(elapsed_time)
        finally:
            for pipe in self.pid_to_pipe.values():
                pipe.close()

    def _check_no_test_errors(self, elapsed_time) -> None:
        if False:
            print('Hello World!')
        "\n        Checks that we didn't have any errors thrown in the child processes.\n        "
        for (i, p) in enumerate(self.processes):
            if p.exitcode is None:
                raise RuntimeError(f'Process {i} timed out after {elapsed_time} seconds')
            self.assertNotEqual(self.TEST_ERROR_EXIT_CODE, p.exitcode)

    def _check_return_codes(self, elapsed_time) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Checks that the return codes of all spawned processes match, and skips\n        tests if they returned a return code indicating a skipping condition.\n        '
        if not self.processes:
            logger.warning('Note: no subprocesses were spawned, test was likely skipped.')
            return
        first_process = self.processes[0]
        errored_processes = [(i, p) for (i, p) in enumerate(self.processes) if p.exitcode == MultiProcessTestCase.TEST_ERROR_EXIT_CODE]
        if errored_processes:
            error = ''
            for (i, process) in errored_processes:
                error_message = self.pid_to_pipe[process.pid].recv()
                error += 'Process {} exited with error code {} and exception:\n{}\n'.format(i, MultiProcessTestCase.TEST_ERROR_EXIT_CODE, error_message)
            raise RuntimeError(error)
        for (i, p) in enumerate(self.processes):
            if p.exitcode is None:
                raise RuntimeError(f'Process {i} terminated or timed out after {elapsed_time} seconds')
            self.assertEqual(p.exitcode, first_process.exitcode, msg='Expect process {} exit code to match Process 0 exit code of {}, but got {}'.format(i, first_process.exitcode, p.exitcode))
        for skip in TEST_SKIPS.values():
            if first_process.exitcode == skip.exit_code:
                if IS_SANDCASTLE:
                    logger.info('Skipping %s on sandcastle for the following reason: %s', self.id(), skip.message)
                    return
                else:
                    raise unittest.SkipTest(skip.message)
        self.assertEqual(first_process.exitcode, 0, msg=f'Expected zero exit code but got {first_process.exitcode} for pid: {first_process.pid}')

    @property
    def is_master(self) -> bool:
        if False:
            while True:
                i = 10
        return self.rank == 0
EFA_PROBE_RESULT = None

def has_efa() -> bool:
    if False:
        print('Hello World!')
    '\n    If shell command `fi_info -p efa -t FI_EP_RDM` returns exit code 0 then we assume that the machine has\n    Libfabric EFA interfaces and EFA software components installed,\n    see https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html.\n    '
    global EFA_PROBE_RESULT
    if EFA_PROBE_RESULT is not None:
        return EFA_PROBE_RESULT
    try:
        EFA_PROBE_RESULT = subprocess.run(['fi_info', '-p', 'efa', '-t', 'FI_EP_RDM'], check=False).returncode == 0
    except FileNotFoundError:
        EFA_PROBE_RESULT = False
    return EFA_PROBE_RESULT

def tp_transports():
    if False:
        for i in range(10):
            print('nop')
    '\n    If the machine has Libfabric EFA interfaces and EFA software components installed it may cause\n    \'RuntimeError: In operator() at tensorpipe/common/ibv.h:172 "": Operation not supported\' if tensorpipe\n    uses InfiniBand transport, so we exclude it from tensorpipe transports,\n    see https://github.com/pytorch/pytorch/issues/73885 and https://github.com/pytorch/pytorch/issues/65022\n    '
    return ['shm', 'uv'] if has_efa() else None

def spawn_threads_and_init_comms(func=None, timeout=TIMEOUT_DEFAULT, world_size=DEFAULT_WORLD_SIZE):
    if False:
        for i in range(10):
            print('nop')
    '\n    Wrapper to use with a test method\n    '
    if func is None:
        return partial(spawn_threads_and_init_comms, timeout=timeout, world_size=world_size)

    def _run_test_method_with_multi_threads(world_size, callback):
        if False:
            for i in range(10):
                print('nop')
        world = _install_threaded_pg()
        global_store = c10d.HashStore()

        def world_is_valid():
            if False:
                for i in range(10):
                    print('nop')
            return world == c10d.distributed_c10d._world

        def worker(rank, world_pg, store):
            if False:
                for i in range(10):
                    print('nop')
            c10d.init_process_group(backend='threaded', rank=rank, world_size=world_size, store=store)
            try:
                callback()
            except BaseException as ex:
                MultiThreadedTestCase.exception_queue.put((rank, sys.exc_info()))
                ProcessLocalGroup.exception_handle(ex)
            finally:
                if world_is_valid():
                    c10d.destroy_process_group()
        threads = []
        for rank in range(world_size):
            t = threading.Thread(target=worker, args=(rank, world, global_store))
            t.start()
            threads.append(t)
        return threads

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if False:
            print('Hello World!')
        threads = _run_test_method_with_multi_threads(world_size, lambda : func(self, *args, **kwargs))
        MultiThreadedTestCase._join_threads(threads, func)
    return wrapper

class MultiThreadedTestCase(TestCase):
    """
    Test runner that runs all tests with the in-proc process group using
    multiple threads with the threaded process group.

    Each test spawns world_size threads and run the test method in each thread.

    Difference from regular MultiProcess test runner:
    Must explicitly defines SetUp and call self._spawn_threads() to run the tests.
    Cannot use setUp / tearDown (must use perThreadSetup / perThreadShutdown)
        to set up / tear down each thread when running each test.
    No global state possible
        How bad of a limitation is this?
    """
    exception_queue = queue.Queue()
    MAIN_THREAD_RANK = -1

    def join_or_run(self, fn):
        if False:
            for i in range(10):
                print('nop')

        @wraps(fn)
        def wrapper(self):
            if False:
                return 10
            if self.rank == self.MAIN_THREAD_RANK:
                self._join_threads(self.threads, fn)
            else:
                fn()
        return types.MethodType(wrapper, self)

    def __init__(self, method_name: str='runTest') -> None:
        if False:
            return 10
        super().__init__(method_name)
        test_fn = getattr(self, method_name, None)
        setattr(self, method_name, self.join_or_run(test_fn))

    def perThreadSetUp(self):
        if False:
            while True:
                i = 10
        pass

    def perThreadTearDown(self):
        if False:
            print('Hello World!')
        pass

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        '\n        setUp only set up things in the main thread, if you want to configure things\n        in the spawned threads, use perThreadSetUp\n        '
        super().setUp()
        self.rank = self.MAIN_THREAD_RANK
        self.threads = []
        os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '1'

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        '\n        tearDown only set up things in the main thread, if you want to configure things\n        in the spawned threads, use perThreadTearDown\n        '
        super().tearDown()
        self.threads = []

    def _spawn_threads(self):
        if False:
            return 10
        '\n        class method to spawn threads and run test, use this method in the SetUp of your TestCase\n        '
        test_name = self._current_test_name
        world = _install_threaded_pg()
        self.__class__.global_store = c10d.HashStore()

        def world_is_valid():
            if False:
                i = 10
                return i + 15
            return world == c10d.distributed_c10d._world
        if not world_is_valid():
            raise RuntimeError('Invalid world')
        for rank in range(self.world_size):
            t = threading.Thread(target=self.__class__._run, args=(test_name, rank, self.world_size))
            t.start()
            self.threads.append(t)

    @classmethod
    def _run(cls, test_name, rank, world_size):
        if False:
            return 10
        self = cls(test_name)
        self.rank = rank
        if hasattr(self, '_tls'):
            self._tls = threading.local()
            self._tls.precision = TestCase._precision
            self._tls.rel_tol = TestCase._rel_tol
        self.run_test_with_threaded_pg(test_name, rank, world_size)

    def run_test_with_threaded_pg(self, test_name, rank, world_size):
        if False:
            while True:
                i = 10
        '\n        Run the current test associated with `test_name` using the threaded process group.\n        '
        c10d.init_process_group(backend='threaded', rank=rank, world_size=world_size, store=self.__class__.global_store)
        self.perThreadSetUp()
        try:
            getattr(self, test_name)()
        except BaseException as ex:
            self.exception_queue.put((rank, sys.exc_info()))
            ProcessLocalGroup.exception_handle(ex)
        finally:
            c10d.destroy_process_group()
            self.perThreadTearDown()

    @classmethod
    def _join_threads(cls, threads, fn):
        if False:
            return 10
        timeout = TIMEOUT_DEFAULT
        try:
            for (idx, thread) in enumerate(threads):
                thread.join(max(0, timeout))
                if thread.is_alive():
                    MultiThreadedTestCase.exception_queue.put((idx, (TimeoutError, TimeoutError(f'Rank failed to join in under {timeout} seconds'), None)))
            ProcessLocalGroup.reset()
            failed_ranks = []
            while not cls.exception_queue.empty():
                failure = cls.exception_queue.get()
                failed_ranks.append(failure)
        finally:
            _uninstall_threaded_pg()
        cls._check_return_codes(failed_ranks, timeout, fn)

    @classmethod
    def _check_return_codes(cls, failed_ranks, timeout, fn):
        if False:
            while True:
                i = 10
        error_msg = ''
        skip_code = -1
        for (rank, exc_info) in failed_ranks:
            exc = exc_info[1]
            if isinstance(exc, unittest.SkipTest):
                logger.info('Thread %s skipping test %s for following reason: %s', rank, fn, str(exc))
                if skip_code < 0:
                    skip_code = TEST_SKIPS['generic'].exit_code
            elif isinstance(exc, TimeoutError):
                msg = f'Thread {rank} terminated or timed out after {timeout} seconds\n'
                logger.error(msg)
                raise RuntimeError(msg)
            elif isinstance(exc, Exception):
                msg = ''.join(traceback.format_exception(*exc_info))
                logger.error('Caught exception: \n%s exiting thread %s', msg, rank)
                error_msg += f'Thread {rank} exited with exception:\n{msg}\n'
            elif isinstance(exc, SystemExit):
                if type(exc.code) == int and skip_code < 0:
                    skip_code = exc.code
        if len(error_msg) > 0:
            raise RuntimeError(error_msg)
        if skip_code > 0:
            for skip in TEST_SKIPS.values():
                if skip_code == skip.exit_code:
                    if IS_SANDCASTLE:
                        logger.info('Skipping %s on sandcastle for the following reason: %s', fn, skip.message)
                        return
                    else:
                        raise unittest.SkipTest(skip.message)

    @property
    def world_size(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return DEFAULT_WORLD_SIZE

    @property
    def _current_test_name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.id().split('.')[-1]

    def assertEqualOnRank(self, x, y, msg=None, *, rank=0):
        if False:
            for i in range(10):
                print('nop')
        '\n        The reason why we have this util function instead of\n        self.assertEqual is all threads are sharing one CPU RNG\n        so the assertion result is only reliable on rank 0\n        '
        if self.rank == rank:
            self.assertEqual(x, y, msg)

    def assertNotEqualOnRank(self, x, y, msg=None, *, rank=0):
        if False:
            print('Hello World!')
        if self.rank == rank:
            self.assertNotEqual(x, y)

class SaveForwardInputsModule(nn.Module):

    def __init__(self, forward_inputs: Dict[nn.Module, torch.Tensor], cast_forward_inputs: bool) -> None:
        if False:
            return 10
        super().__init__()
        self.l = nn.Linear(100, 100)
        self.forward_inputs = forward_inputs
        self.cast_forward_inputs = cast_forward_inputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if False:
            print('Hello World!')
        self.forward_inputs[self] = x
        return self.l(x.to(self.l.weight.dtype) if self.cast_forward_inputs else x)

class SaveForwardInputsModel(nn.Module):

    def __init__(self, forward_inputs: Dict[nn.Module, torch.Tensor], cast_forward_inputs: bool) -> None:
        if False:
            return 10
        super().__init__()
        self.c1 = SaveForwardInputsModule(forward_inputs, cast_forward_inputs)
        self.c2 = SaveForwardInputsModule(forward_inputs, cast_forward_inputs)
        self.forward_inputs = forward_inputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if False:
            print('Hello World!')
        self.forward_inputs[self] = x
        return self.c2(self.c1(x))

@contextmanager
def _dynamo_dist_per_rank_init(rank, world_size, init_pg=True):
    if False:
        return 10
    torch.cuda.set_device(rank)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '6789'
    if init_pg:
        c10d.init_process_group('nccl', rank=rank, world_size=world_size)
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    try:
        yield
    finally:
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        if init_pg:
            c10d.destroy_process_group()

class DynamoDistributedSingleProcTestCase(torch._dynamo.test_case.TestCase):
    """
    Test harness for single-process dynamo distributed tests,
    initializes dist process group.

    Prefer this for simple tests, as it's easier to debug.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        super().setUpClass()
        cls._exit_stack.enter_context(patch.dict(os.environ, {'MASTER_ADDR': 'localhost', 'MASTER_PORT': '12355'}))
        cls.rank = 0
        cls.device = f'cuda:{cls.rank}'
        cls.device_ids = None if 'cuda' in cls.device else [cls.rank]
        c10d.init_process_group('nccl', rank=cls.rank, world_size=1)

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        c10d.destroy_process_group()
        super().tearDownClass()

class DynamoDistributedMultiProcTestCase(MultiProcessTestCase):
    """
    Use this for tests that actually run on multiple GPUs.

    Decorate tests with @skip_if_lt_x_gpu(ngpu)

    Note: MultiProcTestCase spawns processes per test and is slow.
    Prefer MultiThreadedTestCase for most tests. Perhaps use this one
    sparingly for integration tests.
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        if False:
            print('Hello World!')
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def world_size(self) -> int:
        if False:
            i = 10
            return i + 15
        return torch.cuda.device_count()

    @classmethod
    def _run(cls, rank: int, test_name: str, file_name: str, parent_pipe) -> None:
        if False:
            while True:
                i = 10
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name
        self.run_test(test_name, parent_pipe)