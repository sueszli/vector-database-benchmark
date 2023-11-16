import multiprocessing as mp
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid
from contextlib import closing
from unittest import mock
from unittest.mock import Mock, patch
import torch.distributed.run as launch
from torch.distributed.elastic.agent.server.api import RunResult, WorkerState
from torch.distributed.elastic.multiprocessing.errors import ChildFailedError
from torch.distributed.elastic.rendezvous.etcd_server import EtcdServer
from torch.distributed.elastic.utils import get_socket_with_port
from torch.distributed.elastic.utils.distributed import get_free_port
from torch.testing._internal.common_utils import TEST_WITH_DEV_DBG_ASAN, skip_but_pass_in_sandcastle_if

def launch_in_proc(args):
    if False:
        for i in range(10):
            print('nop')
    launch.main(args)

def path(script):
    if False:
        print('Hello World!')
    return os.path.join(os.path.dirname(__file__), script)

def get_child_pids(pid):
    if False:
        i = 10
        return i + 15
    pgrep = subprocess.Popen(args=f'pgrep -P {pid}', shell=True, stdout=subprocess.PIPE)
    pgrep.wait()
    out = pgrep.stdout.read().decode('utf-8').rstrip().split('\n')
    pids = []
    for pid in out:
        if pid:
            pids.append(int(pid))
    return pids

def pid_exists(pid):
    if False:
        return 10
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

class MockException(Exception):
    pass

class ElasticLaunchTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        cls._etcd_server = EtcdServer()
        cls._etcd_server.start()
        cls._etcd_endpoint = cls._etcd_server.get_endpoint()

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        cls._etcd_server.stop()

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.test_dir = tempfile.mkdtemp()
        for env in os.environ.keys():
            if env.startswith('PET_'):
                del os.environ[env]
        os.environ['TEST_SENTINEL_PARENT'] = 'FOOBAR'

    def tearDown(self):
        if False:
            while True:
                i = 10
        shutil.rmtree(self.test_dir)

    def test_launch_user_script_python(self):
        if False:
            while True:
                i = 10
        self._test_launch_user_script_python()

    def _test_launch_user_script_python(self):
        if False:
            while True:
                i = 10
        run_id = str(uuid.uuid4().int)
        nnodes = 1
        nproc_per_node = 4
        world_size = nnodes * nproc_per_node
        args = [f'--nnodes={nnodes}', f'--nproc-per-node={nproc_per_node}', '--rdzv-backend=etcd', f'--rdzv-endpoint={self._etcd_endpoint}', f'--rdzv-id={run_id}', '--monitor-interval=1', '--start-method=spawn', path('bin/test_script.py'), f'--touch-file-dir={self.test_dir}']
        launch.main(args)
        self.assertSetEqual({str(i) for i in range(world_size)}, set(os.listdir(self.test_dir)))

    def test_launch_user_script_python_caffe2_bc(self):
        if False:
            for i in range(10):
                print('nop')
        nnodes = 1
        nproc_per_node = 4
        world_size = nnodes * nproc_per_node
        sock = get_socket_with_port()
        with closing(sock):
            master_port = sock.getsockname()[1]
        args = [f'--nnodes={nnodes}', f'--nproc-per-node={nproc_per_node}', '--monitor-interval=1', '--start-method=spawn', '--master-addr=localhost', f'--master-port={master_port}', '--node-rank=0', path('bin/test_script.py'), f'--touch-file-dir={self.test_dir}']
        launch.main(args)
        self.assertSetEqual({str(i) for i in range(world_size)}, set(os.listdir(self.test_dir)))

    @skip_but_pass_in_sandcastle_if(TEST_WITH_DEV_DBG_ASAN, 'test incompatible with dev/dbg asan')
    def test_launch_user_script_bash(self):
        if False:
            return 10
        run_id = str(uuid.uuid4().int)
        nnodes = 1
        nproc_per_node = 4
        world_size = nnodes * nproc_per_node
        args = [f'--nnodes={nnodes}', f'--nproc-per-node={nproc_per_node}', '--rdzv-backend=etcd', f'--rdzv-endpoint={self._etcd_endpoint}', f'--rdzv-id={run_id}', '--monitor-interval=1', '--start-method=spawn', '--no-python']
        script_args = [path('bin/test_script.sh'), f'{self.test_dir}']
        with self.assertRaises(ValueError):
            launch.main(args + ['--module'] + script_args)
        launch.main(args + script_args)
        self.assertSetEqual({str(i) for i in range(world_size)}, set(os.listdir(self.test_dir)))

    @skip_but_pass_in_sandcastle_if(TEST_WITH_DEV_DBG_ASAN, 'test incompatible with dev/dbg asan')
    def test_launch_user_script_default_nproc(self):
        if False:
            i = 10
            return i + 15
        run_id = str(uuid.uuid4().int)
        nnodes = 1
        world_size = 1
        args = [f'--nnodes={nnodes}', '--rdzv-backend=etcd', f'--rdzv-endpoint={self._etcd_endpoint}', f'--rdzv-id={run_id}', '--monitor-interval=1', '--start-method=spawn', '--no-python']
        script_args = [path('bin/test_script.sh'), f'{self.test_dir}']
        with self.assertRaises(ValueError):
            launch.main(args + ['--module'] + script_args)
        launch.main(args + script_args)
        self.assertSetEqual({str(i) for i in range(world_size)}, set(os.listdir(self.test_dir)))

    @skip_but_pass_in_sandcastle_if(TEST_WITH_DEV_DBG_ASAN, 'test incompatible with dev/dbg asan')
    def test_launch_with_env_vars(self):
        if False:
            i = 10
            return i + 15
        run_id = str(uuid.uuid4().int)
        nnodes = 1
        nproc_per_node = 4
        world_size = nnodes * nproc_per_node
        os.environ['PET_NNODES'] = str(nnodes)
        os.environ['PET_NPROC_PER_NODE'] = str(nproc_per_node)
        os.environ['PET_RDZV_BACKEND'] = 'etcd'
        os.environ['PET_RDZV_ENDPOINT'] = self._etcd_endpoint
        os.environ['PET_RDZV_ID'] = run_id
        os.environ['PET_MONITOR_INTERVAL'] = '1'
        os.environ['PET_START_METHOD'] = 'spawn'
        os.environ['PET_NO_PYTHON'] = '1'
        script_args = [path('bin/test_script.sh'), f'{self.test_dir}']
        with self.assertRaises(ValueError):
            os.environ['PET_MODULE'] = '1'
            launch.main(script_args)
        os.environ['PET_MODULE'] = '0'
        launch.main(script_args)
        self.assertSetEqual({str(i) for i in range(world_size)}, set(os.listdir(self.test_dir)))

    def _test_nproc_launch_configuration(self, nproc_type, expected_number):
        if False:
            i = 10
            return i + 15
        run_id = str(uuid.uuid4().int)
        nnodes = 1
        args = [f'--nnodes={nnodes}', f'--nproc-per-node={nproc_type}', '--rdzv-backend=etcd', f'--rdzv-endpoint={self._etcd_endpoint}', f'--rdzv-id={run_id}', '--monitor-interval=1', '--start-method=spawn', '--no-python']
        script_args = [path('bin/test_script.sh'), f'{self.test_dir}']
        launch.main(args + script_args)
        world_size = nnodes * expected_number
        self.assertSetEqual({str(i) for i in range(world_size)}, set(os.listdir(self.test_dir)))

    @skip_but_pass_in_sandcastle_if(TEST_WITH_DEV_DBG_ASAN, 'test incompatible with dev/dbg asan')
    def test_nproc_launch_auto_configurations(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_nproc_launch_configuration('auto', os.cpu_count())

    @skip_but_pass_in_sandcastle_if(TEST_WITH_DEV_DBG_ASAN, 'test incompatible with dev/dbg asan')
    def test_nproc_launch_number_configurations(self):
        if False:
            return 10
        self._test_nproc_launch_configuration('4', 4)

    @skip_but_pass_in_sandcastle_if(TEST_WITH_DEV_DBG_ASAN, 'test incompatible with dev/dbg asan')
    def test_nproc_launch_unknown_configurations(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(ValueError):
            self._test_nproc_launch_configuration('unknown', 4)

    @skip_but_pass_in_sandcastle_if(TEST_WITH_DEV_DBG_ASAN, 'test incompatible with dev/dbg asan')
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=3)
    def test_nproc_gpu_launch_configurations(self, _mock1, _mock2):
        if False:
            for i in range(10):
                print('nop')
        self._test_nproc_launch_configuration('auto', 3)
        self._test_nproc_launch_configuration('gpu', 3)

    @skip_but_pass_in_sandcastle_if(TEST_WITH_DEV_DBG_ASAN, 'test incompatible with dev/dbg asan')
    def test_launch_elastic(self):
        if False:
            i = 10
            return i + 15
        run_id = str(uuid.uuid4().int)
        min_nodes = 1
        max_nodes = 2
        nproc_per_node = 4
        world_size = nproc_per_node
        args = [f'--nnodes={min_nodes}:{max_nodes}', f'--nproc-per-node={nproc_per_node}', '--rdzv-backend=etcd', f'--rdzv-endpoint={self._etcd_endpoint}', f'--rdzv-id={run_id}', '--monitor-interval=1', '--start-method=spawn', path('bin/test_script.py'), f'--touch-file-dir={self.test_dir}']
        launch.main(args)
        self.assertSetEqual({str(i) for i in range(world_size)}, set(os.listdir(self.test_dir)))

    @mock.patch('torch.distributed.elastic.events.record')
    @skip_but_pass_in_sandcastle_if(TEST_WITH_DEV_DBG_ASAN, 'test incompatible with dev/dbg asan')
    def test_launch_elastic_worker_raise_exception(self, record_mock):
        if False:
            i = 10
            return i + 15
        '\n        Asserts that when the worker program fails and lancher raieses exception\n        to indicate that worker process failed\n\n        '
        run_id = str(uuid.uuid4().int)
        min_nodes = 1
        max_nodes = 2
        nproc_per_node = 4
        args = [f'--nnodes={min_nodes}:{max_nodes}', f'--nproc-per-node={nproc_per_node}', '--rdzv-backend=etcd', f'--rdzv-endpoint={self._etcd_endpoint}', f'--rdzv-id={run_id}', '--monitor-interval=1', '--max-restarts=0', '--start-method=spawn', path('bin/test_script.py'), '--fail']
        with self.assertRaises(ChildFailedError):
            launch.main(args)
        record_mock.assert_called_once()

    @skip_but_pass_in_sandcastle_if(TEST_WITH_DEV_DBG_ASAN, 'test incompatible with dev/dbg asan')
    @mock.patch('torch.distributed.elastic.agent.server.local_elastic_agent.LocalElasticAgent.run')
    @mock.patch('torch.distributed.elastic.events.record')
    def test_launch_elastic_agent_raise_exception(self, record_mock, mock_agent_run):
        if False:
            while True:
                i = 10
        '\n        Asserts that when the agent raises an exception\n        the launcher re-raises the original exception\n        '
        run_id = str(uuid.uuid4().int)
        min_nodes = 1
        max_nodes = 2
        nproc_per_node = 4
        args = [f'--nnodes={min_nodes}:{max_nodes}', f'--nproc-per-node={nproc_per_node}', '--rdzv-backend=etcd', f'--rdzv-endpoint={self._etcd_endpoint}', f'--rdzv-id={run_id}', '--monitor-interval=1', '--max-restarts=0', '--start-method=spawn', path('bin/test_script.py'), f'--touch-file-dir={self.test_dir}']
        mock_agent_run.side_effect = MockException
        with self.assertRaises(MockException):
            launch.main(args)
        record_mock.assert_called_once()

    @skip_but_pass_in_sandcastle_if(TEST_WITH_DEV_DBG_ASAN, 'test incompatible with dev/dbg asan')
    def test_launch_standalone(self):
        if False:
            while True:
                i = 10
        nnodes = 1
        nproc_per_node = 4
        world_size = nnodes * nproc_per_node
        args = [f'--nnodes={nnodes}', f'--nproc-per-node={nproc_per_node}', '--standalone', '--monitor-interval=1', '--start-method=spawn', path('bin/test_script.py'), f'--touch-file-dir={self.test_dir}']
        launch.main(args)
        self.assertSetEqual({str(i) for i in range(world_size)}, set(os.listdir(self.test_dir)))

    @skip_but_pass_in_sandcastle_if(TEST_WITH_DEV_DBG_ASAN, 'test incompatible with dev/dbg asan')
    def test_launch_run_path(self):
        if False:
            for i in range(10):
                print('nop')
        nnodes = 1
        nproc_per_node = 4
        world_size = nnodes * nproc_per_node
        args = ['--run-path', f'--nnodes={nnodes}', f'--nproc-per-node={nproc_per_node}', '--monitor-interval=1', '--start-method=spawn', path('bin/test_script.py'), f'--touch-file-dir={self.test_dir}']
        launch.main(args)
        self.assertSetEqual({str(i) for i in range(world_size)}, set(os.listdir(self.test_dir)))

    @skip_but_pass_in_sandcastle_if(TEST_WITH_DEV_DBG_ASAN, 'test incompatible with dev/dbg asan')
    def test_launch_elastic_multiple_agents(self):
        if False:
            return 10
        run_id = str(uuid.uuid4().int)
        min_nodes = 1
        max_nodes = 2
        nproc_per_node = 4
        nnodes = 2
        world_size = nnodes * nproc_per_node
        args = [f'--nnodes={min_nodes}:{max_nodes}', f'--nproc-per-node={nproc_per_node}', '--rdzv-backend=etcd', f'--rdzv-endpoint={self._etcd_endpoint}', f'--rdzv-id={run_id}', '--monitor-interval=1', '--start-method=spawn', path('bin/test_script.py'), f'--touch-file-dir={self.test_dir}']
        procs = []
        for _ in range(nnodes - 1):
            p = mp.Process(target=launch.main, args=[args])
            procs.append(p)
            p.start()
        launch.main(args)
        for i in range(nnodes - 1):
            p = procs[i]
            p.join()
            self.assertEqual(0, p.exitcode)
        self.assertSetEqual({str(i) for i in range(world_size)}, set(os.listdir(self.test_dir)))

    def test_min_max_nodes_parse(self):
        if False:
            print('Hello World!')
        (min_nodes, max_nodes) = launch.parse_min_max_nnodes('1')
        self.assertTrue(min_nodes, max_nodes)
        self.assertTrue(1, min_nodes)
        (min_nodes, max_nodes) = launch.parse_min_max_nnodes('2:20')
        self.assertTrue(2, min_nodes)
        self.assertTrue(20, max_nodes)
        with self.assertRaises(RuntimeError):
            launch.parse_min_max_nnodes('2:20:30')

    @patch('torch.distributed.launcher.api.LocalElasticAgent')
    def test_launch_shutdown(self, agent_mock_cls):
        if False:
            i = 10
            return i + 15
        nnodes = 1
        nproc_per_node = 4
        args = [f'--nnodes={nnodes}', f'--nproc-per-node={nproc_per_node}', '--monitor-interval=1', '--start-method=spawn', path('bin/test_script.py'), f'--touch-file-dir={self.test_dir}']
        agent_mock = Mock()
        agent_mock.run.return_value = RunResult(WorkerState.SUCCEEDED)
        agent_mock_cls.return_value = agent_mock
        rdzv_handler_mock = Mock()
        with patch('torch.distributed.elastic.rendezvous.registry.get_rendezvous_handler') as param_mock:
            param_mock.return_value = rdzv_handler_mock
            launch.main(args)
            rdzv_handler_mock.shutdown.assert_called_once()

    @skip_but_pass_in_sandcastle_if(TEST_WITH_DEV_DBG_ASAN, 'test incompatible with dev/dbg asan')
    def test_is_torchelastic_launched(self):
        if False:
            for i in range(10):
                print('nop')
        out_file = f"{os.path.join(self.test_dir, 'out')}"
        launch.main(['--run-path', '--nnodes=1', '--nproc-per-node=1', '--monitor-interval=1', path('bin/test_script_is_torchelastic_launched.py'), f'--out-file={out_file}'])
        with open(out_file) as fp:
            is_torchelastic_launched = fp.readline()
            self.assertEqual('True', is_torchelastic_launched)

    def test_is_not_torchelastic_launched(self):
        if False:
            for i in range(10):
                print('nop')
        out_file = f"{os.path.join(self.test_dir, 'out')}"
        with patch.object(sys, 'argv', [path('bin/test_script_is_torchelastic_launched.py'), f'--out-file={out_file}']):
            runpy.run_path(sys.argv[0], run_name='__main__')
            with open(out_file) as fp:
                is_torchelastic_launched = fp.readline()
                self.assertEqual('False', is_torchelastic_launched)

    def test_init_method_tcp(self):
        if False:
            for i in range(10):
                print('nop')
        port = get_free_port()
        with patch.object(sys, 'argv', [path('bin/test_script_init_method.py'), f'--init-method=tcp://localhost:{port}', '--rank=0', '--world-size=1']):
            runpy.run_path(sys.argv[0], run_name='__main__')

    @skip_but_pass_in_sandcastle_if(TEST_WITH_DEV_DBG_ASAN, 'test incompatible with dev/dbg asan')
    def test_init_method_tcp_with_torchelastic(self):
        if False:
            print('Hello World!')
        port = get_free_port()
        launch.main(['--run-path', '--nnodes=1', '--nproc-per-node=4', '--master-addr=localhost', f'--master-port={port}', '--monitor-interval=1', path('bin/test_script_init_method.py'), f'--init-method=tcp://localhost:{port}'])

    def test_init_method_env(self):
        if False:
            for i in range(10):
                print('nop')
        port = get_free_port()
        with patch.dict(os.environ, {'RANK': '0', 'WORLD_SIZE': '1', 'MASTER_ADDR': 'localhost', 'MASTER_PORT': str(port)}), patch.object(sys, 'argv', [path('bin/test_script_init_method.py'), '--init-method=env://']):
            runpy.run_path(sys.argv[0], run_name='__main__')

    @skip_but_pass_in_sandcastle_if(TEST_WITH_DEV_DBG_ASAN, 'test incompatible with dev/dbg asan')
    def test_init_method_env_with_torchelastic(self):
        if False:
            for i in range(10):
                print('nop')
        port = get_free_port()
        launch.main(['--run-path', '--nnodes=1', '--nproc-per-node=4', '--master-addr=localhost', f'--master-port={port}', '--monitor-interval=1', path('bin/test_script_init_method.py'), '--init-method=env://'])