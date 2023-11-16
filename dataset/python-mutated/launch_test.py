import os
import shutil
import tempfile
import unittest
from contextlib import closing
import torch.distributed.launch as launch
from torch.distributed.elastic.utils import get_socket_with_port
from torch.testing._internal.common_utils import TEST_WITH_DEV_DBG_ASAN, skip_but_pass_in_sandcastle_if

def path(script):
    if False:
        print('Hello World!')
    return os.path.join(os.path.dirname(__file__), script)

class LaunchTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.test_dir = tempfile.mkdtemp()
        os.environ['TEST_SENTINEL_PARENT'] = 'FOOBAR'

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        shutil.rmtree(self.test_dir)

    @skip_but_pass_in_sandcastle_if(TEST_WITH_DEV_DBG_ASAN, 'test incompatible with dev/dbg asan')
    def test_launch_without_env(self):
        if False:
            print('Hello World!')
        nnodes = 1
        nproc_per_node = 4
        world_size = nnodes * nproc_per_node
        sock = get_socket_with_port()
        with closing(sock):
            master_port = sock.getsockname()[1]
        args = [f'--nnodes={nnodes}', f'--nproc-per-node={nproc_per_node}', '--monitor-interval=1', '--start-method=spawn', '--master-addr=localhost', f'--master-port={master_port}', '--node-rank=0', path('bin/test_script_local_rank.py')]
        launch.main(args)

    @skip_but_pass_in_sandcastle_if(TEST_WITH_DEV_DBG_ASAN, 'test incompatible with dev/dbg asan')
    def test_launch_with_env(self):
        if False:
            for i in range(10):
                print('nop')
        nnodes = 1
        nproc_per_node = 4
        world_size = nnodes * nproc_per_node
        sock = get_socket_with_port()
        with closing(sock):
            master_port = sock.getsockname()[1]
        args = [f'--nnodes={nnodes}', f'--nproc-per-node={nproc_per_node}', '--monitor-interval=1', '--start-method=spawn', '--master-addr=localhost', f'--master-port={master_port}', '--node-rank=0', '--use-env', path('bin/test_script.py'), f'--touch-file-dir={self.test_dir}']
        launch.main(args)
        self.assertSetEqual({str(i) for i in range(world_size)}, set(os.listdir(self.test_dir)))