import os
import tempfile
import unittest
from paddle.distributed.fleet.elastic.collective import CollectiveLauncher
from paddle.distributed.fleet.launch import launch_collective
fake_python_code = '\nprint("test")\n'

class TestCollectiveLauncher(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.temp_dir = tempfile.TemporaryDirectory()
        self.code_path = os.path.join(self.temp_dir.name, 'fake_python_for_elastic.py')
        with open(self.code_path, 'w') as f:
            f.write(fake_python_code)

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.temp_dir.cleanup()

    def test_launch(self):
        if False:
            while True:
                i = 10

        class Argument:
            elastic_server = '127.0.0.1:2379'
            job_id = 'test_job_id_123'
            np = '1'
            gpus = '0'
            nproc_per_node = 1
            host = None
            curr_host = None
            ips = '127.0.0.1'
            scale = None
            force = None
            backend = 'gloo'
            enable_auto_mapping = False
            run_mode = 'cpuonly'
            servers = None
            rank_mapping_path = None
            training_script = self.code_path
            training_script_args = ['--use_amp false']
            log_dir = None
        args = Argument()
        launch = CollectiveLauncher(args)
        try:
            args.backend = 'gloo'
            launch.launch()
            launch.stop()
        except Exception as e:
            pass
        try:
            args.backend = 'gloo'
            launch_collective(args)
        except Exception as e:
            pass

    def test_stop(self):
        if False:
            return 10

        class Argument:
            elastic_server = '127.0.0.1:2379'
            job_id = 'test_job_id_123'
            np = '1'
            gpus = '0'
            nproc_per_node = 1
            host = None
            curr_host = None
            ips = '127.0.0.1'
            scale = None
            force = None
            backend = 'gloo'
            enable_auto_mapping = False
            run_mode = 'cpuonly'
            servers = None
            rank_mapping_path = None
            training_script = self.code_path
            training_script_args = ['--use_amp false']
            log_dir = None
        args = Argument()
        try:
            launch = CollectiveLauncher(args)
            launch.tmp_dir = tempfile.mkdtemp()
            launch.stop()
        except Exception as e:
            pass
if __name__ == '__main__':
    unittest.main()