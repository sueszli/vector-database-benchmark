from contextlib import contextmanager
import os
import shutil
import sys
import textwrap
from typing import Any, Callable, Dict, Tuple
import unittest
from pyspark import SparkConf, SparkContext
from pyspark.ml.deepspeed.deepspeed_distributor import DeepspeedTorchDistributor
from pyspark.sql import SparkSession
from pyspark.ml.torch.tests.test_distributor import get_local_mode_conf, set_up_test_dirs, get_distributed_mode_conf
have_deepspeed = True
try:
    import deepspeed
except ImportError:
    have_deepspeed = False

class DeepspeedTorchDistributorUnitTests(unittest.TestCase):

    def _get_env_var(self, var_name: str, default_value: Any) -> Any:
        if False:
            return 10
        value = os.getenv(var_name)
        if value:
            return value
        os.environ[var_name] = str(default_value)
        return default_value

    def _get_env_variables_distributed(self) -> Tuple[Any, Any, Any]:
        if False:
            print('Hello World!')
        master_addr = self._get_env_var('MASTER_ADDR', '127.0.0.1')
        master_port = self._get_env_var('MASTER_PORT', 2000)
        rank = self._get_env_var('RANK', 0)
        return (master_addr, master_port, rank)

    def test_get_torchrun_args_local(self) -> None:
        if False:
            print('Hello World!')
        number_of_processes = 5
        expected_torchrun_args_local = ['--standalone', '--nnodes=1']
        expected_processes_per_node_local = number_of_processes
        (get_local_mode_torchrun_args, process_per_node) = DeepspeedTorchDistributor._get_torchrun_args(True, number_of_processes)
        self.assertEqual(get_local_mode_torchrun_args, expected_torchrun_args_local)
        self.assertEqual(expected_processes_per_node_local, process_per_node)

    def test_get_torchrun_args_distributed(self) -> None:
        if False:
            return 10
        number_of_processes = 5
        (master_addr, master_port, rank) = self._get_env_variables_distributed()
        expected_torchrun_args_distributed = [f'--nnodes={number_of_processes}', f'--node_rank={rank}', f'--rdzv_endpoint={master_addr}:{master_port}', '--rdzv_id=0']
        (torchrun_args_distributed, process_per_node) = DeepspeedTorchDistributor._get_torchrun_args(False, number_of_processes)
        self.assertEqual(torchrun_args_distributed, expected_torchrun_args_distributed)
        self.assertEqual(process_per_node, 1)

    def test_create_torchrun_command_local(self) -> None:
        if False:
            return 10
        deepspeed_conf = 'path/to/deepspeed'
        train_file_path = 'path/to/exec'
        num_procs = 10
        input_params: Dict[str, Any] = {}
        input_params['local_mode'] = True
        input_params['num_processes'] = num_procs
        input_params['deepspeed_config'] = deepspeed_conf
        torchrun_local_args_expected = ['--standalone', '--nnodes=1']
        with self.subTest(msg='Testing local training with no extra args'):
            local_cmd_no_args_expected = [sys.executable, '-m', 'torch.distributed.run', *torchrun_local_args_expected, f'--nproc_per_node={num_procs}', train_file_path, '--deepspeed', '--deepspeed_config', deepspeed_conf]
            local_cmd = DeepspeedTorchDistributor._create_torchrun_command(input_params, train_file_path)
            self.assertEqual(local_cmd, local_cmd_no_args_expected)
        with self.subTest(msg='Testing local training with extra args for the training script'):
            local_mode_version_args = ['--arg1', '--arg2']
            local_cmd_args_expected = [sys.executable, '-m', 'torch.distributed.run', *torchrun_local_args_expected, f'--nproc_per_node={num_procs}', train_file_path, *local_mode_version_args, '--deepspeed', '--deepspeed_config', deepspeed_conf]
            local_cmd_with_args = DeepspeedTorchDistributor._create_torchrun_command(input_params, train_file_path, *local_mode_version_args)
            self.assertEqual(local_cmd_with_args, local_cmd_args_expected)

    def test_create_torchrun_command_distributed(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        deepspeed_conf = 'path/to/deepspeed'
        train_file_path = 'path/to/exec'
        num_procs = 10
        input_params: Dict[str, Any] = {}
        input_params['local_mode'] = True
        input_params['num_processes'] = num_procs
        input_params['deepspeed_config'] = deepspeed_conf
        (distributed_master_address, distributed_master_port, distributed_rank) = self._get_env_variables_distributed()
        distributed_torchrun_args = [f'--nnodes={num_procs}', f'--node_rank={distributed_rank}', f'--rdzv_endpoint={distributed_master_address}:{distributed_master_port}', '--rdzv_id=0']
        with self.subTest(msg='Distributed training command verification with no extra args'):
            distributed_cmd_no_args_expected = [sys.executable, '-m', 'torch.distributed.run', *distributed_torchrun_args, '--nproc_per_node=1', train_file_path, '--deepspeed', '--deepspeed_config', deepspeed_conf]
            input_params['local_mode'] = False
            distributed_command = DeepspeedTorchDistributor._create_torchrun_command(input_params, train_file_path)
            self.assertEqual(distributed_cmd_no_args_expected, distributed_command)
        with self.subTest(msg='Distributed training command verification with extra arguments'):
            distributed_extra_args = ['-args1', '--args2']
            distributed_cmd_args_expected = [sys.executable, '-m', 'torch.distributed.run', *distributed_torchrun_args, '--nproc_per_node=1', train_file_path, *distributed_extra_args, '--deepspeed', '--deepspeed_config', deepspeed_conf]
            distributed_command_with_args = DeepspeedTorchDistributor._create_torchrun_command(input_params, train_file_path, *distributed_extra_args)
            self.assertEqual(distributed_cmd_args_expected, distributed_command_with_args)

def _create_basic_function() -> Callable:
    if False:
        i = 10
        return i + 15

    def pythagoras(leg1: float, leg2: float) -> float:
        if False:
            return 10
        import deepspeed
        print(deepspeed.__version__)
        return (leg1 * leg1 + leg2 * leg2) ** 0.5
    return pythagoras

@contextmanager
def _create_pytorch_training_test_file():
    if False:
        i = 10
        return i + 15
    str_to_write = textwrap.dedent(' \n            import sys\n            def pythagorean_thm(x : int, y: int): # type: ignore \n                import deepspeed # type: ignore\n                return (x*x + y*y)**0.5 # type: ignore\n            print(pythagorean_thm(int(sys.argv[1]), int(sys.argv[2])))')
    cp_path = '/tmp/test_deepspeed_training_file.py'
    with open(cp_path, 'w') as f:
        f.write(str_to_write)
    yield cp_path
    os.remove(cp_path)

@unittest.skipIf(not have_deepspeed, 'deepspeed is required for these tests')
class DeepspeedTorchDistributorDistributedEndToEnd(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            i = 10
            return i + 15
        (cls.gpu_discovery_script_file_name, cls.mnist_dir_path) = set_up_test_dirs()
        conf = SparkConf(loadDefaults=False)
        for (k, v) in get_distributed_mode_conf().items():
            conf = conf.set(k, v)
        conf = conf.set('spark.worker.resource.gpu.discoveryScript', cls.gpu_discovery_script_file_name)
        sc = SparkContext('local-cluster[2,2,512]', cls.__name__, conf=conf)
        cls.spark = SparkSession(sc)

    @classmethod
    def tearDownClass(cls) -> None:
        if False:
            for i in range(10):
                print('nop')
        shutil.rmtree(cls.mnist_dir_path)
        os.unlink(cls.gpu_discovery_script_file_name)
        cls.spark.stop()

    def test_simple_function_e2e(self) -> None:
        if False:
            print('Hello World!')
        train_fn = _create_basic_function()
        x = 3
        y = 4
        dist = DeepspeedTorchDistributor(numGpus=2, useGpu=False, localMode=False)
        output = dist.run(train_fn, x, y)
        self.assertEqual(output, 5)

    def test_pytorch_file_e2e(self) -> None:
        if False:
            i = 10
            return i + 15
        with _create_pytorch_training_test_file() as cp_path:
            dist = DeepspeedTorchDistributor(numGpus=True, useGpu=False, localMode=False)
            dist.run(cp_path, 2, 5)

@unittest.skipIf(not have_deepspeed, 'deepspeed is required for these tests')
class DeepspeedDistributorLocalEndToEndTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            for i in range(10):
                print('nop')
        (cls.gpu_discovery_script_file_name, cls.mnist_dir_path) = set_up_test_dirs()
        conf = SparkConf()
        for (k, v) in get_local_mode_conf().items():
            conf = conf.set(k, v)
        conf = conf.set('spark.driver.resource.gpu.discoveryScript', cls.gpu_discovery_script_file_name)
        sc = SparkContext('local-cluster[2,2,512]', cls.__name__, conf=conf)
        cls.spark = SparkSession(sc)

    @classmethod
    def tearDownClass(cls) -> None:
        if False:
            i = 10
            return i + 15
        shutil.rmtree(cls.mnist_dir_path)
        os.unlink(cls.gpu_discovery_script_file_name)
        cls.spark.stop()

    def test_simple_function_e2e(self) -> None:
        if False:
            return 10
        train_fn = _create_basic_function()
        x = 3
        y = 4
        dist = DeepspeedTorchDistributor(numGpus=2, useGpu=False, localMode=True)
        output = dist.run(train_fn, x, y)
        self.assertEqual(output, 5)

    def test_pytorch_file_e2e(self) -> None:
        if False:
            print('Hello World!')
        with _create_pytorch_training_test_file() as path_to_train_file:
            dist = DeepspeedTorchDistributor(numGpus=2, useGpu=False, localMode=True)
            dist.run(path_to_train_file, 2, 5)
if __name__ == '__main__':
    from pyspark.ml.deepspeed.tests.test_deepspeed_distributor import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)