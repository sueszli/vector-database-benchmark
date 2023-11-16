import multiprocessing
import os
import unittest
import paddle
import paddle.distributed as dist
from paddle.base import core
from paddle.distributed.spawn import _get_default_nprocs, _get_subprocess_env_list, _options_valid_check

@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestInitParallelEnv(unittest.TestCase):

    def test_check_env_failed(self):
        if False:
            while True:
                i = 10
        os.environ['FLAGS_selected_gpus'] = '0'
        os.environ['PADDLE_TRAINER_ID'] = '0'
        os.environ['PADDLE_TRAINERS_NUM'] = '2'
        with self.assertRaises(ValueError):
            dist.init_parallel_env()

    def test_init_parallel_env_break(self):
        if False:
            for i in range(10):
                print('nop')
        from paddle.distributed import parallel_helper
        os.environ['FLAGS_selected_gpus'] = '0'
        os.environ['PADDLE_TRAINER_ID'] = '0'
        os.environ['PADDLE_CURRENT_ENDPOINT'] = '127.0.0.1:6170'
        os.environ['PADDLE_TRAINERS_NUM'] = '1'
        os.environ['PADDLE_TRAINER_ENDPOINTS'] = '127.0.0.1:6170'
        dist.init_parallel_env()
        self.assertFalse(parallel_helper._is_parallel_ctx_initialized())

@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestSpawnAssistMethod(unittest.TestCase):

    def test_nprocs_greater_than_device_num_error(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(RuntimeError):
            _get_subprocess_env_list(nprocs=100, options={})

    def test_selected_devices_error(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(ValueError):
            options = {}
            options['selected_devices'] = '100,101'
            _get_subprocess_env_list(nprocs=2, options=options)

    def test_get_correct_env(self):
        if False:
            i = 10
            return i + 15
        options = {}
        options['print_config'] = True
        env_dict = _get_subprocess_env_list(nprocs=1, options=options)[0]
        self.assertEqual(env_dict['PADDLE_TRAINER_ID'], '0')
        self.assertEqual(env_dict['PADDLE_TRAINERS_NUM'], '1')

    def test_nprocs_not_equal_to_selected_devices(self):
        if False:
            return 10
        with self.assertRaises(ValueError):
            options = {}
            options['selected_devices'] = '100,101,102'
            _get_subprocess_env_list(nprocs=2, options=options)

    def test_options_valid_check(self):
        if False:
            i = 10
            return i + 15
        options = {}
        options['selected_devices'] = '100,101,102'
        _options_valid_check(options)
        with self.assertRaises(ValueError):
            options['error'] = 'error'
            _options_valid_check(options)

    def test_get_default_nprocs(self):
        if False:
            while True:
                i = 10
        paddle.set_device('cpu')
        nprocs = _get_default_nprocs()
        self.assertEqual(nprocs, multiprocessing.cpu_count())
        paddle.set_device('gpu')
        nprocs = _get_default_nprocs()
        self.assertEqual(nprocs, core.get_cuda_device_count())
if __name__ == '__main__':
    unittest.main()