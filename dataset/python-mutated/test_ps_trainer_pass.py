import unittest
from ps_pass_test_base import PsPassTestBase, remove_path_if_exists
from paddle.distributed.ps.utils.public import logger, ps_log_root_dir

class TestPsTrainerPass(PsPassTestBase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        pass

    def tearDown(self):
        if False:
            print('Hello World!')
        pass

    def check(self, file1, file2):
        if False:
            i = 10
            return i + 15
        with open(file1, 'r', encoding='utf-8') as f:
            text1 = f.read()
        with open(file2, 'r', encoding='utf-8') as f:
            text2 = f.read()
        if text1 == text2:
            return True
        else:
            return False

    def test_ps_optimizer_minimize_cpu_async(self):
        if False:
            print('Hello World!')
        self.init()
        self.config['ps_mode_config'] = '../ps/cpu_async_ps_config.yaml'
        self.config['run_minimize'] = '1'
        self.config['debug_new_minimize'] = '0'
        self.config['log_dir'] = ps_log_root_dir + 'async_cpu_log_old_minimize'
        remove_path_if_exists(self.config['log_dir'])
        self.ps_launch()
        self.config['debug_new_minimize'] = '1'
        self.config['log_dir'] = ps_log_root_dir + 'async_cpu_log_new_minimize'
        remove_path_if_exists(self.config['log_dir'])
        self.ps_launch()
        file1 = './ps_log/async_run_minimize_debug:_0_worker_main.prototxt'
        file2 = './ps_log/async_run_minimize_debug:_1_worker_main.prototxt'
        if self.check(file1, file2):
            logger.info('test_ps_optimizer_minimize_cpu_async passed!')
        else:
            logger.error('test_ps_optimizer_minimize_cpu_async failed!')

    def test_ps_optimizer_minimize_cpu_sync(self):
        if False:
            while True:
                i = 10
        self.init()
        self.config['ps_mode_config'] = '../ps/cpu_sync_ps_config.yaml'
        self.config['run_minimize'] = '1'
        self.config['debug_new_minimize'] = '0'
        self.config['log_dir'] = ps_log_root_dir + 'sync_cpu_log_old_minimize'
        remove_path_if_exists(self.config['log_dir'])
        self.ps_launch()
        self.config['debug_new_minimize'] = '1'
        self.config['log_dir'] = ps_log_root_dir + 'sync_cpu_log_new_minimize'
        remove_path_if_exists(self.config['log_dir'])
        self.ps_launch()
        "\n        file1 = './ps_log/sync_run_minimize_debug:_0_worker_main.prototxt'\n        file2 = './ps_log/sync_run_minimize_debug:_1_worker_main.prototxt'\n        if self.check(file1, file2):\n            logger.info('test_ps_optimizer_minimize_cpu_sync passed!')\n        else:\n            logger.error('test_ps_optimizer_minimize_cpu_sync failed!')\n        "

    def test_ps_optimizer_minimize_cpu_geo(self):
        if False:
            return 10
        self.init()
        self.config['ps_mode_config'] = '../ps/cpu_geo_ps_config.yaml'
        self.config['run_minimize'] = '1'
        self.config['debug_new_minimize'] = '0'
        self.config['log_dir'] = ps_log_root_dir + 'geo_cpu_log_old_minimize'
        remove_path_if_exists(self.config['log_dir'])
        self.ps_launch()
        self.config['debug_new_minimize'] = '1'
        self.config['log_dir'] = ps_log_root_dir + 'geo_cpu_log_new_minimize'
        remove_path_if_exists(self.config['log_dir'])
        self.ps_launch()
        file1 = './ps_log/geo_run_minimize_debug:_0_worker_main.prototxt'
        file2 = './ps_log/geo_run_minimize_debug:_1_worker_main.prototxt'
        if self.check(file1, file2):
            logger.info('test_ps_optimizer_minimize_cpu_geo passed!')
        else:
            logger.error('test_ps_optimizer_minimize_cpu_geo failed!')

    def test_ps_optimizer_minimize_heter(self):
        if False:
            for i in range(10):
                print('nop')
        self.init()
        self.config['worker_num'] = '2'
        self.config['server_num'] = '2'
        self.config['heter_worker_num'] = '2'
        self.config['heter_devices'] = 'gpu'
        self.config['run_minimize'] = '1'
        self.config['ps_mode_config'] = '../ps/heter_ps_config.yaml'
        self.config['debug_new_minimize'] = '0'
        self.config['log_dir'] = ps_log_root_dir + 'heter_log_old_minimize'
        remove_path_if_exists(self.config['log_dir'])
        self.ps_launch('heter-ps')
        self.config['debug_new_minimize'] = '1'
        self.config['log_dir'] = ps_log_root_dir + 'heter_log_new_minimize'
        remove_path_if_exists(self.config['log_dir'])
        self.ps_launch('heter-ps')
        "\n        file1 = './ps_log/heter_run_minimize_debug:_0_worker_main.prototxt'\n        file2 = './ps_log/heter_run_minimize_debug:_1_worker_main.prototxt'\n        file3 = './ps_log/heter_run_minimize_debug:_0_heter_worker_main.prototxt'\n        file4 = './ps_log/heter_run_minimize_debug:_1_heter_worker_main.prototxt'\n        if self.check(file1, file2) and self.check(file3, file4):\n            logger.info('test_ps_optimizer_minimize_heter passed!')\n        else:\n            logger.error('test_ps_optimizer_minimize_heter failed!')\n        "

    def test_ps_optimizer_minimize_gpu(self):
        if False:
            return 10
        self.init()
        self.config['run_minimize'] = '1'
        self.config['ps_mode_config'] = '../ps/gpu_ps_config.yaml'
        self.config['debug_new_minimize'] = '0'
        self.config['log_dir'] = ps_log_root_dir + 'gpubox_log_old_minimize'
        remove_path_if_exists(self.config['log_dir'])
        self.config['debug_new_minimize'] = '1'
        self.config['log_dir'] = ps_log_root_dir + 'gpubox_log_new_minimize'
        remove_path_if_exists(self.config['log_dir'])

    def test_append_send_ops_pass(self):
        if False:
            for i in range(10):
                print('nop')
        self.init()
        self.config['run_single_pass'] = '1'
        self.config['ps_mode_config'] = '../ps/cpu_async_ps_config.yaml'
        self.config['applied_pass_name'] = 'append_send_ops_pass'
        self.config['debug_new_pass'] = '0'
        self.config['log_dir'] = ps_log_root_dir + 'log_old_' + self.config['applied_pass_name']
        remove_path_if_exists(self.config['log_dir'])
        self.ps_launch('cpu-ps')
        self.config['debug_new_pass'] = '1'
        self.config['log_dir'] = ps_log_root_dir + 'log_new_' + self.config['applied_pass_name']
        remove_path_if_exists(self.config['log_dir'])
        self.ps_launch('cpu-ps')
        file1 = './ps_log/async_append_send_ops_pass_debug:_0_worker_main.prototxt'
        file2 = './ps_log/async_append_send_ops_pass_debug:_1_worker_main.prototxt'
        if self.check(file1, file2):
            logger.info('test_append_send_ops_pass passed!')
        else:
            logger.info('test_append_send_ops_pass failed!')

    def test_distributed_ops_pass(self):
        if False:
            while True:
                i = 10
        pass
if __name__ == '__main__':
    remove_path_if_exists('./ps_log')
    unittest.main()