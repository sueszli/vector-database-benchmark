import sys
import unittest
from google.protobuf import text_format
sys.path.append('../distributed_passes')
from ps_pass_test_base import PsPassTestBase, remove_path_if_exists
import paddle.distributed.fleet.proto.the_one_ps_pb2 as ps_pb2
from paddle.distributed.ps.utils.public import logger, ps_log_root_dir

class TestTheOnePs(PsPassTestBase):

    def setUp(self):
        if False:
            return 10
        pass

    def tearDown(self):
        if False:
            return 10
        pass

    def check(self, file1, file2):
        if False:
            for i in range(10):
                print('nop')
        '\n        f = open(file1, "rb")\n        ps_desc_1 = ps_pb2.PSParameter()\n        text_format.Parse(f.read(), ps_desc_1)\n        f.close()\n\n        f = open(file2, "rb")\n        ps_desc_2 = ps_pb2.PSParameter()\n        text_format.Parse(f.read(), ps_desc_2)\n        f.close()\n        str1 = text_format.MessageToString(ps_desc_1)\n        str2 = text_format.MessageToString(ps_desc_2)\n        #logger.info(\'### msg10: {}\'.format(str1))\n        #logger.info(\'### msg20: {}\'.format(str2))\n        if str1 == str2:\n            return True\n        else:\n            return False\n        '
        pass

    def test_ps_cpu_async(self):
        if False:
            i = 10
            return i + 15
        self.init()
        self.config['ps_mode_config'] = '../ps/cpu_async_ps_config.yaml'
        self.config['run_the_one_ps'] = '1'
        self.config['debug_the_one_ps'] = '0'
        self.config['log_dir'] = ps_log_root_dir + 'async_cpu_log_old_the_one_ps'
        remove_path_if_exists(self.config['log_dir'])
        self.ps_launch()
        self.config['debug_the_one_ps'] = '1'
        self.config['log_dir'] = ps_log_root_dir + 'async_cpu_log_new_the_one_ps'
        remove_path_if_exists(self.config['log_dir'])
        self.ps_launch()
        desc1 = '/ps_desc_baseline/async_worker_ps_desc'
        desc2 = '/ps_log/async_new_worker_ps_desc'
        desc3 = '/ps_desc_baseline/async_server_ps_desc'
        desc4 = '/ps_log/async_new_server_ps_desc'
        if self.check(desc1, desc2):
            logger.info('test_ps_cpu_async ps_desc: worker passed!')
        else:
            logger.info('test_ps_cpu_async ps_desc: worker failed!')
        if self.check(desc3, desc4):
            logger.info('test_ps_cpu_async ps_desc: server passed!')
        else:
            logger.info('test_ps_cpu_async ps_desc: server failed!')

    def test_ps_cpu_geo(self):
        if False:
            print('Hello World!')
        self.init()
        self.config['ps_mode_config'] = '../ps/cpu_geo_ps_config.yaml'
        self.config['run_the_one_ps'] = '1'
        self.config['debug_the_one_ps'] = '0'
        self.config['log_dir'] = ps_log_root_dir + 'geo_cpu_log_old_the_one_ps'
        remove_path_if_exists(self.config['log_dir'])
        self.ps_launch()
        self.config['debug_the_one_ps'] = '1'
        self.config['log_dir'] = ps_log_root_dir + 'geo_cpu_log_new_the_one_ps'
        remove_path_if_exists(self.config['log_dir'])
        self.ps_launch()
        desc1 = '/ps_desc_baseline/geo_worker_ps_desc'
        desc2 = '/ps_log/geo_new_worker_ps_desc'
        desc3 = '/ps_desc_baseline/geo_server_ps_desc'
        desc4 = '/ps_log/geo_new_server_ps_desc'
        if self.check(desc1, desc2):
            logger.info('test_ps_cpu_geo ps_desc: worker passed!')
        else:
            logger.info('test_ps_cpu_geo ps_desc: worker failed!')
        if self.check(desc3, desc4):
            logger.info('test_ps_cpu_geo ps_desc: server passed!')
        else:
            logger.info('test_ps_cpu_geo ps_desc: server failed!')
if __name__ == '__main__':
    unittest.main()