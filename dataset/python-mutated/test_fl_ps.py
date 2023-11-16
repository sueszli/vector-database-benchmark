import os
import shlex
import sys
import unittest
sys.path.append('../distributed_passes')
from dist_pass_test_base import remove_path_if_exists

class FlPsTest(unittest.TestCase):

    def test_launch_fl_ps(self):
        if False:
            return 10
        '\n        cmd = [\n            \'python\', \'-m\', \'paddle.distributed.fleet.launch\', \'--log_dir\',\n            \'/ps_log/fl_ps\', \'--servers\', "127.0.0.1:8070", \'--workers\',\n            "127.0.0.1:8080,127.0.0.1:8081", \'--heter_workers\',\n            "127.0.0.1:8090,127.0.0.1:8091", \'--heter_devices\', "cpu",\n            \'--worker_num\', "2", \'--heter_worker_num\', "2", \'fl_ps_trainer.py\'\n        ]\n        cmd = [shlex.quote(c) for c in cmd]\n        prepare_python_path_and_return_module(__file__)\n        exitcode = os.system(\' \'.join(cmd))\n        '
if __name__ == '__main__':
    remove_path_if_exists('/ps_log')
    remove_path_if_exists('/ps_usr_print_log')
    if not os.path.exists('./train_data'):
        os.system('sh download_data.sh')
        os.system('rm -rf ctr_data.tar.gz')
        os.sysyem('rm -rf train_data_full')
        os.sysyem('rm -rf test_data_full')
    unittest.main()
    if os.path.exists('./train_data'):
        os.system('rm -rf train_data')
        os.system('rm -rf test_data')