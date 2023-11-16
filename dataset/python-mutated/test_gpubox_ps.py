import os
import shlex
import unittest

class GpuBoxTest(unittest.TestCase):

    def test_gpubox(self):
        if False:
            return 10
        if not os.path.exists('./train_data'):
            os.system('bash download_criteo_data.sh')
        exitcode = os.system('bash gpubox_run.sh')
        if os.path.exists('./train_data'):
            os.system('rm -rf train_data')
        if exitcode:
            os.system('cat ./log/worker.0.log')
        assert exitcode == 0
if __name__ == '__main__':
    unittest.main()