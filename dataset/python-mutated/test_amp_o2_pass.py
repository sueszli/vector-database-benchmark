import os
import subprocess
import sys
import tempfile
import unittest

class TestAMPO2(unittest.TestCase):

    def test_bf16(self):
        if False:
            for i in range(10):
                print('nop')
        file_dir = os.path.dirname(os.path.abspath(__file__))
        launch_model_path = os.path.join(file_dir, 'amp_o2_pass.py')
        if os.environ.get('WITH_COVERAGE', 'OFF') == 'ON':
            coverage_args = ['-m', 'coverage', 'run', '--branch', '-p']
        else:
            coverage_args = []
        tmp_dir = tempfile.TemporaryDirectory()
        cmd = [sys.executable, '-u'] + coverage_args + ['-m', 'paddle.distributed.launch', '--devices', '0,1', '--log_dir', tmp_dir.name, launch_model_path]
        process = subprocess.Popen(cmd)
        process.wait()
        self.assertEqual(process.returncode, 0)
        tmp_dir.cleanup()
if __name__ == '__main__':
    unittest.main()