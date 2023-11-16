import os
import subprocess
import sys
import tempfile
import unittest

class TestGradientMergePass(unittest.TestCase):

    def test_dp2(self):
        if False:
            return 10
        file_dir = os.path.dirname(os.path.abspath(__file__))
        launch_model_path = os.path.join(file_dir, 'gradient_merge_pass_unittest.py')
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