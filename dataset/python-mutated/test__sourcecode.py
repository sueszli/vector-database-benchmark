import os
import subprocess
import sys
import unittest

def find_root():
    if False:
        i = 10
        return i + 15
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TestFlake8(unittest.TestCase):

    def test_flake8(self):
        if False:
            while True:
                i = 10
        try:
            import flake8
        except ImportError:
            raise unittest.SkipTest('flake8 module is missing')
        root_path = find_root()
        config_path = os.path.join(root_path, '.flake8')
        if not os.path.exists(config_path):
            raise RuntimeError('could not locate .flake8 file')
        try:
            subprocess.run([sys.executable, '-m', 'flake8', '--config', config_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=root_path)
        except subprocess.CalledProcessError as ex:
            output = ex.output.decode()
            raise AssertionError('flake8 validation failed:\n{}'.format(output)) from None