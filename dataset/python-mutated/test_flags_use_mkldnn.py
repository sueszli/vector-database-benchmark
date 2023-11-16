import os
import re
import subprocess
import sys
import unittest

class TestFlagsUseMkldnn(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self._python_interp = sys.executable
        self._python_interp += ' check_flags_use_mkldnn.py'
        self.env = os.environ.copy()
        self.env['GLOG_v'] = '1'
        self.env['DNNL_VERBOSE'] = '1'
        self.env['FLAGS_use_mkldnn'] = '1'
        self.relu_regex = b'^onednn_verbose,exec,cpu,eltwise,.+alg:eltwise_relu alpha:0 beta:0,10x20x30'

    def _print_when_false(self, cond, out, err):
        if False:
            return 10
        if not cond:
            print('out', out)
            print('err', err)
        return cond

    def found(self, regex, out, err):
        if False:
            while True:
                i = 10
        _found = re.search(regex, out, re.MULTILINE)
        return self._print_when_false(_found, out, err)

    def test_flags_use_mkl_dnn(self):
        if False:
            for i in range(10):
                print('nop')
        cmd = self._python_interp
        proc = subprocess.Popen(cmd.split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=self.env)
        (out, err) = proc.communicate()
        returncode = proc.returncode
        assert returncode == 0
        assert self.found(self.relu_regex, out, err)
if __name__ == '__main__':
    unittest.main()