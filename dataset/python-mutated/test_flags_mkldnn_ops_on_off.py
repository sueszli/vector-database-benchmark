import os
import re
import subprocess
import sys
import unittest

class TestFlagsUseMkldnn(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self._python_interp = sys.executable
        self._python_interp += ' check_flags_mkldnn_ops_on_off.py'
        self.env = os.environ.copy()
        self.env['DNNL_VERBOSE'] = '1'
        self.env['FLAGS_use_mkldnn'] = '1'
        self.relu_regex = b'^onednn_verbose,exec,cpu,eltwise,.+alg:eltwise_relu alpha:0 beta:0,10x20x20'
        self.ew_add_regex = b'^onednn_verbose,exec,cpu,binary.+alg:binary_add,10x20x30:10x20x30'
        self.matmul_regex = b'^onednn_verbose,exec,cpu,matmul,.*10x20x30:10x30x20:10x20x20'

    def flags_use_mkl_dnn_common(self, e):
        if False:
            while True:
                i = 10
        cmd = self._python_interp
        env = dict(self.env, **e)
        proc = subprocess.Popen(cmd.split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        (out, err) = proc.communicate()
        returncode = proc.returncode
        assert returncode == 0
        return (out, err)

    def _print_when_false(self, cond, out, err):
        if False:
            for i in range(10):
                print('nop')
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

    def not_found(self, regex, out, err):
        if False:
            while True:
                i = 10
        _not_found = not re.search(regex, out, re.MULTILINE)
        return self._print_when_false(_not_found, out, err)

    def test_flags_use_mkl_dnn_on_empty_off_empty(self):
        if False:
            while True:
                i = 10
        (out, err) = self.flags_use_mkl_dnn_common({})
        assert self.found(self.relu_regex, out, err)
        assert self.found(self.ew_add_regex, out, err)
        assert self.found(self.matmul_regex, out, err)

    def test_flags_use_mkl_dnn_on(self):
        if False:
            while True:
                i = 10
        env = {'FLAGS_tracer_mkldnn_ops_on': 'relu'}
        (out, err) = self.flags_use_mkl_dnn_common(env)
        assert self.found(self.relu_regex, out, err)
        assert self.not_found(self.ew_add_regex, out, err)
        assert self.not_found(self.matmul_regex, out, err)

    def test_flags_use_mkl_dnn_on_multiple(self):
        if False:
            i = 10
            return i + 15
        env = {'FLAGS_tracer_mkldnn_ops_on': 'relu,elementwise_add'}
        (out, err) = self.flags_use_mkl_dnn_common(env)
        assert self.found(self.relu_regex, out, err)
        assert self.found(self.ew_add_regex, out, err)
        assert self.not_found(self.matmul_regex, out, err)

    def test_flags_use_mkl_dnn_off(self):
        if False:
            for i in range(10):
                print('nop')
        env = {'FLAGS_tracer_mkldnn_ops_off': 'matmul_v2'}
        (out, err) = self.flags_use_mkl_dnn_common(env)
        assert self.found(self.relu_regex, out, err)
        assert self.found(self.ew_add_regex, out, err)
        assert self.not_found(self.matmul_regex, out, err)

    def test_flags_use_mkl_dnn_off_multiple(self):
        if False:
            print('Hello World!')
        env = {'FLAGS_tracer_mkldnn_ops_off': 'matmul_v2,relu'}
        (out, err) = self.flags_use_mkl_dnn_common(env)
        assert self.not_found(self.relu_regex, out, err)
        assert self.found(self.ew_add_regex, out, err)
        assert self.not_found(self.matmul_regex, out, err)

    def test_flags_use_mkl_dnn_on_off(self):
        if False:
            for i in range(10):
                print('nop')
        env = {'FLAGS_tracer_mkldnn_ops_on': 'elementwise_add', 'FLAGS_tracer_mkldnn_ops_off': 'matmul_v2'}
        (out, err) = self.flags_use_mkl_dnn_common(env)
        assert self.not_found(self.relu_regex, out, err)
        assert self.found(self.ew_add_regex, out, err)
        assert self.not_found(self.matmul_regex, out, err)
if __name__ == '__main__':
    unittest.main()