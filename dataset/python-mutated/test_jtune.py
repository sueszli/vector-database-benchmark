import unittest
import jittor as jt
import os
import re
import sys

class TestJtune(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        if False:
            while True:
                i = 10
        (n, m, k) = (2, 6, 16)
        a = jt.random((n, m, 1))
        b = jt.random((1, m, k))
        jt.fetch_sync([a, b])
        with jt.profile_scope(compile_options={'jtune': 1}) as rep:
            c = (a * b).sum(1)
            c.sync()
        assert len(rep) == 2
        self.fname = rep[1][1]
        self.jtune_path = os.path.join(jt.flags.jittor_path, 'utils/jtune.py')

    def run_cmd(self, cmd):
        if False:
            for i in range(10):
                print('nop')
        cmd = f'warmup=0 rerun=0 {sys.executable} {self.jtune_path} {self.fname} {cmd}'
        return jt.compiler.run_cmd(cmd)

    def test_run_so(self):
        if False:
            while True:
                i = 10
        res = self.run_cmd('run_so').splitlines()
        assert res[0] == 'Enter fake_main entry.', res
        assert res[1] == '     Count TotalTime   AvgTime   MinTime   MaxTime     Input    Output   Compute', res
        nums = res[2].split()
        assert nums[0] == '1', nums

    def test_cc_to_so(self):
        if False:
            i = 10
            return i + 15
        self.run_cmd('cc_to_so')

    def test_cc_to_s(self):
        if False:
            i = 10
            return i + 15
        self.run_cmd('cc_to_s')
        sname = self.fname[:-2] + 's'
        with open(sname) as f:
            src = f.read()
        fma_ins = re.findall('fma.*', src)
        assert len(fma_ins) >= 4, f'fma instructions should be used for matmul. {fma_ins}'
        self.run_cmd('s_to_so')
if __name__ == '__main__':
    unittest.main()