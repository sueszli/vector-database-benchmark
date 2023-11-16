import os
import subprocess
import sys
import unittest
sys.path.append('.')

class TestCSoftmaxWithCrossEntropy(unittest.TestCase):

    def pdrun(self, need_envs={}):
        if False:
            return 10
        cmd = [sys.executable, '-m', 'paddle.distributed.launch', '--devices', '0,1', 'c_softmax_with_cross_entropy_op.py']
        envs = os.environ.copy()
        envs.update(need_envs)
        proc = subprocess.Popen(cmd, env=envs)
        return proc

    def test_c_softmax_with_cross_entropy_op(self):
        if False:
            while True:
                i = 10
        p = self.pdrun()
        p.wait()

    def test_c_softmax_with_cross_entropy_new_comm(self):
        if False:
            return 10
        p = self.pdrun(need_envs={'FLAGS_dynamic_static_unified_comm': '1'})
        p.wait()
if __name__ == '__main__':
    unittest.main()