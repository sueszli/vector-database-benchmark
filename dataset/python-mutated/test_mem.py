import unittest
import jittor as jt
import numpy as np
import os
model_test = os.environ.get('model_test', '') == '1'
skip_model_test = not model_test

class TestMem(unittest.TestCase):

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        jt.clean()
        jt.gc()

    @unittest.skipIf(not jt.has_cuda, 'no cuda found')
    @unittest.skipIf(skip_model_test, 'skip_model_test')
    @jt.flag_scope(use_cuda=1)
    def test_oom(self):
        if False:
            for i in range(10):
                print('nop')
        backups = []
        jt.flags.use_cuda = 1
        one_g = np.ones((1024 * 1024 * 1024 // 4,), 'float32')
        meminfo = jt.get_mem_info()
        n = int(meminfo.total_cuda_ram // 1024 ** 3 * 0.6)
        for i in range(n):
            a = jt.array(one_g)
            b = a + 1
            b.sync()
            backups.append((a, b))
        jt.sync_all(True)
        backups = []
if __name__ == '__main__':
    unittest.main()