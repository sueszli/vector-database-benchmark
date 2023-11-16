import unittest
from test_dist_base import TestDistBase

class TestDistMnist2x2Lars(TestDistBase):

    def _setup_config(self):
        if False:
            for i in range(10):
                print('nop')
        self._sync_mode = True
        self._use_reduce = False

    def test_se_resnext(self):
        if False:
            return 10
        self.check_with_place('dist_mnist_lars.py', delta=1e-05)
if __name__ == '__main__':
    unittest.main()