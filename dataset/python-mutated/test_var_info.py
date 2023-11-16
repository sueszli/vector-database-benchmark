"""
TestCases for Dataset,
including create, config, run, etc.
"""
import unittest
import numpy as np
import paddle

class TestVarInfo(unittest.TestCase):
    """TestCases for Dataset."""

    def test_var_info(self):
        if False:
            for i in range(10):
                print('nop')
        'Testcase for get and set info for variable.'
        value = np.random.randn(1)
        var = paddle.static.create_global_var([1], value, 'float32')
        var._set_info('name', 'test')
        ret = var._get_info('name')
        assert ret == 'test'
        ret = var._get_info('not_exist')
        assert ret is None
if __name__ == '__main__':
    unittest.main()