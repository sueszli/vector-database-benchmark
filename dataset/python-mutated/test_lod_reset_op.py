import unittest
import numpy as np
from op_test import OpTest

class TestLodResetOpByAttr(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'lod_reset'
        x = np.random.random((10, 20)).astype('float64')
        lod = [[3, 2, 5]]
        target_offset_lod = [0, 7, 10]
        target_lod = [7, 3]
        self.inputs = {'X': (x, lod)}
        self.attrs = {'target_lod': target_offset_lod}
        self.outputs = {'Out': (x, [target_lod])}

    def test_check_output(self):
        if False:
            return 10
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        if False:
            print('Hello World!')
        self.check_grad(['X'], 'Out', check_dygraph=False)

class TestLodResetOpByInput(OpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.op_type = 'lod_reset'
        x = np.random.random((10, 20)).astype('float64')
        lod = [[3, 2, 5]]
        target_offset_lod = [0, 4, 7, 10]
        target_lod = [4, 3, 3]
        self.inputs = {'X': (x, lod), 'Y': np.array([target_offset_lod]).astype('int32')}
        self.outputs = {'Out': (x, [target_lod])}

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        if False:
            print('Hello World!')
        self.check_grad(['X'], 'Out', no_grad_set=set('Y'), check_dygraph=False)

class TestLodResetOpBoth(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'lod_reset'
        x = np.random.random((10, 20)).astype('float64')
        lod = [[3, 2, 5]]
        target_offset_lod_attr = [0, 7, 10]
        target_offset_lod_in = [0, 4, 7, 10]
        target_lod_in = [4, 3, 3]
        self.inputs = {'X': (x, lod), 'Y': np.array(target_offset_lod_in).astype('int32')}
        self.attrs = {'target_lod': target_offset_lod_attr}
        self.outputs = {'Out': (x, [target_lod_in])}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_grad(['X'], 'Out', no_grad_set=set('Y'), check_dygraph=False)

class TestLodResetOpYIsLoDTensor(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'lod_reset'
        x = np.random.random((10, 20)).astype('float64')
        lod = [[3, 2, 5]]
        y = np.random.random((10, 10)).astype('float64')
        target_lod = [[4, 3, 3]]
        self.inputs = {'X': (x, lod), 'Y': (y, target_lod)}
        self.outputs = {'Out': (x, target_lod)}

    def test_check_output(self):
        if False:
            return 10
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        if False:
            return 10
        self.check_grad(['X'], 'Out', no_grad_set=set('Y'), check_dygraph=False)

class TestLodAppendOpByAttr(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'lod_reset'
        x = np.random.random((10, 20)).astype('float64')
        lod = [[3, 2, 5]]
        target_offset_lod = list(range(11))
        self.inputs = {'X': (x, lod)}
        out_lod = [[3, 2, 5], [1] * 10]
        self.attrs = {'target_lod': target_offset_lod, 'append': True}
        self.outputs = {'Out': (x, out_lod)}

    def test_check_output(self):
        if False:
            return 10
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        if False:
            print('Hello World!')
        self.check_grad(['X'], 'Out', check_dygraph=False)
if __name__ == '__main__':
    unittest.main()