import unittest
import numpy as np
from op_test_ipu import IPUOpTest
import paddle
import paddle.static

class TestBase(IPUOpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.set_atol()
        self.set_training()
        self.set_data_feed()
        self.set_feed_attr()
        self.set_op_attrs()

    def set_data_feed(self):
        if False:
            i = 10
            return i + 15
        x = np.random.uniform(size=[1, 2, 6, 10])
        self.feed_fp32 = {'x': x.astype(np.float32)}
        self.feed_fp16 = {'x': x.astype(np.float16)}

    def set_feed_attr(self):
        if False:
            i = 10
            return i + 15
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())
        self.feed_dtype = [x.dtype for x in self.feed_fp32.values()]

    def set_op_attrs(self):
        if False:
            i = 10
            return i + 15
        self.attrs = {}
        self.attrs['size'] = [12, 12]

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            while True:
                i = 10
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        out = paddle.nn.functional.interpolate(x, **self.attrs)
        self.fetch_list = [out.name]

    def run_model(self, exec_mode):
        if False:
            i = 10
            return i + 15
        self.run_op_test(exec_mode)

    def test(self):
        if False:
            i = 10
            return i + 15
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model()
                self.run_model(m)
        self.check()

class TestCase0(TestBase):

    def set_op_attrs(self):
        if False:
            while True:
                i = 10
        self.attrs = {}
        self.attrs['size'] = [3, 4]

class TestCase1(TestBase):

    def set_op_attrs(self):
        if False:
            print('Hello World!')
        self.attrs = {}
        self.attrs['scale_factor'] = [2, 1]

@unittest.skip('Only one of size or scale_factor should be defined')
class TestCase2(TestBase):

    def set_op_attrs(self):
        if False:
            return 10
        self.attrs = {'size': [12, 12], 'scale_factor': [2, 1]}

class TestCase3(TestBase):

    def set_op_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        self.attrs = {'scale_factor': 2.5}

class TestBilinear(TestBase):

    @property
    def fp16_enabled(self):
        if False:
            for i in range(10):
                print('nop')
        return False

    def set_atol(self):
        if False:
            i = 10
            return i + 15
        self.atol = 1e-06
        self.rtol = 1e-06
        self.atol_fp16 = 0.001
        self.rtol_fp16 = 0.001

    def set_op_attrs(self):
        if False:
            print('Hello World!')
        self.attrs = {'size': [12, 12], 'mode': 'bilinear'}

class TestBicubic(TestBase):

    @property
    def fp16_enabled(self):
        if False:
            i = 10
            return i + 15
        return False

    def set_atol(self):
        if False:
            print('Hello World!')
        self.atol = 1e-06
        self.rtol = 1e-06
        self.atol_fp16 = 0.001
        self.rtol_fp16 = 0.001

    def set_op_attrs(self):
        if False:
            i = 10
            return i + 15
        self.attrs = {'size': [12, 12], 'mode': 'bicubic'}

class TestTrilinear(TestBase):

    @property
    def fp16_enabled(self):
        if False:
            return 10
        return False

    def set_atol(self):
        if False:
            for i in range(10):
                print('nop')
        self.atol = 1e-06
        self.rtol = 1e-06
        self.atol_fp16 = 0.001
        self.rtol_fp16 = 0.001

    def set_data_feed(self):
        if False:
            i = 10
            return i + 15
        x = np.random.uniform(size=[2, 3, 3, 6, 10])
        self.feed_fp32 = {'x': x.astype(np.float32)}
        self.feed_fp16 = {'x': x.astype(np.float16)}

    def set_op_attrs(self):
        if False:
            i = 10
            return i + 15
        self.attrs = {'size': [12, 12, 12], 'mode': 'trilinear', 'data_format': 'NCDHW'}

class TestLinear(TestBase):

    @property
    def fp16_enabled(self):
        if False:
            return 10
        return False

    def set_atol(self):
        if False:
            return 10
        self.atol = 1e-06
        self.rtol = 1e-06
        self.atol_fp16 = 0.001
        self.rtol_fp16 = 0.001

    def set_data_feed(self):
        if False:
            i = 10
            return i + 15
        x = np.random.uniform(size=[3, 6, 10])
        self.feed_fp32 = {'x': x.astype(np.float32)}
        self.feed_fp16 = {'x': x.astype(np.float16)}

    def set_op_attrs(self):
        if False:
            print('Hello World!')
        self.attrs = {'size': [12], 'mode': 'linear', 'data_format': 'NCW'}

@unittest.skip('Transfer to Pool Op with 2-D ksize, now we only support 1-D ksize.')
class TestArea(TestBase):

    def set_data_feed(self):
        if False:
            print('Hello World!')
        x = np.random.uniform(size=[2, 3, 6, 6])
        self.feed_fp32 = {'x': x.astype(np.float32)}
        self.feed_fp16 = {'x': x.astype(np.float16)}

    def set_op_attrs(self):
        if False:
            i = 10
            return i + 15
        self.attrs = {'size': 12, 'mode': 'area'}

class TestAlignCorners(TestBase):

    @property
    def fp16_enabled(self):
        if False:
            for i in range(10):
                print('nop')
        return False

    def set_op_attrs(self):
        if False:
            i = 10
            return i + 15
        self.attrs = {'size': [12, 12], 'align_corners': True, 'mode': 'bilinear'}

class TestAlignMode(TestBase):

    def set_op_attrs(self):
        if False:
            while True:
                i = 10
        self.attrs = {'size': [12, 12], 'align_mode': 1}
if __name__ == '__main__':
    unittest.main()