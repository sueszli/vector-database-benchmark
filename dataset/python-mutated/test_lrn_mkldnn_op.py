import unittest
from test_lrn_op import TestLRNOp

class TestLRNMKLDNNOp(TestLRNOp):

    def get_attrs(self):
        if False:
            print('Hello World!')
        attrs = TestLRNOp.get_attrs(self)
        attrs['use_mkldnn'] = True
        return attrs

    def test_check_output(self):
        if False:
            return 10
        self.check_output(atol=0.002, no_check_set=['MidOut'], check_dygraph=False)

    def test_check_grad_normal(self):
        if False:
            print('Hello World!')
        self.check_grad(['X'], 'Out', max_relative_error=0.01, check_dygraph=False)

class TestLRNMKLDNNOpWithIsTest(TestLRNMKLDNNOp):

    def get_attrs(self):
        if False:
            i = 10
            return i + 15
        attrs = TestLRNMKLDNNOp.get_attrs(self)
        attrs['is_test'] = True
        return attrs

    def test_check_grad_normal(self):
        if False:
            while True:
                i = 10

        def check_raise_is_test():
            if False:
                print('Hello World!')
            try:
                self.check_grad(['X'], 'Out', max_relative_error=0.01, check_dygraph=False)
            except Exception as e:
                t = 'is_test attribute should be set to False in training phase.'
                if t in str(e):
                    raise AttributeError
        self.assertRaises(AttributeError, check_raise_is_test)

class TestLRNMKLDNNOpNHWC(TestLRNMKLDNNOp):

    def init_test_case(self):
        if False:
            for i in range(10):
                print('nop')
        self.data_format = 'NHWC'
if __name__ == '__main__':
    from paddle import enable_static
    enable_static()
    unittest.main()