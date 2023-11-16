import unittest
from cinn import framework
from test_utils import SingleOpTester

class OpTest_add_0(SingleOpTester):

    def create_target_data(self, inputs_data, attrs):
        if False:
            for i in range(10):
                print('nop')
        [X, Y] = inputs_data
        return X + Y

    def test_op(self):
        if False:
            for i in range(10):
                print('nop')
        attrs = framework.NodeAttr()
        attrs.set_attr('axis', 0)
        self.to_test_op([[100, 32], [100, 32]], [[100, 32]], 'elementwise_add', attrs)

class OpTest_add_1(SingleOpTester):

    def create_target_data(self, inputs_data, attrs):
        if False:
            i = 10
            return i + 15
        [X, Y] = inputs_data
        return X + Y

    def test_op(self):
        if False:
            print('Hello World!')
        attrs = framework.NodeAttr()
        attrs.set_attr('axis', 1)
        self.to_test_op([[3, 2], [2]], [[3, 2]], 'elementwise_add', attrs)

class OpTest_mul_0(SingleOpTester):

    def create_target_data(self, inputs_data, attrs):
        if False:
            for i in range(10):
                print('nop')
        [X, Y] = inputs_data
        return X * Y

    def test_op(self):
        if False:
            print('Hello World!')
        attrs = framework.NodeAttr()
        attrs.set_attr('axis', 0)
        self.to_test_op([[100, 32], [100, 32]], [[100, 32]], 'elementwise_mul', attrs)

class OpTest_mul_1(SingleOpTester):

    def create_target_data(self, inputs_data, attrs):
        if False:
            i = 10
            return i + 15
        [X, Y] = inputs_data
        return X * Y

    def test_op(self):
        if False:
            print('Hello World!')
        attrs = framework.NodeAttr()
        attrs.set_attr('axis', 1)
        self.to_test_op([[3, 2], [2]], [[3, 2]], 'elementwise_mul', attrs)

class OpTest_scale_0(SingleOpTester):

    def create_target_data(self, inputs_data, attrs):
        if False:
            return 10
        [X] = inputs_data
        return X * attrs.attr_store['scale'] + attrs.attr_store['bias']

    def test_op(self):
        if False:
            while True:
                i = 10
        attrs = framework.NodeAttr()
        attrs.set_attr('scale', 0.7)
        attrs.set_attr('bias', 0.3)
        self.to_test_op([[100, 32]], [[100, 32]], 'scale', attrs)

class OpTest_scale_1(SingleOpTester):

    def create_target_data(self, inputs_data, attrs):
        if False:
            print('Hello World!')
        [X] = inputs_data
        return (X + attrs.attr_store['bias']) * attrs.attr_store['scale']

    def test_op(self):
        if False:
            for i in range(10):
                print('nop')
        attrs = framework.NodeAttr()
        attrs.set_attr('scale', 0.6)
        attrs.set_attr('bias', 0.4)
        attrs.set_attr('bias_after_scale', False)
        self.to_test_op([[100, 32]], [[100, 32]], 'scale', attrs)
if __name__ == '__main__':
    unittest.main()