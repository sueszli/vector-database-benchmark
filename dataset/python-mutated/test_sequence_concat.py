import unittest
import numpy as np
from op_test import OpTest
import paddle

class TestSequenceConcat(OpTest):

    def setLoD(self):
        if False:
            for i in range(10):
                print('nop')
        self.lod1 = [7, 3]
        self.lod2 = [12, 8]
        self.out_lod = [19, 11]

    def setUp(self):
        if False:
            while True:
                i = 10
        x1 = np.random.random(size=(10, 80)).astype('float64')
        x2 = np.random.random(size=(20, 80)).astype('float64')
        self.setLoD()
        out = np.concatenate((x1[0:self.lod1[0]], x2[0:self.lod2[0]], x1[self.lod1[0]:], x2[self.lod2[0]:]))
        self.op_type = 'sequence_concat'
        self.inputs = {'X': [('x1', (x1, [self.lod1])), ('x2', (x2, [self.lod2]))]}
        self.outputs = {'Out': (out, [self.out_lod])}

    def test_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output()

    def test_dx(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_grad(inputs_to_check=['x1', 'x2'], output_names='Out')

class TestSequenceConcatCase2(TestSequenceConcat):

    def setLoD(self):
        if False:
            for i in range(10):
                print('nop')
        self.lod1 = [10, 0]
        self.lod2 = [12, 8]
        self.out_lod = [22, 8]

class TestSequenceConcatCase3(TestSequenceConcat):

    def setLoD(self):
        if False:
            print('Hello World!')
        self.lod1 = [10, 0]
        self.lod2 = [20, 0]
        self.out_lod = [30, 0]

class TestSequenceConcatCase4(TestSequenceConcat):

    def setLoD(self):
        if False:
            while True:
                i = 10
        self.lod1 = [0, 10]
        self.lod2 = [0, 20]
        self.out_lod = [0, 30]

class TestSequenceConcatCase5(TestSequenceConcat):

    def setLoD(self):
        if False:
            i = 10
            return i + 15
        self.lod1 = [0, 10]
        self.lod2 = [20, 0]
        self.out_lod = [20, 10]

class TestSequenceConcatOpError(unittest.TestCase):

    def test_errors(self):
        if False:
            print('Hello World!')

        def test_input_list():
            if False:
                return 10
            x_data = paddle.static.data(name='x', shape=[-1, 4], dtype='float32')
            paddle.static.nn.sequence_lod.sequence_concat(input=x_data)
        self.assertRaises(TypeError, test_input_list)

        def test_variable1():
            if False:
                return 10
            x1_data = np.array([[3, 5]]).astype('float32')
            y1_data = paddle.static.data(name='y1', shape=[-1, 4], dtype='float32')
            paddle.static.nn.sequence_lod.sequence_concat(input=[x1_data, y1_data])

        def test_variable2():
            if False:
                for i in range(10):
                    print('nop')
            x2_data = np.array([[3, 5]]).astype('float32')
            y2_data = paddle.static.data(name='y2', shape=[-1, 4], dtype='float32')
            paddle.static.nn.sequence_lod.sequence_concat(input=[y2_data, x2_data])
        for i in range(2):
            if i == 0:
                self.assertRaises(TypeError, test_variable1)
            else:
                self.assertRaises(TypeError, test_variable2)

        def test_dtype():
            if False:
                print('Hello World!')
            x3_data = paddle.static.data(name='x3', shape=[-1, 3, 5], dtype='int32')
            y3_data = paddle.static.data(name='y3', shape=[-1, 3, 5], dtype='int16')
            input_list = [x3_data, y3_data]
            paddle.static.nn.sequence_lod.sequence_concat(input=input_list)
        self.assertRaises(TypeError, test_dtype)

        def test_0_shape():
            if False:
                print('Hello World!')
            x4_data = paddle.static.data(name='x4', shape=[0], dtype='float32')
            y4_data = paddle.static.data(name='y4', shape=[1], dtype='float32')
            input_list = [x4_data, y4_data]
            paddle.static.nn.sequence_lod.sequence_concat(input=input_list)
        self.assertRaises(ValueError, test_0_shape)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()