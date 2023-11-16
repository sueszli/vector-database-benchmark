import unittest
import numpy as np
from paddle.base import core

class EagerStringTensorTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.str_arr = np.array([['15.4寸笔记本的键盘确实爽，基本跟台式机差不多了，蛮喜欢数字小键盘，输数字特方便，样子也很美观，做工也相当不错'], ['One of the very best Three Stooges shorts ever.']])

    def test_constructor_with_args(self):
        if False:
            while True:
                i = 10
        ST1 = core.eager.StringTensor()
        self.assertEqual(ST1.name, 'generated_string_tensor_0')
        self.assertEqual(ST1.shape, [])
        self.assertEqual(ST1.numpy(), '')
        shape = [2, 3]
        ST2 = core.eager.StringTensor(shape, 'ST2')
        self.assertEqual(ST2.name, 'ST2')
        self.assertEqual(ST2.shape, shape)
        np.testing.assert_array_equal(ST2.numpy(), np.empty(shape, dtype=np.str_))
        ST3 = core.eager.StringTensor(self.str_arr, 'ST3')
        self.assertEqual(ST3.name, 'ST3')
        self.assertEqual(ST3.shape, list(self.str_arr.shape))
        np.testing.assert_array_equal(ST3.numpy(), self.str_arr)
        ST4 = core.eager.StringTensor(self.str_arr)
        self.assertEqual(ST4.name, 'generated_string_tensor_1')
        self.assertEqual(ST4.shape, list(self.str_arr.shape))
        np.testing.assert_array_equal(ST4.numpy(), self.str_arr)
        ST5 = core.eager.StringTensor(ST4)
        self.assertEqual(ST5.name, 'generated_string_tensor_2')
        self.assertEqual(ST5.shape, list(self.str_arr.shape))
        np.testing.assert_array_equal(ST5.numpy(), self.str_arr)
        ST6 = core.eager.StringTensor(ST5, 'ST6')
        self.assertEqual(ST6.name, 'ST6')
        self.assertEqual(ST6.shape, list(self.str_arr.shape))
        np.testing.assert_array_equal(ST6.numpy(), self.str_arr)
        for st in [ST1, ST2, ST3, ST4, ST5, ST6]:
            self.assertTrue(st.place._equals(core.CPUPlace()))

    def test_constructor_with_kwargs(self):
        if False:
            return 10
        shape = [2, 3]
        ST1 = core.eager.StringTensor(dims=shape, name='ST1')
        self.assertEqual(ST1.name, 'ST1')
        self.assertEqual(ST1.shape, shape)
        np.testing.assert_array_equal(ST1.numpy(), np.empty(shape, dtype=np.str_))
        ST2 = core.eager.StringTensor(self.str_arr, name='ST2')
        self.assertEqual(ST2.name, 'ST2')
        self.assertEqual(ST2.shape, list(self.str_arr.shape))
        np.testing.assert_array_equal(ST2.numpy(), self.str_arr)
        ST3 = core.eager.StringTensor(ST2, name='ST3')
        self.assertEqual(ST3.name, 'ST3')
        self.assertEqual(ST3.shape, list(self.str_arr.shape))
        np.testing.assert_array_equal(ST3.numpy(), self.str_arr)
        ST4 = core.eager.StringTensor(value=ST2, name='ST4')
        self.assertEqual(ST4.name, 'ST4')
        self.assertEqual(ST4.shape, list(self.str_arr.shape))
        np.testing.assert_array_equal(ST4.numpy(), self.str_arr)
        for st in [ST1, ST2, ST3, ST4]:
            self.assertTrue(st.place._equals(core.CPUPlace()))
if __name__ == '__main__':
    unittest.main()