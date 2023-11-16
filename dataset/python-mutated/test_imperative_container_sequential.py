import unittest
import numpy as np
import paddle
from paddle import base
from paddle.nn import Linear

class TestImperativeContainerSequential(unittest.TestCase):

    def test_sequential(self):
        if False:
            i = 10
            return i + 15
        data = np.random.uniform(-1, 1, [5, 10]).astype('float32')
        with base.dygraph.guard():
            data = base.dygraph.to_variable(data)
            model1 = paddle.nn.Sequential(Linear(10, 1), Linear(1, 2))
            res1 = model1(data)
            self.assertListEqual(res1.shape, [5, 2])
            model1[1] = Linear(1, 3)
            res1 = model1(data)
            self.assertListEqual(res1.shape, [5, 3])
            loss1 = paddle.mean(res1)
            loss1.backward()
            l1 = Linear(10, 1)
            l2 = Linear(1, 3)
            model2 = paddle.nn.Sequential(('l1', l1), ('l2', l2))
            self.assertEqual(len(model2), 2)
            res2 = model2(data)
            self.assertTrue(l1 is model2.l1)
            self.assertListEqual(res2.shape, res1.shape)
            self.assertEqual(len(model1.parameters()), len(model2.parameters()))
            del model2['l2']
            self.assertEqual(len(model2), 1)
            res2 = model2(data)
            self.assertListEqual(res2.shape, [5, 1])
            model2.add_sublayer('l3', Linear(1, 3))
            model2.add_sublayer('l4', Linear(3, 4))
            self.assertEqual(len(model2), 3)
            res2 = model2(data)
            self.assertListEqual(res2.shape, [5, 4])
            loss2 = paddle.mean(res2)
            loss2.backward()

    def test_sequential_list_params(self):
        if False:
            return 10
        data = np.random.uniform(-1, 1, [5, 10]).astype('float32')
        with base.dygraph.guard():
            data = base.dygraph.to_variable(data)
            model1 = paddle.nn.Sequential(Linear(10, 1), Linear(1, 2))
            res1 = model1(data)
            self.assertListEqual(res1.shape, [5, 2])
            model1[1] = Linear(1, 3)
            res1 = model1(data)
            self.assertListEqual(res1.shape, [5, 3])
            loss1 = paddle.mean(res1)
            loss1.backward()
            l1 = Linear(10, 1)
            l2 = Linear(1, 3)
            model2 = paddle.nn.Sequential(['l1', l1], ['l2', l2])
            self.assertEqual(len(model2), 2)
            res2 = model2(data)
            self.assertTrue(l1 is model2.l1)
            self.assertListEqual(res2.shape, res1.shape)
            self.assertEqual(len(model1.parameters()), len(model2.parameters()))
            del model2['l2']
            self.assertEqual(len(model2), 1)
            res2 = model2(data)
            self.assertListEqual(res2.shape, [5, 1])
            model2.add_sublayer('l3', Linear(1, 3))
            model2.add_sublayer('l4', Linear(3, 4))
            self.assertEqual(len(model2), 3)
            res2 = model2(data)
            self.assertListEqual(res2.shape, [5, 4])
            loss2 = paddle.mean(res2)
            loss2.backward()
if __name__ == '__main__':
    unittest.main()