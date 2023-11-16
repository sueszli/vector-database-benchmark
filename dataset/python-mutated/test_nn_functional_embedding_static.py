import unittest
import numpy as np
import paddle
from paddle import base
from paddle.nn import functional

class EmbeddingStatic(unittest.TestCase):

    def test_1(self):
        if False:
            for i in range(10):
                print('nop')
        prog = base.Program()
        with base.program_guard(prog):

            def test_bad_x():
                if False:
                    while True:
                        i = 10
                initializer = paddle.nn.initializer.Assign(np.random.random(size=(128, 100)))
                param_attr = base.ParamAttr(name='emb_weight', learning_rate=0.5, initializer=initializer, trainable=True)
                weight = prog.global_block().create_parameter((128, 100), attr=param_attr, dtype='float32')
                label = paddle.static.data(name='label', shape=[-1, 4], dtype='int64')
                emb = functional.embedding(x=label, weight=weight, sparse=True, name='embedding')
            test_bad_x()

    def test_2(self):
        if False:
            print('Hello World!')
        prog = base.Program()
        with base.program_guard(prog):

            def test_bad_x():
                if False:
                    for i in range(10):
                        print('nop')
                initializer = paddle.nn.initializer.Assign(np.random.random(size=(128, 100)))
                param_attr = base.ParamAttr(name='emb_weight', learning_rate=0.5, initializer=initializer, trainable=True)
                weight = prog.global_block().create_parameter((128, 100), attr=param_attr, dtype='float32')
                label = paddle.static.data(name='label', shape=[-1, 4], dtype='int32')
                emb = functional.embedding(x=label, weight=weight, padding_idx=129, sparse=True, name='embedding')
        with self.assertRaises(ValueError):
            test_bad_x()
if __name__ == '__main__':
    unittest.main()