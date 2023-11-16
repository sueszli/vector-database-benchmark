import unittest
import numpy as np
from op_test import OpTest
import paddle
from paddle.base.framework import Program, program_guard
from paddle.pir_utils import test_with_pir_api

class TestGatherTreeOp(OpTest):

    def setUp(self):
        if False:
            return 10
        self.op_type = 'gather_tree'
        self.python_api = paddle.nn.functional.gather_tree
        (max_length, batch_size, beam_size) = (5, 2, 2)
        ids = np.random.randint(0, high=10, size=(max_length, batch_size, beam_size))
        parents = np.random.randint(0, high=beam_size, size=(max_length, batch_size, beam_size))
        self.inputs = {'Ids': ids, 'Parents': parents}
        self.outputs = {'Out': self.backtrace(ids, parents)}

    def test_check_output(self):
        if False:
            print('Hello World!')
        self.check_output(check_pir=True)

    @staticmethod
    def backtrace(ids, parents):
        if False:
            print('Hello World!')
        out = np.zeros_like(ids)
        (max_length, batch_size, beam_size) = ids.shape
        for batch in range(batch_size):
            for beam in range(beam_size):
                out[max_length - 1, batch, beam] = ids[max_length - 1, batch, beam]
                parent = parents[max_length - 1, batch, beam]
                for step in range(max_length - 2, -1, -1):
                    out[step, batch, beam] = ids[step, batch, parent]
                    parent = parents[step, batch, parent]
        return out

class TestGatherTreeOpAPI(unittest.TestCase):

    @test_with_pir_api
    def test_case(self):
        if False:
            i = 10
            return i + 15
        paddle.enable_static()
        ids = paddle.static.data(name='ids', shape=[5, 2, 2], dtype='int64')
        parents = paddle.static.data(name='parents', shape=[5, 2, 2], dtype='int64')
        final_sequences = paddle.nn.functional.gather_tree(ids, parents)
        paddle.disable_static()

    def test_case2(self):
        if False:
            print('Hello World!')
        ids = paddle.to_tensor([[[2, 2], [6, 1]], [[3, 9], [6, 1]], [[0, 1], [9, 0]]])
        parents = paddle.to_tensor([[[0, 0], [1, 1]], [[1, 0], [1, 0]], [[0, 0], [0, 1]]])
        final_sequences = paddle.nn.functional.gather_tree(ids, parents)

class TestGatherTreeOpError(unittest.TestCase):

    @test_with_pir_api
    def test_errors(self):
        if False:
            i = 10
            return i + 15
        paddle.enable_static()
        with program_guard(Program(), Program()):
            ids = paddle.static.data(name='ids', shape=[5, 2, 2], dtype='int64')
            parents = paddle.static.data(name='parents', shape=[5, 2, 2], dtype='int64')

            def test_Variable_ids():
                if False:
                    return 10
                np_ids = np.random.random((5, 2, 2), dtype='int64')
                paddle.nn.functional.gather_tree(np_ids, parents)
            self.assertRaises(TypeError, test_Variable_ids)

            def test_Variable_parents():
                if False:
                    i = 10
                    return i + 15
                np_parents = np.random.random((5, 2, 2), dtype='int64')
                paddle.nn.functional.gather_tree(ids, np_parents)
            self.assertRaises(TypeError, test_Variable_parents)
        paddle.disable_static()

class TestGatherTreeOpErrorForOthers(unittest.TestCase):

    def test_errors(self):
        if False:
            print('Hello World!')
        paddle.enable_static()
        with program_guard(Program(), Program()):
            ids = paddle.static.data(name='ids', shape=[5, 2, 2], dtype='int64')
            parents = paddle.static.data(name='parents', shape=[5, 2, 2], dtype='int64')

            def test_type_ids():
                if False:
                    i = 10
                    return i + 15
                bad_ids = paddle.static.data(name='bad_ids', shape=[5, 2, 2], dtype='float32')
                paddle.nn.functional.gather_tree(bad_ids, parents)
            self.assertRaises(TypeError, test_type_ids)

            def test_type_parents():
                if False:
                    for i in range(10):
                        print('nop')
                bad_parents = paddle.static.data(name='bad_parents', shape=[5, 2, 2], dtype='float32')
                paddle.nn.functional.gather_tree(ids, bad_parents)
            self.assertRaises(TypeError, test_type_parents)

            def test_ids_ndim():
                if False:
                    i = 10
                    return i + 15
                bad_ids = paddle.static.data(name='bad_test_ids', shape=[5, 2], dtype='int64')
                paddle.nn.functional.gather_tree(bad_ids, parents)
            self.assertRaises(ValueError, test_ids_ndim)

            def test_parents_ndim():
                if False:
                    print('Hello World!')
                bad_parents = paddle.static.data(name='bad_test_parents', shape=[5, 2], dtype='int64')
                paddle.nn.functional.gather_tree(ids, bad_parents)
            self.assertRaises(ValueError, test_parents_ndim)
        paddle.disable_static()
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()