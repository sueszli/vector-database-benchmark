import unittest
import numpy as np
import paddle
import paddle.static
paddle.enable_static()

class TestIpuShard(unittest.TestCase):

    def _test(self):
        if False:
            print('Hello World!')
        main_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog):
            a = paddle.static.data(name='data', shape=[None, 1], dtype='int32')
            b = a + 2
            with paddle.static.ipu_shard_guard(index=1):
                c = b + 1
                with paddle.static.ipu_shard_guard(index=2):
                    d = c * 2
                with paddle.static.ipu_shard_guard(index=3):
                    e = d + 3
                    with paddle.static.ipu_shard_guard(index=1):
                        e = e + 3
                        with paddle.static.ipu_shard_guard(index=2):
                            e = e + 3
            with paddle.static.ipu_shard_guard(index=1):
                f = paddle.tensor.pow(e, 2.0)
            with paddle.static.ipu_shard_guard(index=2):
                g = f - 1
            h = g + 1
        ipu_index_list = []
        for op in main_prog.global_block().ops:
            if op.desc.has_attr('ipu_index'):
                ipu_index_list.append(op.desc.attr('ipu_index'))
        return ipu_index_list

    def test_ipu_shard(self):
        if False:
            while True:
                i = 10
        ipu_index_list = self._test()
        expected_ipu_index_list = [1, 2, 3, 1, 2, 1, 2]
        np.testing.assert_allclose(ipu_index_list, expected_ipu_index_list, rtol=1e-05, atol=0)

class TestIpuPipeline(unittest.TestCase):

    def _test(self):
        if False:
            for i in range(10):
                print('nop')
        main_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog):
            a = paddle.static.data(name='data', shape=[None, 1], dtype='int32')
            b = a + 2
            with paddle.static.ipu_shard_guard(stage=1):
                c = b + 1
                with paddle.static.ipu_shard_guard(stage=2):
                    d = c * 2
                with paddle.static.ipu_shard_guard(stage=3):
                    e = d + 3
                    with paddle.static.ipu_shard_guard(stage=1):
                        e = e + 3
                        with paddle.static.ipu_shard_guard(stage=2):
                            e = e + 3
            with paddle.static.ipu_shard_guard(stage=1):
                f = paddle.tensor.pow(e, 2.0)
            with paddle.static.ipu_shard_guard(stage=2):
                g = f - 1
            h = g + 1
        ipu_index_list = []
        for op in main_prog.global_block().ops:
            if op.desc.has_attr('ipu_stage'):
                ipu_index_list.append(op.desc.attr('ipu_stage'))
        return ipu_index_list

    def test_ipu_shard(self):
        if False:
            for i in range(10):
                print('nop')
        ipu_index_list = self._test()
        expected_ipu_index_list = [1, 2, 3, 1, 2, 1, 2]
        np.testing.assert_allclose(ipu_index_list, expected_ipu_index_list, rtol=1e-05, atol=0)
if __name__ == '__main__':
    unittest.main()