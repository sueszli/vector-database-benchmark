import unittest
from dygraph_to_static_utils_new import Dy2StTestBase, test_ast_only
import paddle

class TestSetDynamicShape(Dy2StTestBase):

    @test_ast_only
    def test_start(self):
        if False:
            return 10

        def dygraph_func(loop_number):
            if False:
                i = 10
                return i + 15
            mask = paddle.randn([2, 2])
            paddle.jit.dy2static.utils_helper.set_dynamic_shape(mask, [-1, 2])
            n = paddle.randn([1, 2])
            for i in range(loop_number):
                mask = paddle.concat([mask, n], axis=0)
                if mask.shape[0] == 5:
                    break
            return mask
        loop_num = paddle.to_tensor(10)
        expected_shape = dygraph_func(loop_num).shape
        actual_shape = paddle.jit.to_static(dygraph_func)(loop_num).shape
        self.assertEqual(expected_shape, actual_shape)
if __name__ == '__main__':
    unittest.main()