import os
import unittest
import paddle

class TestAdamWFP16XPU(unittest.TestCase):

    def test_tensor_scale_value(self):
        if False:
            for i in range(10):
                print('nop')
        x = paddle.to_tensor([9.876, 5.432, 2.10987])
        self.assertEqual(x.get_tensor().get_xpu_scale_value(), -1)
        x.get_tensor().set_xpu_scale_value(-1.25)
        self.assertEqual(x.get_tensor().get_xpu_scale_value(), -1.25)

    def test_state_dict(self):
        if False:
            while True:
                i = 10
        os.environ['xpu_adamw_moment_dtype'] = 'fp16'
        linear = paddle.nn.Linear(10, 10)
        inp = paddle.rand([10, 10], dtype='float32')
        out = linear(inp)
        loss = paddle.mean(out)
        beta1 = paddle.to_tensor([0.9], dtype='float32')
        beta2 = paddle.to_tensor([0.99], dtype='float32')
        adam = paddle.optimizer.AdamW(learning_rate=0.1, parameters=linear.parameters(), beta1=beta1, beta2=beta2, weight_decay=0.01)
        out.backward()
        adam.step()
        state_dict_1 = adam.state_dict()
        self.assertTrue('linear_0.w_0_moment1_0.SCALE_VALUE' in state_dict_1)
        self.assertTrue('linear_0.b_0_moment1_0.SCALE_VALUE' in state_dict_1)
        state_dict_1['linear_0.w_0_moment1_0.SCALE_VALUE'] = 0.75
        state_dict_1['linear_0.b_0_moment1_0.SCALE_VALUE'] = 12.3125
        adam.set_state_dict(state_dict_1)
        state_dict_2 = adam.state_dict()
        self.assertTrue('linear_0.w_0_moment1_0.SCALE_VALUE' in state_dict_2)
        self.assertTrue('linear_0.b_0_moment1_0.SCALE_VALUE' in state_dict_2)
        self.assertEqual(state_dict_2['linear_0.w_0_moment1_0.SCALE_VALUE'], 0.75)
        self.assertEqual(state_dict_2['linear_0.b_0_moment1_0.SCALE_VALUE'], 12.3125)
if __name__ == '__main__':
    paddle.disable_static()
    unittest.main()