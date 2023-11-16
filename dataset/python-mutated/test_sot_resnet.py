import unittest
from test_case_base import TestCaseBase, test_instruction_translator_cache_context
import paddle
from paddle.vision.models.resnet import resnet18

def resnet_call(x: paddle.Tensor, net: paddle.nn.Layer):
    if False:
        for i in range(10):
            print('nop')
    return net(x)

class TestResNet(TestCaseBase):

    def test_resnet_eval(self):
        if False:
            i = 10
            return i + 15
        x = paddle.rand((10, 3, 224, 224))
        net = resnet18(pretrained=False)
        net.eval()
        with test_instruction_translator_cache_context() as ctx:
            self.assert_results(resnet_call, x, net)
            self.assertEqual(ctx.translate_count, 1)
            self.assert_results(resnet_call, x, net)
            self.assertEqual(ctx.translate_count, 1)
            net.train()
            self.assert_results(resnet_call, x, net)
            self.assertEqual(ctx.translate_count, 2)

    def test_resnet_train(self):
        if False:
            i = 10
            return i + 15
        x = paddle.rand((10, 3, 224, 224))
        net = resnet18(pretrained=False)
        net.train()
        with test_instruction_translator_cache_context() as ctx:
            self.assert_results(resnet_call, x, net)
            self.assertEqual(ctx.translate_count, 1)
            self.assert_results(resnet_call, x, net)
            self.assertEqual(ctx.translate_count, 1)
            net.eval()
            self.assert_results(resnet_call, x, net)
            self.assertEqual(ctx.translate_count, 2)
if __name__ == '__main__':
    unittest.main()