import random
import unittest
import paddle
from paddle.vision.models._utils import IntermediateLayerGetter

class TestBase:

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.init_model()
        self.model.eval()
        self.layer_names = [(order, name) for (order, (name, _)) in enumerate(self.model.named_children())]
        (self.start, self.end) = sorted(random.sample(self.layer_names, 2), key=lambda x: x[0])
        self.return_layers_dic = {self.start[1]: 'feat1', self.end[1]: 'feat2'}
        self.new_model = IntermediateLayerGetter(self.model, self.return_layers_dic)

    def init_model(self):
        if False:
            print('Hello World!')
        self.model = None

    @paddle.no_grad()
    def test_inter_result(self):
        if False:
            return 10
        inp = paddle.randn([1, 3, 80, 80])
        inter_oup = self.new_model(inp)
        for (layer_name, layer) in self.model.named_children():
            if isinstance(layer, paddle.nn.Linear) and inp.ndim == 4 or (len(layer.sublayers()) > 0 and isinstance(layer.sublayers()[0], paddle.nn.Linear) and (inp.ndim == 4)):
                inp = paddle.flatten(inp, 1)
            inp = layer(inp)
            if layer_name in self.return_layers_dic:
                feat_name = self.return_layers_dic[layer_name]
                self.assertTrue((inter_oup[feat_name] == inp).all())

class TestIntermediateLayerGetterResNet18(TestBase, unittest.TestCase):

    def init_model(self):
        if False:
            i = 10
            return i + 15
        self.model = paddle.vision.models.resnet18(pretrained=False)

class TestIntermediateLayerGetterDenseNet121(TestBase, unittest.TestCase):

    def init_model(self):
        if False:
            i = 10
            return i + 15
        self.model = paddle.vision.models.densenet121(pretrained=False)

class TestIntermediateLayerGetterVGG11(TestBase, unittest.TestCase):

    def init_model(self):
        if False:
            print('Hello World!')
        self.model = paddle.vision.models.vgg11(pretrained=False)

class TestIntermediateLayerGetterMobileNetV3Small(TestBase, unittest.TestCase):

    def init_model(self):
        if False:
            print('Hello World!')
        self.model = paddle.vision.models.MobileNetV3Small()

class TestIntermediateLayerGetterShuffleNetV2(TestBase, unittest.TestCase):

    def init_model(self):
        if False:
            for i in range(10):
                print('nop')
        self.model = paddle.vision.models.shufflenet_v2_x0_25()
if __name__ == '__main__':
    unittest.main()