import unittest
import numpy as np
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import LayerDesc, PipelineLayer
from paddle.nn import Layer, Sequential

class ReshapeHelp(Layer):

    def __init__(self, shape):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.shape = shape

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        return x.reshape(shape=self.shape)

class AlexNet(Layer):

    def __init__(self, num_classes=10):
        if False:
            print('Hello World!')
        super().__init__()
        self.features = Sequential(nn.Conv2D(1, 64, kernel_size=11, stride=4, padding=5), nn.ReLU(), nn.MaxPool2D(kernel_size=2, stride=2), nn.Conv2D(64, 192, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2D(kernel_size=2, stride=2), nn.Conv2D(192, 384, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2D(384, 256, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2D(256, 256, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2D(kernel_size=2, stride=2))
        self.reshape_layer = ReshapeHelp(shape=[-1, 256])
        self.classifier = nn.Linear(256, num_classes)
        self.loss_fn = nn.loss.CrossEntropyLoss()

    def forward(self, x, y):
        if False:
            i = 10
            return i + 15
        x = self.features(x)
        x = self.reshape_layer(x)
        x = self.classifier(x)
        return self.loss_fn(x, y)

class AlexNetPipe(AlexNet):

    def to_layers(self):
        if False:
            i = 10
            return i + 15
        feat = [self.features[i] for i in range(len(self.features))]
        loss_fn = [self.reshape_layer, self.classifier]
        feat.extend(loss_fn)
        return feat

class AlexNetPipeDesc(PipelineLayer):

    def __init__(self, num_classes=10, **kwargs):
        if False:
            i = 10
            return i + 15
        self.num_classes = num_classes
        decs = [LayerDesc(nn.Conv2D, 1, 64, kernel_size=11, stride=4, padding=5), LayerDesc(nn.ReLU), LayerDesc(nn.MaxPool2D, kernel_size=2, stride=2), LayerDesc(nn.Conv2D, 64, 192, kernel_size=5, padding=2), F.relu, LayerDesc(nn.MaxPool2D, kernel_size=2, stride=2), LayerDesc(nn.Conv2D, 192, 384, kernel_size=3, padding=1), F.relu, LayerDesc(nn.Conv2D, 384, 256, kernel_size=3, padding=1), F.relu, LayerDesc(nn.Conv2D, 256, 256, kernel_size=3, padding=1), F.relu, LayerDesc(nn.MaxPool2D, kernel_size=2, stride=2), LayerDesc(ReshapeHelp, shape=[-1, 256]), LayerDesc(nn.Linear, 256, self.num_classes)]
        super().__init__(layers=decs, loss_fn=nn.CrossEntropyLoss(), **kwargs)

class TestPipeLayerAPI(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        strategy = fleet.DistributedStrategy()
        self.pipeline_parallel_size = 2
        strategy.hybrid_configs = {'dp_degree': 1, 'mp_degree': 1, 'pp_degree': self.pipeline_parallel_size}
        fleet.init(is_collective=True, strategy=strategy)
        self.hcg = fleet.get_hybrid_communicate_group()

    def test_pipelayer_desc(self):
        if False:
            while True:
                i = 10
        pipe_model = AlexNetPipeDesc(num_stages=self.pipeline_parallel_size)
        np.testing.assert_array_equal(len(pipe_model.parameters()), 6)

    def test_pipelayer_sequential(self):
        if False:
            i = 10
            return i + 15
        init_net = AlexNetPipe()
        pipe_model = PipelineLayer(layers=init_net.to_layers(), num_stages=self.pipeline_parallel_size, loss_fn=nn.CrossEntropyLoss())
        stage_id = self.hcg.get_stage_id()
        init_parameters = init_net.parameters()
        pipe_parameters = pipe_model.parameters()
        part_number = len(init_parameters) // 2
        if stage_id == 0:
            for idx in range(part_number):
                param_a = init_parameters[idx]
                param_b = pipe_parameters[idx]
                np.testing.assert_array_equal(param_a.name, param_b.name)
                np.testing.assert_allclose(param_a.numpy(), param_b.numpy())
        elif stage_id == 1:
            for idx in range(part_number):
                param_a = init_parameters[idx + part_number]
                param_b = pipe_parameters[idx]
                np.testing.assert_array_equal(param_a.name, param_b.name)
                np.testing.assert_allclose(param_a.numpy(), param_b.numpy())

    def test_pipelayer_segment_method(self):
        if False:
            for i in range(10):
                print('nop')
        init_net = AlexNetPipe()
        pipe_model = PipelineLayer(layers=init_net.to_layers(), num_stages=self.pipeline_parallel_size, seg_method=[0, 4], loss_fn=nn.CrossEntropyLoss())
        stage_id = self.hcg.get_stage_id()
        if stage_id == 0:
            np.testing.assert_array_equal(len(pipe_model.parameters()), 4)
        elif stage_id == 1:
            np.testing.assert_array_equal(len(pipe_model.parameters()), 8)
if __name__ == '__main__':
    unittest.main()