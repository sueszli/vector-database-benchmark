import unittest
import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import LayerDesc, PipelineLayer, PipelineParallelWithInterleave
from paddle.nn import Layer

class ReshapeHelp(Layer):

    def __init__(self, shape):
        if False:
            return 10
        super().__init__()
        self.shape = shape

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        return x.reshape(shape=self.shape)

class MLPForVirtualStageLayerTest(PipelineLayer):

    def __init__(self, num_classes=10, **kwargs):
        if False:
            return 10
        self.num_classes = num_classes
        decs = [LayerDesc(nn.Linear, 2, self.num_classes), LayerDesc(nn.Linear, self.num_classes, 2), LayerDesc(nn.Linear, 2, self.num_classes), LayerDesc(nn.Linear, self.num_classes, 2), LayerDesc(nn.Linear, 2, self.num_classes), LayerDesc(nn.Linear, self.num_classes, 2), LayerDesc(nn.Linear, 2, self.num_classes), LayerDesc(nn.Linear, self.num_classes, 2)]
        super().__init__(layers=decs, loss_fn=nn.CrossEntropyLoss(), **kwargs)

class TestPipeLayerAPI(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        strategy = fleet.DistributedStrategy()
        self.pipeline_parallel_size = 2
        strategy.hybrid_configs = {'dp_degree': 1, 'mp_degree': 1, 'pp_degree': self.pipeline_parallel_size}
        strategy.pipeline_configs = {'accumulate_steps': 2}
        fleet.init(is_collective=True, strategy=strategy)
        self.rank = fleet.worker_index()
        self.hcg = fleet.get_hybrid_communicate_group()

    def test_pipelayer_desc(self):
        if False:
            print('Hello World!')
        pipe_model = MLPForVirtualStageLayerTest(seg_method='layer:Linear', num_stages=self.pipeline_parallel_size, num_virtual_pipeline_stages=2, recompute_interval=1, recompute_ctx={'mp_group': self.hcg.get_model_parallel_group(), 'offload': False, 'partition': False})
        assert len(pipe_model.parameters()) > 0
        model_chunks = pipe_model.get_model_chunks()
        assert model_chunks is not None
        assert len(model_chunks) == 2
        optimizer = paddle.optimizer.SGD(parameters=pipe_model.parameters())
        try:
            model_chunks[0](paddle.to_tensor([1.0, 2.0]))
            raise NotImplementedError
        except PermissionError:
            pass
        for i in range(len(model_chunks)):
            out = pipe_model(paddle.to_tensor([1.0, 2.0]), chunk_id=i)
            assert list(out.shape) == [2]
            out = F.relu(out)
            loss = paddle.mean(out)
            loss.backward()
        optimizer.step()
        dist_model = fleet.distributed_model(pipe_model)
        assert isinstance(dist_model, PipelineParallelWithInterleave)
if __name__ == '__main__':
    unittest.main()