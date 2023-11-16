import unittest
import paddle
from paddle import nn
from paddle.incubate.nn.layer.fused_transformer import FusedFeedForward, FusedMultiHeadAttention

class PreModel(nn.Layer):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.attn = FusedMultiHeadAttention(embed_dim=1024, num_heads=16, normalize_before=False)
        self.ffn = FusedFeedForward(d_model=1024, dim_feedforward=4096, normalize_before=False)

    def forward(self, x):
        if False:
            return 10
        x = self.attn(x)
        x = self.ffn(x)

class PostModel(nn.Layer):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.attn = FusedMultiHeadAttention(embed_dim=1024, num_heads=16, normalize_before=True)
        self.ffn = FusedFeedForward(d_model=1024, dim_feedforward=4096, normalize_before=True)

    def forward(self, x):
        if False:
            return 10
        x = self.attn(x)
        x = self.ffn(x)

class TestFusedTransformerWithAmpDecorator(unittest.TestCase):

    def get_model(self):
        if False:
            i = 10
            return i + 15
        self.pre_model = PreModel()
        self.post_model = PostModel()

    def test_run(self):
        if False:
            i = 10
            return i + 15
        self.get_model()
        pre_model = paddle.amp.decorate(models=self.pre_model, level='O2', save_dtype='float32')
        post_model = paddle.amp.decorate(models=self.post_model, level='O2', save_dtype='float32')
if __name__ == '__main__':
    unittest.main()