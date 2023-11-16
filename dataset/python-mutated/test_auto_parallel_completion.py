import unittest
import unittest.mock
import paddle
import paddle.nn.functional as F
from paddle import nn, static, tensor, utils
from paddle.distributed.auto_parallel.static.completion import Completer
from paddle.distributed.auto_parallel.static.dist_context import DistributedContext
from paddle.distributed.fleet import auto
paddle.enable_static()
_global_parallel_strategy = None
_global_process_mesh = None
_global_process_mesh2 = None

class MLPLayer(nn.Layer):

    def __init__(self, hidden_size=1024, intermediate_size=4 * 1024, dropout_ratio=0.1, initializer_range=0.02):
        if False:
            i = 10
            return i + 15
        super().__init__()
        d_model = hidden_size
        dim_feedforward = intermediate_size
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0.0, std=initializer_range))
        bias_attr = None
        self.linear0 = nn.Linear(d_model, dim_feedforward, weight_attr, bias_attr=bias_attr)
        self.linear1 = nn.Linear(dim_feedforward, d_model, weight_attr, bias_attr=bias_attr)
        self.norm = nn.LayerNorm(d_model, epsilon=1e-05)
        self.dropout = nn.Dropout(dropout_ratio, mode='upscale_in_train')

    def forward(self, input):
        if False:
            while True:
                i = 10
        if _global_parallel_strategy in ['mp', 'dp_mp']:
            auto.shard_tensor(self.linear0.weight, process_mesh=_global_process_mesh, shard_spec=[None, 'mp'])
            auto.shard_tensor(self.linear1.weight, process_mesh=_global_process_mesh, shard_spec=['mp', None])
        out = self.norm(input)
        out = self.linear0(out)
        out = F.gelu(out, approximate=True)
        out = self.linear1(out)
        out = self.dropout(out)
        return out

def mlp_pretrain_forward(train_program, start_program):
    if False:
        return 10
    with static.program_guard(train_program, start_program), utils.unique_name.guard():
        batch_size = 4
        hidden_size = 1024
        sequence_len = 512
        input = static.data(name='input', shape=[batch_size, sequence_len, hidden_size], dtype='float32')
        if _global_parallel_strategy in ['dp', 'dp_mp']:
            auto.shard_tensor(input, process_mesh=_global_process_mesh, shard_spec=['dp', None, None])
        mlp = MLPLayer(hidden_size=hidden_size, intermediate_size=4 * hidden_size, dropout_ratio=0.1, initializer_range=0.02)
        out = mlp(input)
    return (train_program, start_program)

class TestMLPAutoCompletion(unittest.TestCase):

    def test_mlp_dp(self):
        if False:
            return 10
        global _global_parallel_strategy
        _global_parallel_strategy = 'dp'
        global _global_process_mesh
        _global_process_mesh = auto.ProcessMesh(mesh=[0, 1, 2, 3], dim_names=['dp'])
        train_program = static.Program()
        start_program = static.Program()
        dist_context = DistributedContext()
        (train_program, start_program) = mlp_pretrain_forward(train_program, start_program)
        completer = Completer(dist_context)
        complete_train_program = completer.complete_forward_annotation(train_program)
        self.assertTrue(dist_context.validate_dist_attr_for_program())

    def test_mlp_mp(self):
        if False:
            while True:
                i = 10
        global _global_parallel_strategy
        _global_parallel_strategy = 'mp'
        global _global_process_mesh
        _global_process_mesh = auto.ProcessMesh(mesh=[0, 1, 2, 3], dim_names=['mp'])
        train_program = static.Program()
        start_program = static.Program()
        dist_context = DistributedContext()
        (train_program, start_program) = mlp_pretrain_forward(train_program, start_program)
        completer = Completer(dist_context)
        complete_train_program = completer.complete_forward_annotation(train_program)
        self.assertTrue(dist_context.validate_dist_attr_for_program())

    def test_mlp_dp_mp(self):
        if False:
            print('Hello World!')
        global _global_parallel_strategy
        _global_parallel_strategy = 'dp_mp'
        global _global_process_mesh
        _global_process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2, 3], [4, 5, 6, 7]], dim_names=['dp', 'mp'])
        train_program = static.Program()
        start_program = static.Program()
        dist_context = DistributedContext()
        (train_program, start_program) = mlp_pretrain_forward(train_program, start_program)
        completer = Completer(dist_context)
        complete_train_program = completer.complete_forward_annotation(train_program)
        self.assertTrue(dist_context.validate_dist_attr_for_program())

class AttentionLayer(nn.Layer):

    def __init__(self, hidden_size=1024, sequence_len=512, intermediate_size=4 * 1024, num_heads=16, dropout_ratio=0.1, initializer_range=0.02):
        if False:
            while True:
                i = 10
        super().__init__()
        self.hidden_size = hidden_size
        self.sequence_len = sequence_len
        self.embed_dim = self.hidden_size
        self.kdim = self.embed_dim
        self.vdim = self.embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        self.dropout_ratio = dropout_ratio
        self.initializer_range = initializer_range
        self.training = True
        self.attn_mask = None
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0.0, std=initializer_range))
        bias_attr = None
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, weight_attr, bias_attr=bias_attr)
        self.k_proj = nn.Linear(self.kdim, self.embed_dim, weight_attr, bias_attr=bias_attr)
        self.v_proj = nn.Linear(self.vdim, self.embed_dim, weight_attr, bias_attr=bias_attr)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, weight_attr, bias_attr=bias_attr)

    def forward(self, input):
        if False:
            i = 10
            return i + 15
        if _global_parallel_strategy in ['dp', 'dp_mp']:
            auto.shard_tensor(input, process_mesh=_global_process_mesh, shard_spec=['dp', None, None])
        q = self.q_proj(input)
        q = tensor.reshape(x=q, shape=[0, 0, self.num_heads, self.head_dim])
        q = tensor.transpose(x=q, perm=[0, 2, 1, 3])
        k = self.k_proj(input)
        v = self.v_proj(input)
        if _global_parallel_strategy in ['mp', 'dp_mp']:
            auto.shard_tensor(self.q_proj.weight, process_mesh=_global_process_mesh, shard_spec=[None, 'mp'])
            auto.shard_tensor(self.k_proj.weight, process_mesh=_global_process_mesh, shard_spec=[None, 'mp'])
            auto.shard_tensor(self.v_proj.weight, process_mesh=_global_process_mesh, shard_spec=[None, 'mp'])
        k = tensor.reshape(x=k, shape=[0, 0, self.num_heads, self.head_dim])
        k = tensor.transpose(x=k, perm=[0, 2, 1, 3])
        v = tensor.reshape(x=v, shape=[0, 0, self.num_heads, self.head_dim])
        v = tensor.transpose(x=v, perm=[0, 2, 1, 3])
        product = tensor.matmul(x=q, y=k, transpose_y=True)
        product = tensor.scale(product, scale=self.head_dim ** (-0.5))
        if self.attn_mask is not None:
            product = product + self.attn_mask
        weights = F.softmax(product)
        if self.dropout_ratio:
            weights = F.dropout(weights, self.dropout_ratio, training=self.training, mode='upscale_in_train')
        out = tensor.matmul(weights, v)
        out = tensor.transpose(out, perm=[0, 2, 1, 3])
        out = tensor.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])
        out = self.out_proj(out)
        if _global_parallel_strategy in ['mp', 'dp_mp']:
            auto.shard_tensor(self.out_proj.weight, process_mesh=_global_process_mesh, shard_spec=['mp', None])
        return out

def attn_pretrain_forward(train_program, start_program):
    if False:
        while True:
            i = 10
    with static.program_guard(train_program, start_program), utils.unique_name.guard():
        batch_size = 4
        hidden_size = 1024
        sequence_len = 512
        input = static.data(name='query', shape=[batch_size, sequence_len, hidden_size], dtype='float32')
        attn = AttentionLayer(hidden_size=hidden_size, sequence_len=sequence_len, intermediate_size=4 * hidden_size, num_heads=16, dropout_ratio=0.1, initializer_range=0.02)
        out = attn(input)
    return (train_program, start_program)

class TestAttentionAutoCompletion(unittest.TestCase):

    def test_attn_dp(self):
        if False:
            while True:
                i = 10
        global _global_parallel_strategy
        _global_parallel_strategy = 'dp'
        global _global_process_mesh
        _global_process_mesh = auto.ProcessMesh(mesh=[0, 1, 2, 3], dim_names=['dp'])
        train_program = static.Program()
        start_program = static.Program()
        dist_context = DistributedContext()
        (train_program, start_program) = attn_pretrain_forward(train_program, start_program)
        completer = Completer(dist_context)
        complete_train_program = completer.complete_forward_annotation(train_program)
        self.assertTrue(dist_context.validate_dist_attr_for_program())

    def test_attn_mp(self):
        if False:
            for i in range(10):
                print('nop')
        global _global_parallel_strategy
        _global_parallel_strategy = 'mp'
        global _global_process_mesh
        _global_process_mesh = auto.ProcessMesh(mesh=[0, 1, 2, 3], dim_names=['mp'])
        train_program = static.Program()
        start_program = static.Program()
        dist_context = DistributedContext()
        (train_program, start_program) = attn_pretrain_forward(train_program, start_program)
        completer = Completer(dist_context)
        complete_train_program = completer.complete_forward_annotation(train_program)
        self.assertTrue(dist_context.validate_dist_attr_for_program())

    def test_attn_dp_mp(self):
        if False:
            i = 10
            return i + 15
        global _global_parallel_strategy
        _global_parallel_strategy = 'dp_mp'
        global _global_process_mesh
        _global_process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2, 3], [4, 5, 6, 7]], dim_names=['dp', 'mp'])
        train_program = static.Program()
        start_program = static.Program()
        dist_context = DistributedContext()
        (train_program, start_program) = attn_pretrain_forward(train_program, start_program)
        completer = Completer(dist_context)
        complete_train_program = completer.complete_forward_annotation(train_program)
        self.assertTrue(dist_context.validate_dist_attr_for_program())

class DecoderLayer(nn.Layer):

    def __init__(self, vocab_size=32768, hidden_size=1024, sequence_len=512, max_position_embeddings=512, intermediate_size=4 * 1024, num_heads=16, dropout_ratio=0.1, initializer_range=0.02):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.sequence_len = sequence_len
        self.embed_dim = self.hidden_size
        self.kdim = self.embed_dim
        self.vdim = self.embed_dim
        self.num_heads = num_heads
        self.dropout_ratio = dropout_ratio
        self.initializer_range = initializer_range
        self.training = True
        self.attn_mask = None
        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        self.word_embeddings = nn.Embedding(self.vocab_size, self.hidden_size, weight_attr=paddle.ParamAttr(name='word_embeddings', initializer=nn.initializer.Normal(mean=0.0, std=self.initializer_range)))
        self.position_embeddings = nn.Embedding(self.max_position_embeddings, self.hidden_size, weight_attr=paddle.ParamAttr(name='pos_embeddings', initializer=nn.initializer.Normal(mean=0.0, std=self.initializer_range)))
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0.0, std=self.initializer_range))
        bias_attr = None
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, weight_attr, bias_attr=bias_attr)
        self.k_proj = nn.Linear(self.kdim, self.embed_dim, weight_attr, bias_attr=bias_attr)
        self.v_proj = nn.Linear(self.vdim, self.embed_dim, weight_attr, bias_attr=bias_attr)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, weight_attr, bias_attr=bias_attr)
        intermediate_size = 4 * self.hidden_size
        d_model = self.hidden_size
        dim_feedforward = intermediate_size
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0.0, std=self.initializer_range))
        bias_attr = None
        self.linear0 = nn.Linear(d_model, dim_feedforward, weight_attr, bias_attr=bias_attr)
        self.linear1 = nn.Linear(dim_feedforward, d_model, weight_attr, bias_attr=bias_attr)
        self.norm1 = nn.LayerNorm(d_model, epsilon=1e-05)
        self.norm2 = nn.LayerNorm(d_model, epsilon=1e-05)
        self.dropout1 = nn.Dropout(self.dropout_ratio)
        self.dropout2 = nn.Dropout(self.dropout_ratio, mode='upscale_in_train')
        self.dropout3 = nn.Dropout(self.dropout_ratio, mode='upscale_in_train')

    def forward(self, input_ids, position_ids):
        if False:
            for i in range(10):
                print('nop')
        if _global_parallel_strategy in ['dp', 'dp_mp']:
            auto.shard_tensor(input_ids, process_mesh=_global_process_mesh, shard_spec=['dp', None])
        input_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        if _global_parallel_strategy in ['mp', 'dp_mp']:
            auto.shard_tensor(self.word_embeddings.weight, process_mesh=_global_process_mesh, shard_spec=['mp', None])
        embeddings = input_embeddings + position_embeddings
        embeddings = self.dropout1(embeddings)
        target = self.norm1(embeddings)
        q = self.q_proj(target)
        q = tensor.reshape(x=q, shape=[0, 0, self.num_heads, self.head_dim])
        q = tensor.transpose(x=q, perm=[0, 2, 1, 3])
        k = self.k_proj(target)
        v = self.v_proj(target)
        if _global_parallel_strategy in ['mp', 'dp_mp']:
            auto.shard_tensor(self.q_proj.weight, process_mesh=_global_process_mesh, shard_spec=[None, 'mp'])
            auto.shard_tensor(self.k_proj.weight, process_mesh=_global_process_mesh, shard_spec=[None, 'mp'])
            auto.shard_tensor(self.v_proj.weight, process_mesh=_global_process_mesh, shard_spec=[None, 'mp'])
        k = tensor.reshape(x=k, shape=[0, 0, self.num_heads, self.head_dim])
        k = tensor.transpose(x=k, perm=[0, 2, 1, 3])
        v = tensor.reshape(x=v, shape=[0, 0, self.num_heads, self.head_dim])
        v = tensor.transpose(x=v, perm=[0, 2, 1, 3])
        product = tensor.matmul(x=q, y=k, transpose_y=True)
        product = tensor.scale(product, scale=self.head_dim ** (-0.5))
        if self.attn_mask is not None:
            product = product + self.attn_mask
        weights = F.softmax(product)
        if self.dropout_ratio:
            weights = F.dropout(weights, self.dropout_ratio, training=self.training, mode='upscale_in_train')
        out = tensor.matmul(weights, v)
        out = tensor.transpose(out, perm=[0, 2, 1, 3])
        out = tensor.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])
        out = self.out_proj(out)
        if _global_parallel_strategy in ['mp', 'dp_mp']:
            auto.shard_tensor(self.out_proj.weight, process_mesh=_global_process_mesh, shard_spec=['mp', None])
        residual = embeddings + self.dropout2(out)
        out0 = self.norm2(residual)
        out1 = self.linear0(out0)
        out2 = F.gelu(out1, approximate=True)
        out3 = self.linear1(out2)
        if _global_parallel_strategy in ['mp', 'dp_mp']:
            auto.shard_tensor(self.linear0.weight, process_mesh=_global_process_mesh, shard_spec=[None, 'mp'])
            auto.shard_tensor(self.linear1.weight, process_mesh=_global_process_mesh, shard_spec=['mp', None])
        final = residual + self.dropout3(out3)
        return final

def decoder_pretrain_forward(train_program, start_program):
    if False:
        while True:
            i = 10
    with static.program_guard(train_program, start_program), utils.unique_name.guard():
        batch_size = 4
        hidden_size = 1024
        sequence_len = 512
        input_ids = static.data(name='input_ids', shape=[batch_size, sequence_len], dtype='int64')
        position_ids = static.data(name='position_ids', shape=[batch_size, sequence_len], dtype='int64')
        decoder = DecoderLayer(vocab_size=32768, hidden_size=hidden_size, sequence_len=sequence_len, max_position_embeddings=512, intermediate_size=4 * hidden_size, num_heads=16, dropout_ratio=0.1, initializer_range=0.02)
        out = decoder(input_ids, position_ids)
    return (train_program, start_program)

class TestDecoderLayerAutoCompletion(unittest.TestCase):

    def test_decoder_dp(self):
        if False:
            for i in range(10):
                print('nop')
        global _global_parallel_strategy
        _global_parallel_strategy = 'dp'
        global _global_process_mesh
        _global_process_mesh = auto.ProcessMesh(mesh=[0, 1, 2, 3], dim_names=['dp'])
        train_program = static.Program()
        start_program = static.Program()
        dist_context = DistributedContext()
        (train_program, start_program) = decoder_pretrain_forward(train_program, start_program)
        completer = Completer(dist_context)
        complete_train_program = completer.complete_forward_annotation(train_program)
        self.assertTrue(dist_context.validate_dist_attr_for_program())

    def test_decoder_mp(self):
        if False:
            return 10
        global _global_parallel_strategy
        _global_parallel_strategy = 'mp'
        global _global_process_mesh
        _global_process_mesh = auto.ProcessMesh(mesh=[0, 1, 2, 3], dim_names=['mp'])
        train_program = static.Program()
        start_program = static.Program()
        dist_context = DistributedContext()
        (train_program, start_program) = decoder_pretrain_forward(train_program, start_program)
        completer = Completer(dist_context)
        complete_train_program = completer.complete_forward_annotation(train_program)
        self.assertTrue(dist_context.validate_dist_attr_for_program())

    def test_decoder_dp_mp(self):
        if False:
            print('Hello World!')
        global _global_parallel_strategy
        _global_parallel_strategy = 'dp_mp'
        global _global_process_mesh
        _global_process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2, 3], [4, 5, 6, 7]], dim_names=['dp', 'mp'])
        train_program = static.Program()
        start_program = static.Program()
        dist_context = DistributedContext()
        (train_program, start_program) = decoder_pretrain_forward(train_program, start_program)
        completer = Completer(dist_context)
        complete_train_program = completer.complete_forward_annotation(train_program)
        self.assertTrue(dist_context.validate_dist_attr_for_program())
if __name__ == '__main__':
    unittest.main()