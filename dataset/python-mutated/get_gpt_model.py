import random
import sys
sys.path.append('../legacy_test')
import auto_parallel_gpt_model as modeling
import numpy as np
from auto_parallel_gpt_model import GPTForPretraining, GPTModel, GPTPretrainingCriterion
import paddle
from paddle.distributed.fleet import auto

class FakeDataset(paddle.io.Dataset):

    def __init__(self, num_samples, vocab_size=1000, sequence_len=512):
        if False:
            print('Hello World!')
        self.num_samples = num_samples
        self.sequence_len = sequence_len
        self.vocab_size = vocab_size

    def __getitem__(self, idx):
        if False:
            while True:
                i = 10
        paddle.seed(2021)
        np.random.seed(2021)
        random.seed(2021)
        tokens = np.random.randint(self.vocab_size, size=self.sequence_len)
        position_ids = np.arange(self.sequence_len)
        attention_mask = np.tril(np.ones(self.sequence_len)).reshape((1, self.sequence_len, self.sequence_len)).astype(np.float32)
        labels = np.random.randint(self.vocab_size, size=self.sequence_len)
        loss_mask = np.ones(self.sequence_len).astype(np.float32)
        return (tokens, position_ids, attention_mask, labels, loss_mask)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.num_samples

def create_data_holder(batch_size, vocab_size=1000, sequence_len=512):
    if False:
        i = 10
        return i + 15
    tokens = paddle.static.InputSpec(name='tokens', shape=[batch_size, sequence_len], dtype='int64')
    position_ids = paddle.static.InputSpec(name='position_ids', shape=[batch_size, sequence_len], dtype='int64')
    attention_mask = paddle.static.InputSpec(name='attention_mask', shape=[batch_size, 1, sequence_len, sequence_len], dtype='float32')
    labels = paddle.static.InputSpec(name='labels', shape=[batch_size, sequence_len], dtype='int64')
    loss_mask = paddle.static.InputSpec(name='loss_mask', shape=[batch_size, sequence_len], dtype='float32')
    return ([tokens, position_ids, attention_mask], [labels, loss_mask])

def generate_model(strategy, dropout_prob=0.0):
    if False:
        print('Hello World!')
    modeling.init_global()
    ranks = list(range(paddle.distributed.get_world_size()))
    modeling._global_process_mesh = auto.ProcessMesh(mesh=ranks, dim_names=['x'])
    if strategy == 'serial':
        modeling._global_parallel_strategy = 'serial'
    elif strategy == 'mp':
        modeling._global_parallel_strategy = 'mp'
    elif strategy == 'dp':
        modeling._global_parallel_strategy = 'dp'
    elif strategy == 'pp':
        modeling._global_parallel_strategy = 'pp'
        modeling.PP_MESH_LIST = [auto.ProcessMesh(mesh=[0]), auto.ProcessMesh(mesh=[1])]
    else:
        raise ValueError('Only support serial, mp2, dp2 and pp2.')
    gpt = GPTModel(vocab_size=1000, hidden_size=64, num_hidden_layers=2, num_attention_heads=8, intermediate_size=256, hidden_act='gelu', hidden_dropout_prob=dropout_prob, attention_probs_dropout_prob=dropout_prob, max_position_embeddings=1024, type_vocab_size=1, initializer_range=0.02, pad_token_id=0, eos_token_id=7, bos_token_id=0, eol_token_id=3, pp_degree=2 if strategy == 'pp' else None)
    model = GPTForPretraining(gpt, vocab_size=1000, hidden_size=64, initializer_range=0.02)
    criterion = GPTPretrainingCriterion()
    return (model, criterion)