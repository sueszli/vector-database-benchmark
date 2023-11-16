import argparse
import random
import unittest
import numpy as np
from dygraph_to_static_utils_new import Dy2StTestBase, test_legacy_and_pir
from simnet_dygraph_model_v2 import BOW, HingeLoss
import paddle
SEED = 102
random.seed(SEED)

def create_conf_dict():
    if False:
        print('Hello World!')
    conf_dict = {}
    conf_dict['task_mode'] = 'pairwise'
    conf_dict['net'] = {'emb_dim': 128, 'bow_dim': 128, 'hidden_dim': 128}
    conf_dict['loss'] = {'margin': 0.1}
    return conf_dict

def parse_args():
    if False:
        return 10
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help="Total examples' number in batch for training.")
    parser.add_argument('--seq_len', type=int, default=32, help='The length of each sentence.')
    parser.add_argument('--epoch', type=int, default=1, help='The number of training epoch.')
    parser.add_argument('--fake_sample_size', type=int, default=128, help='The number of samples of fake data.')
    args = parser.parse_args([])
    return args
args = parse_args()

def fake_vocabulary():
    if False:
        for i in range(10):
            print('nop')
    vocab = {}
    vocab['<unk>'] = 0
    for i in range(26):
        c = chr(ord('a') + i)
        vocab[c] = i + 1
    return vocab
vocab = fake_vocabulary()

class FakeReaderProcessor(paddle.io.Dataset):

    def __init__(self, args, vocab, length):
        if False:
            i = 10
            return i + 15
        self.vocab = vocab
        self.seq_len = args.seq_len
        self.sample_size = args.fake_sample_size
        self.data_samples = []
        for i in range(self.sample_size):
            query = [random.randint(0, 26) for i in range(self.seq_len)]
            pos_title = query[:]
            neg_title = [26 - q for q in query]
            self.data_samples.append(np.array([query, pos_title, neg_title]).astype(np.int64))
        self.query = []
        self.pos_title = []
        self.neg_title = []
        self._init_data(length)

    def get_reader(self, mode, epoch=0):
        if False:
            i = 10
            return i + 15

        def reader_with_pairwise():
            if False:
                i = 10
                return i + 15
            if mode == 'train':
                for i in range(self.sample_size):
                    yield self.data_samples[i]
        return reader_with_pairwise

    def _init_data(self, length):
        if False:
            print('Hello World!')
        reader = self.get_reader('train', epoch=args.epoch)()
        for (i, yield_data) in enumerate(reader):
            if i >= length:
                break
            self.query.append(yield_data[0])
            self.pos_title.append(yield_data[1])
            self.neg_title.append(yield_data[2])

    def __getitem__(self, idx):
        if False:
            return 10
        return (self.query[idx], self.pos_title[idx], self.neg_title[idx])

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self.query)
simnet_process = FakeReaderProcessor(args, vocab, args.batch_size * (args.epoch + 1))

def train(conf_dict, to_static):
    if False:
        while True:
            i = 10
    '\n    train process\n    '
    paddle.jit.enable_to_static(to_static)
    if paddle.is_compiled_with_cuda():
        place = paddle.CUDAPlace(0)
    else:
        place = paddle.CPUPlace()
    paddle.disable_static(place)
    paddle.seed(SEED)
    paddle.framework.random._manual_program_seed(SEED)
    conf_dict['dict_size'] = len(vocab)
    conf_dict['seq_len'] = args.seq_len
    net = BOW(conf_dict)
    loss = HingeLoss(conf_dict)
    optimizer = paddle.optimizer.Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, parameters=net.parameters())
    metric = paddle.metric.Auc(name='auc')
    global_step = 0
    losses = []
    train_loader = paddle.io.DataLoader(simnet_process, batch_size=args.batch_size)
    for (left, pos_right, neg_right) in train_loader():
        left = paddle.reshape(left, shape=[-1, 1])
        pos_right = paddle.reshape(pos_right, shape=[-1, 1])
        neg_right = paddle.reshape(neg_right, shape=[-1, 1])
        net.train()
        global_step += 1
        (left_feat, pos_score) = net(left, pos_right)
        pred = pos_score
        (_, neg_score) = net(left, neg_right)
        avg_cost = loss.compute(pos_score, neg_score)
        losses.append(np.mean(avg_cost.numpy()))
        avg_cost.backward()
        optimizer.minimize(avg_cost)
        net.clear_gradients()
    paddle.enable_static()
    return losses

class TestSimnet(Dy2StTestBase):

    @test_legacy_and_pir
    def test_dygraph_static_same_loss(self):
        if False:
            return 10
        if paddle.is_compiled_with_cuda():
            paddle.base.set_flags({'FLAGS_cudnn_deterministic': True})
        conf_dict = create_conf_dict()
        dygraph_loss = train(conf_dict, to_static=False)
        static_loss = train(conf_dict, to_static=True)
        self.assertEqual(len(dygraph_loss), len(static_loss))
        for i in range(len(dygraph_loss)):
            self.assertAlmostEqual(dygraph_loss[i], static_loss[i])
if __name__ == '__main__':
    unittest.main()