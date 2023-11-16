import sys
sys.path.append('..')
from legacy_test.parallel_dygraph_sparse_embedding import SimpleNet, TestSparseEmbedding, fake_sample_reader
from legacy_test.test_dist_base import runtime_main
import paddle
batch_size = 4
batch_num = 200
hidden_size = 10
vocab_size = 10
num_steps = 3
init_scale = 0.1

class TestSparseEmbeddingOverHeight(TestSparseEmbedding):

    def get_model(self):
        if False:
            while True:
                i = 10
        model = SimpleNet(hidden_size=hidden_size, vocab_size=vocab_size, num_steps=num_steps, init_scale=init_scale, is_sparse=True)
        train_reader = paddle.batch(fake_sample_reader(), batch_size=batch_size, drop_last=True)
        optimizer = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
        return (model, train_reader, optimizer)
if __name__ == '__main__':
    runtime_main(TestSparseEmbeddingOverHeight)