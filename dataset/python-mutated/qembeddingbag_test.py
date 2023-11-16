import numpy
import torch
import torch.ao.nn.quantized as nnq
from pt import configs
import operator_benchmark as op_bench
'\nMicrobenchmarks for qEmbeddingBag operators.\n'

class QEmbeddingBagBenchmark(op_bench.TorchBenchmarkBase):

    def init(self, embeddingbags, dim, mode, input_size, offset, sparse, include_last_offset, device):
        if False:
            i = 10
            return i + 15
        self.embedding = nnq.EmbeddingBag(num_embeddings=embeddingbags, embedding_dim=dim, mode=mode, include_last_offset=include_last_offset).to(device=device)
        numpy.random.seed((1 << 32) - 1)
        self.input = torch.tensor(numpy.random.randint(0, embeddingbags, input_size), device=device).long()
        offset = torch.LongTensor([offset], device=device)
        self.offset = torch.cat((offset, torch.tensor([self.input.size(0)], dtype=torch.long)), 0)
        self.inputs = {'input': self.input, 'offset': self.offset}
        self.set_module_name('qEmbeddingBag')

    def forward(self, input, offset):
        if False:
            for i in range(10):
                print('nop')
        return self.embedding(input, offset)
op_bench.generate_pt_test(configs.embeddingbag_short_configs, QEmbeddingBagBenchmark)
if __name__ == '__main__':
    op_bench.benchmark_runner.main()