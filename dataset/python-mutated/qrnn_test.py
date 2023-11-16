import torch
from torch import nn
import operator_benchmark as op_bench
'\nMicrobenchmarks for RNNs.\n'
qrnn_configs = op_bench.config_list(attrs=[[1, 3, 1], [5, 7, 4]], attr_names=['I', 'H', 'NL'], cross_product_configs={'B': (True,), 'D': (False, True), 'dtype': (torch.qint8,)}, tags=['short'])

class LSTMBenchmark(op_bench.TorchBenchmarkBase):

    def init(self, I, H, NL, B, D, dtype):
        if False:
            for i in range(10):
                print('nop')
        sequence_len = 128
        batch_size = 16
        scale = 1.0 / 256
        zero_point = 0
        cell_nn = nn.LSTM(input_size=I, hidden_size=H, num_layers=NL, bias=B, batch_first=False, dropout=0.0, bidirectional=D)
        cell_temp = nn.Sequential(cell_nn)
        self.cell = torch.ao.quantization.quantize_dynamic(cell_temp, {nn.LSTM, nn.Linear}, dtype=dtype)[0]
        x = torch.randn(sequence_len, batch_size, I)
        h = torch.randn(NL * (D + 1), batch_size, H)
        c = torch.randn(NL * (D + 1), batch_size, H)
        self.inputs = {'x': x, 'h': h, 'c': c}
        self.set_module_name('QLSTM')

    def forward(self, x, h, c):
        if False:
            i = 10
            return i + 15
        return self.cell(x, (h, c))[0]
op_bench.generate_pt_test(qrnn_configs, LSTMBenchmark)
if __name__ == '__main__':
    op_bench.benchmark_runner.main()