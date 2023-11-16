import torch
import operator_benchmark as op_bench
'\nMicrobenchmarks for the gelu operators.\n'
gelu_configs_long = op_bench.cross_product_configs(N=[1, 4], C=[3], H=[16, 256], W=[16, 256], device=['cpu'], tags=['long'])

class GeluBenchmark(op_bench.TorchBenchmarkBase):

    def init(self, N, C, H, W, device):
        if False:
            print('Hello World!')
        self.inputs = {'input': torch.rand(N, C, H, W, device=device)}

    def forward(self, input):
        if False:
            while True:
                i = 10
        return torch.nn.functional.gelu(input)
op_bench.generate_pt_test(gelu_configs_long, GeluBenchmark)
if __name__ == '__main__':
    op_bench.benchmark_runner.main()