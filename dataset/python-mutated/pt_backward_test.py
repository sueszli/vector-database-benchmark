import operator_benchmark as op_bench
import torch
add_configs = op_bench.cross_product_configs(M=[8, 1], N=[8, 2], K=[8, 4], tags=['short'])

class AddBenchmark(op_bench.TorchBenchmarkBase):

    def init(self, M, N, K):
        if False:
            for i in range(10):
                print('nop')
        self.input_one = torch.rand(M, N, K, requires_grad=self.auto_set())
        self.input_two = torch.rand(M, N, K, requires_grad=self.auto_set())
        self.set_module_name('add')

    def forward(self):
        if False:
            print('Hello World!')
        return torch.add(self.input_one, self.input_two)
op_bench.generate_pt_test(add_configs, AddBenchmark)
op_bench.generate_pt_gradient_test(add_configs, AddBenchmark)
if __name__ == '__main__':
    op_bench.benchmark_runner.main()