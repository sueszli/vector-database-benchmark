import random
from typing import List
import torch
import operator_benchmark as op_bench
'Microbenchmarks for Stack operator'
stack_configs_static_runtime = op_bench.config_list(attr_names=['sizes', 'N'], attrs=[[(20, 40), 5], [(1, 40), 5]], cross_product_configs={'device': ['cpu', 'cuda'], 'dim': list(range(3))}, tags=['static_runtime'])
stack_configs_short = op_bench.config_list(attr_names=['sizes', 'N'], attrs=[[(1, 1, 1), 2], [(512, 512, 2), 2], [(128, 1024, 2), 2]], cross_product_configs={'device': ['cpu', 'cuda'], 'dim': list(range(4))}, tags=['short'])
stack_configs_long = op_bench.config_list(attr_names=['sizes', 'N'], attrs=[[(2 ** 10, 2 ** 10, 2), 2], [(2 ** 10 + 1, 2 ** 10 - 1, 2), 2], [(2 ** 10, 2 ** 10, 2), 2]], cross_product_configs={'device': ['cpu', 'cuda'], 'dim': list(range(4))}, tags=['long'])
stack_configs_multidim = op_bench.config_list(attr_names=['sizes', 'N'], attrs=[[(2 ** 6, 2 ** 5, 2 ** 2, 2 ** 4, 2 ** 5), 2], [(2 ** 4, 2 ** 5, 2 ** 2, 2 ** 4, 2 ** 5), 8], [(2 ** 3 + 1, 2 ** 5 - 1, 2 ** 2 + 1, 2 ** 4 - 1, 2 ** 5 + 1), 17]], cross_product_configs={'device': ['cpu', 'cuda'], 'dim': list(range(6))}, tags=['multidim'])

class StackBenchmark(op_bench.TorchBenchmarkBase):

    def init(self, sizes, N, dim, device):
        if False:
            while True:
                i = 10
        random.seed(42)
        inputs = []
        gen_sizes = []
        if type(sizes) == list and N == -1:
            gen_sizes = sizes
        else:
            for i in range(N):
                gen_sizes.append([old_size() if callable(old_size) else old_size for old_size in sizes])
        for s in gen_sizes:
            inputs.append(torch.rand(s, device=device))
        result = torch.rand(gen_sizes[0], device=device)
        self.inputs = {'result': result, 'inputs': inputs, 'dim': dim}
        self.set_module_name('stack')

    def forward(self, result: torch.Tensor, inputs: List[torch.Tensor], dim: int):
        if False:
            print('Hello World!')
        return torch.stack(inputs, dim=dim, out=result)
op_bench.generate_pt_test(stack_configs_static_runtime + stack_configs_short + stack_configs_long + stack_configs_multidim, StackBenchmark)
if __name__ == '__main__':
    op_bench.benchmark_runner.main()