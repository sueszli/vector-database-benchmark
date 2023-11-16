import operator_benchmark as op_bench
import torch
intraop_bench_configs = op_bench.config_list(attrs=[[8, 16]], attr_names=['M', 'N'], tags=['short'])

@torch.jit.script
def torch_sumall(a, iterations):
    if False:
        print('Hello World!')
    result = 0.0
    for _ in range(iterations):
        result += float(torch.sum(a))
        a[0][0] += 0.01
    return result

class TorchSumBenchmark(op_bench.TorchBenchmarkBase):

    def init(self, M, N):
        if False:
            return 10
        self.input_one = torch.rand(M, N)
        self.set_module_name('sum')

    def jit_forward(self, iters):
        if False:
            while True:
                i = 10
        return torch_sumall(self.input_one, iters)
op_bench.generate_pt_test(intraop_bench_configs, TorchSumBenchmark)
if __name__ == '__main__':
    op_bench.benchmark_runner.main()