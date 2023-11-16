import torch
import operator_benchmark as op_bench
'Microbenchmarks for quantized unary operators (point-wise and reduction).'
qunary_ops_configs_short = op_bench.config_list(attr_names=['M', 'N'], attrs=[[512, 512]], cross_product_configs={'dtype': [torch.quint8]}, tags=['short'])
qunary_ops_configs_long = op_bench.cross_product_configs(M=[256, 1024], N=[256, 1024], dtype=[torch.quint8, torch.qint8, torch.qint32], tags=['long'])

class QUnaryOpBenchmark(op_bench.TorchBenchmarkBase):

    def init(self, M, N, dtype, op_func):
        if False:
            return 10
        f_input = torch.rand(M, N)
        scale = 1.0
        zero_point = 0
        self.inputs = {'q_input': torch.quantize_per_tensor(f_input, scale=scale, zero_point=zero_point, dtype=dtype)}
        self.op_func = op_func

    def forward(self, q_input):
        if False:
            print('Hello World!')
        return self.op_func(q_input)
qunary_ops_list = op_bench.op_list(attr_names=['op_name', 'op_func'], attrs=[['q_argsort', torch.argsort], ['q_clone', torch.clone], ['q_mean', torch.mean], ['q_relu', torch.relu], ['q_relu_', torch.relu_], ['q_sort', torch.sort]])
op_bench.generate_pt_tests_from_op_list(qunary_ops_list, qunary_ops_configs_short + qunary_ops_configs_long, QUnaryOpBenchmark)
qunary_ops_topk_configs_short = op_bench.config_list(attr_names=['M', 'N', 'k'], attrs=[[512, 512, 5]], cross_product_configs={'dtype': [torch.quint8]}, tags=['short'])
qunary_ops_topk_configs_long = op_bench.cross_product_configs(M=[256, 1024], N=[256, 1024], k=[1, 3, 5], dtype=[torch.quint8, torch.qint8, torch.qint32], tags=['long'])

class QTopkOpBenchmark(op_bench.TorchBenchmarkBase):

    def init(self, M, N, dtype, k):
        if False:
            print('Hello World!')
        f_input = torch.rand(M, N)
        scale = 1.0
        zero_point = 0
        self.inputs = {'q_input': torch.quantize_per_tensor(f_input, scale=scale, zero_point=zero_point, dtype=dtype), 'k': k}
        self.set_module_name('qtopk')

    def forward(self, q_input, k: int):
        if False:
            i = 10
            return i + 15
        return torch.topk(q_input, k)
op_bench.generate_pt_test(qunary_ops_topk_configs_short + qunary_ops_topk_configs_long, QTopkOpBenchmark)
if __name__ == '__main__':
    op_bench.benchmark_runner.main()