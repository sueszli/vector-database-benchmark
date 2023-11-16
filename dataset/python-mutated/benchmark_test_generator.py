from benchmark_core import _register_test
from benchmark_pytorch import create_pytorch_op_test_case

def generate_pt_test(configs, pt_bench_op):
    if False:
        while True:
            i = 10
    'This function creates PyTorch op test based on the given operator'
    _register_test(configs, pt_bench_op, create_pytorch_op_test_case, False)

def generate_pt_gradient_test(configs, pt_bench_op):
    if False:
        print('Hello World!')
    'This function creates PyTorch op test based on the given operator'
    _register_test(configs, pt_bench_op, create_pytorch_op_test_case, True)

def generate_pt_tests_from_op_list(ops_list, configs, pt_bench_op):
    if False:
        while True:
            i = 10
    'This function creates pt op tests one by one from a list of dictionaries.\n    ops_list is a list of dictionary. Each dictionary includes\n    the name of the operator and the math operation. Here is an example of using this API:\n    unary_ops_configs = op_bench.config_list(\n        attrs=[...],\n        attr_names=["M", "N"],\n    )\n    unary_ops_list = op_bench.op_list(\n        attr_names=["op_name", "op_func"],\n        attrs=[\n            ["abs", torch.abs],\n        ],\n    )\n    class UnaryOpBenchmark(op_bench.TorchBenchmarkBase):\n        def init(self, M, N, op_name, op_func):\n            ...\n        def forward(self):\n            ...\n    op_bench.generate_pt_tests_from_op_list(unary_ops_list, unary_ops_configs, UnaryOpBenchmark)\n    '
    for op in ops_list:
        _register_test(configs, pt_bench_op, create_pytorch_op_test_case, False, op)

def generate_pt_gradient_tests_from_op_list(ops_list, configs, pt_bench_op):
    if False:
        while True:
            i = 10
    for op in ops_list:
        _register_test(configs, pt_bench_op, create_pytorch_op_test_case, True, op)