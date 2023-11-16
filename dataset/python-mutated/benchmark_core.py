import ast
import copy
import functools
import json
import timeit
from collections import namedtuple
import benchmark_utils
import numpy as np
import torch
import torch.utils.cpp_extension as cpp_extension
'Performance microbenchmarks.\n\nThis module contains core functionalities for performance microbenchmark tests.\n'
"\nThis is used to store configs of tests\nAn example input is:\nTestConfig(test_name='add_M8_N2_K1', input_config='M: 8, N: 2, K: 1',\n    tag='long', run_backward=False)\n"
TestConfig = namedtuple('TestConfig', 'test_name input_config tag run_backward')
BENCHMARK_TESTER = []

def _register_test(*test_metainfo):
    if False:
        return 10
    'save the metainfo needed to create a test. Currently test_metainfo\n    takes two different inputs:\n    1) This input when adds single op to the benchmark\n     _register_test(configs, pt_bench_op, create_pytorch_op_test_case,\n                      run_backward=True)\n    2) This input when addes a list of ops to the benchmark\n    _register_test(configs, pt_bench_op, create_pytorch_op_test_case,\n                      run_backward=False,\n                      op_name_function=op)\n    '
    BENCHMARK_TESTER.append(test_metainfo)

def _create_test(bench_op_obj, orig_test_attrs, tags, OperatorTestCase, run_backward, bwd_input):
    if False:
        i = 10
        return i + 15
    'Create tests with the benchmark backend.\n    Args:\n        bench_op_obj: an object which instantiated from a subclass of\n            Caffe2BenchmarkBase/TorchBenchmarkBase which includes tensor\n            creation and operator execution.\n        orig_test_attrs: a dictionary includes test configs.\n        tags: a attribute in test config to filter inputs\n        OperatorTestCase: a named tuple to save the metadata of an test\n        run_backward: a bool parameter indicating backward path\n    '
    test_attrs = copy.deepcopy(orig_test_attrs)
    test_attrs = {k: str(v) for (k, v) in test_attrs.items()}
    ascii_test_attrs = ast.literal_eval(json.dumps(test_attrs))
    input_config = str(ascii_test_attrs)[1:-1].replace("'", '')
    if bwd_input:
        test_attrs.update({'bwd': bwd_input})
    test_name = bench_op_obj.test_name(**test_attrs)
    test_config = TestConfig(test_name, input_config, tags, run_backward)
    return OperatorTestCase(bench_op_obj, test_config)

def _build_test(configs, bench_op, OperatorTestCase, run_backward, op_name_function=None):
    if False:
        while True:
            i = 10
    'Generate PyTorch/Caffe2 tests of operators with different inputs.\n    Args:\n        configs: a dictionary that has the input shapes\n        bench_op: a subclass of Caffe2BenchmarkBase/TorchBenchmarkBase which includes tensor\n            creation and operator execution\n        OperatorTestCase: a named tuple to save the metadata of an test\n        run_backward: a bool parameter indicating backward path\n        op_name_function: a dictionary includes operator name and function\n    '
    for config in configs:
        test_attrs = {}
        tags = None
        keep_config = True
        for attr in config:
            if 'tags' in attr:
                tags = attr['tags']
                continue
            if 'cuda' in attr.values():
                if not torch.cuda.is_available():
                    keep_config = False
                    break
            test_attrs.update(attr)
        if not keep_config:
            continue
        if tags is None:
            raise ValueError('Missing tags in configs')
        input_config = str(test_attrs)[1:-1].replace("'", '')
        op = bench_op()
        assert op is not None, "Can't create test"
        tensor_error_info = None
        init_dict = copy.deepcopy(test_attrs)
        if op_name_function is not None:
            op_name = op_name_function['op_name']
            init_dict.update({'op_func': op_name_function['op_func']})
            op.set_module_name(op_name)
        op._set_backward_test(run_backward)
        op.init(**init_dict)
        op.extract_inputs_tuple()
        if not run_backward:
            for attr in vars(op).values():
                if isinstance(attr, torch.nn.Module):
                    for param in attr.parameters():
                        param.requires_grad = False
        input_name = None
        if op._num_inputs_require_grads > 0:
            input_name = 'all'
        yield _create_test(op, test_attrs, tags, OperatorTestCase, run_backward, input_name)
        for i in range(op._num_inputs_require_grads):
            op._pass_count += 1
            op._auto_set_counter = 0
            new_op = copy.deepcopy(op)
            new_op.init(**init_dict)
            input_name = i + 1
            yield _create_test(new_op, test_attrs, tags, OperatorTestCase, run_backward, input_name)

class BenchmarkRunner:
    """BenchmarkRunner is responsible for benchmarking all the registered
    benchmark test groups.

    Attributes:
        tag_filter (str): control the benchmarks which matches the tag.
        operator (str): only run benchmark test cases that contains
    this filter string in the test case's id.
        test_name (str): only run benchmark test cases that matches this filter,
        this is a case-sensitive substring match and it happens in
        the _keep_test method.
    """

    def __init__(self, args):
        if False:
            while True:
                i = 10
        self.args = args
        self.iters = 100
        self.has_explicit_iteration_count = False
        self.multiplier = 2
        self.predefined_minimum_secs = 1
        self.max_iters = 1000000.0
        self.use_jit = args.use_jit
        self.num_runs = args.num_runs
        self.print_per_iter = False
        self.operator_range = benchmark_utils.get_operator_range(args.operator_range)
        if self.args.warmup_iterations == -1:
            self.args.warmup_iterations = 100
        if self.args.iterations and self.args.iterations != -1:
            self.has_explicit_iteration_count = True
            self.iters = self.args.iterations
        if self.args.test_name is not None:
            self.args.tag_filter = None

    def _print_header(self):
        if False:
            i = 10
            return i + 15
        DASH_LINE = '-' * 40
        print(f'# {DASH_LINE}\n# PyTorch/Caffe2 Operator Micro-benchmarks\n# {DASH_LINE}\n# Tag : {self.args.tag_filter}\n')
        if self.args.list_tests:
            print('# List of tests:')
        elif self.args.list_ops:
            print('# List of Operators to run:')
            self.printed_ops_list = set()
            if self.args.operators:
                print(f'# {self.args.operators}')

    def _print_perf_result(self, reported_run_time_us, test_case):
        if False:
            print('Hello World!')
        if self.args.report_aibench:
            return
            test_name = '_'.join([test_case.framework, test_case.test_config.test_name])
            for run in range(self.num_runs):
                print(f'{test_case.framework}Observer ' + json.dumps({'type': test_name, 'metric': 'latency', 'unit': 'us', 'value': str(reported_run_time_us[run])}))
        else:
            if test_case.framework == 'PyTorch':
                print(f"# Mode: {('JIT' if self.use_jit else 'Eager')}")
            print(f'# Name: {test_case.test_config.test_name}\n# Input: {test_case.test_config.input_config}')
            mode = 'Backward' if test_case.test_config.run_backward else 'Forward'
            if self.num_runs > 1:
                for run in range(self.num_runs):
                    print(f'Run: {run}, {mode} Execution Time (us) : {reported_run_time_us[run]:.3f}')
                print()
            else:
                print(f'{mode} Execution Time (us) : {reported_run_time_us[0]:.3f}\n')

    def _predict_num_iter_needed(self, i):
        if False:
            while True:
                i = 10
        return i * self.multiplier

    def _iteration_result_is_significant(self, iters, run_time_sec, curr_test_total_time, has_explicit_iteration_count):
        if False:
            return 10
        'This function decides whether the measured time can be reported based on the\n        following conditions: 1) the number of iterations is larger than the max_iters.\n        2) the execution time is larger than the predefined minimum_time\n        3) the execution time is larger than user defined minimum_time\n        '
        return (iters > self.max_iters or run_time_sec > self.predefined_minimum_secs or has_explicit_iteration_count) and curr_test_total_time > self.args.min_time_per_test

    def _launch_forward(self, test_case, iters, print_per_iter):
        if False:
            return 10
        "Use Python's timeit module to measure execution time (unit: second)."
        cuda_sync = 'cuda' in test_case.test_config.test_name
        func = test_case.run_forward
        if self.use_jit:
            func = test_case.run_jit_forward
        forward_time = timeit.timeit(functools.partial(func, iters, print_per_iter, cuda_sync), number=1)
        return forward_time

    def _launch_backward(self, test_case, iters, print_per_iter=False):
        if False:
            i = 10
            return i + 15
        'This function runs forward path of an op to get an output. Then the backward path is executed\n        and the execution time is reported\n        '
        test_case.run_forward(num_runs=1, print_per_iter=False, cuda_sync=False)
        if test_case.framework == 'PyTorch':
            test_case._output_mean()
        backward_time = timeit.timeit(functools.partial(test_case.run_backward, iters, print_per_iter), number=1)
        return backward_time

    def _measure_time(self, launch_test, test_case, iters, print_per_iter):
        if False:
            while True:
                i = 10
        "\n        This function execute the operator for <iters> iterations then look at the time.\n        If it's not significant, the number of iterations will be increased before rerun.\n        The execution stops when the time becomes significant.\n        "
        curr_test_total_time = 0
        time_trace = []
        while True:
            run_time_sec = launch_test(test_case, iters, print_per_iter)
            curr_test_total_time += run_time_sec
            results_are_significant = self._iteration_result_is_significant(iters, run_time_sec, curr_test_total_time, self.has_explicit_iteration_count)
            report_run_time = 1000000.0 * run_time_sec / iters
            time_trace.append(report_run_time)
            if self.args.report_aibench:
                mode = 'JIT' if self.use_jit else 'Eager'
                test_name = '_'.join([test_case.framework, test_case.test_config.test_name, mode])
                print('PyTorchObserver ' + json.dumps({'type': test_name, 'metric': 'latency', 'unit': 'ms', 'value': str(report_run_time / 1000.0)}))
            if results_are_significant:
                break
            iters = self._predict_num_iter_needed(iters)
        reported_run_time_us = np.percentile(np.array(time_trace), 50)
        return reported_run_time_us

    def _check_keep(self, test_flag, cmd_flag):
        if False:
            print('Hello World!')
        return cmd_flag is None or test_flag == cmd_flag

    def _check_operator_first_char(self, test_flag, cmd_flag):
        if False:
            return 10
        if cmd_flag is None or test_flag[:1].lower() in cmd_flag:
            return True
        return False

    def _check_keep_list(self, test_flag, cmd_flag_list):
        if False:
            i = 10
            return i + 15
        if cmd_flag_list is None or any((test_flag == cmd_flag for cmd_flag in cmd_flag_list)):
            return True
        return False

    def _keep_test(self, test_case):
        if False:
            i = 10
            return i + 15
        op_test_config = test_case.test_config
        if self.args.framework:
            frameworks = benchmark_utils.process_arg_list(self.args.framework)
        operators = benchmark_utils.process_arg_list(self.args.operators) if self.args.operators else None
        if self._check_keep(op_test_config.test_name, self.args.test_name) and self._check_keep_list(test_case.op_bench.module_name(), operators) and self._check_keep_list(test_case.framework, frameworks) and self._check_operator_first_char(test_case.op_bench.module_name(), self.operator_range) and (self.args.tag_filter == 'all' or self._check_keep(op_test_config.tag, self.args.tag_filter)) and (not self.args.forward_only or op_test_config.run_backward != self.args.forward_only) and (self.args.device == 'None' or 'device' not in test_case.test_config.input_config or self.args.device in op_test_config.test_name):
            return True
        return False

    def _print_test_case_info(self, test_case):
        if False:
            return 10
        if self.args.list_tests:
            print(f'# {test_case.test_config.test_name}')
            return True
        elif self.args.list_ops:
            if self.args.operators is None:
                op_name = test_case.op_bench.module_name()
                if op_name not in self.printed_ops_list:
                    print(f'# {op_name}')
                    self.printed_ops_list.add(op_name)
            return True
        return False

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        self._print_header()
        for test_metainfo in BENCHMARK_TESTER:
            for test in _build_test(*test_metainfo):
                (full_test_id, test_case) = test
                op_test_config = test_case.test_config
                if self._print_test_case_info(test_case):
                    continue
                if not self._keep_test(test_case):
                    continue
                np.random.seed(seed=hash(full_test_id) & (1 << 32) - 1)
                print(f'# Benchmarking {test_case.framework}: {test_case.op_bench.module_name()}')
                if op_test_config.run_backward:
                    launch_func = self._launch_backward
                else:
                    launch_func = self._launch_forward
                launch_func(test_case, self.args.warmup_iterations, print_per_iter=False)
                reported_time = [self._measure_time(launch_func, test_case, self.iters, self.print_per_iter) for _ in range(self.num_runs)]
                self._print_perf_result(reported_time, test_case)