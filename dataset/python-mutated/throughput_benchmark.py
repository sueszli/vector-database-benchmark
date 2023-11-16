import torch._C

def format_time(time_us=None, time_ms=None, time_s=None):
    if False:
        while True:
            i = 10
    'Define time formatting.'
    assert sum([time_us is not None, time_ms is not None, time_s is not None]) == 1
    US_IN_SECOND = 1000000.0
    US_IN_MS = 1000.0
    if time_us is None:
        if time_ms is not None:
            time_us = time_ms * US_IN_MS
        elif time_s is not None:
            time_us = time_s * US_IN_SECOND
        else:
            raise AssertionError("Shouldn't reach here :)")
    if time_us >= US_IN_SECOND:
        return f'{time_us / US_IN_SECOND:.3f}s'
    if time_us >= US_IN_MS:
        return f'{time_us / US_IN_MS:.3f}ms'
    return f'{time_us:.3f}us'

class ExecutionStats:

    def __init__(self, c_stats, benchmark_config):
        if False:
            return 10
        self._c_stats = c_stats
        self.benchmark_config = benchmark_config

    @property
    def latency_avg_ms(self):
        if False:
            return 10
        return self._c_stats.latency_avg_ms

    @property
    def num_iters(self):
        if False:
            while True:
                i = 10
        return self._c_stats.num_iters

    @property
    def iters_per_second(self):
        if False:
            print('Hello World!')
        'Return total number of iterations per second across all calling threads.'
        return self.num_iters / self.total_time_seconds

    @property
    def total_time_seconds(self):
        if False:
            return 10
        return self.num_iters * (self.latency_avg_ms / 1000.0) / self.benchmark_config.num_calling_threads

    def __str__(self):
        if False:
            while True:
                i = 10
        return '\n'.join(['Average latency per example: ' + format_time(time_ms=self.latency_avg_ms), f'Total number of iterations: {self.num_iters}', f'Total number of iterations per second (across all threads): {self.iters_per_second:.2f}', 'Total time: ' + format_time(time_s=self.total_time_seconds)])

class ThroughputBenchmark:
    """
    This class is a wrapper around a c++ component throughput_benchmark::ThroughputBenchmark.

    This wrapper on the throughput_benchmark::ThroughputBenchmark component is responsible
    for executing a PyTorch module (nn.Module or ScriptModule) under an inference
    server like load. It can emulate multiple calling threads to a single module
    provided. In the future we plan to enhance this component to support inter and
    intra-op parallelism as well as multiple models running in a single process.

    Please note that even though nn.Module is supported, it might incur an overhead
    from the need to hold GIL every time we execute Python code or pass around
    inputs as Python objects. As soon as you have a ScriptModule version of your
    model for inference deployment it is better to switch to using it in this
    benchmark.

    Example::

        >>> # xdoctest: +SKIP("undefined vars")
        >>> from torch.utils import ThroughputBenchmark
        >>> bench = ThroughputBenchmark(my_module)
        >>> # Pre-populate benchmark's data set with the inputs
        >>> for input in inputs:
        ...     # Both args and kwargs work, same as any PyTorch Module / ScriptModule
        ...     bench.add_input(input[0], x2=input[1])
        >>> # Inputs supplied above are randomly used during the execution
        >>> stats = bench.benchmark(
        ...     num_calling_threads=4,
        ...     num_warmup_iters = 100,
        ...     num_iters = 1000,
        ... )
        >>> print("Avg latency (ms): {}".format(stats.latency_avg_ms))
        >>> print("Number of iterations: {}".format(stats.num_iters))
    """

    def __init__(self, module):
        if False:
            print('Hello World!')
        if isinstance(module, torch.jit.ScriptModule):
            self._benchmark = torch._C.ThroughputBenchmark(module._c)
        else:
            self._benchmark = torch._C.ThroughputBenchmark(module)

    def run_once(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Given input id (input_idx) run benchmark once and return prediction.\n\n        This is useful for testing that benchmark actually runs the module you\n        want it to run. input_idx here is an index into inputs array populated\n        by calling add_input() method.\n        '
        return self._benchmark.run_once(*args, **kwargs)

    def add_input(self, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Store a single input to a module into the benchmark memory and keep it there.\n\n        During the benchmark execution every thread is going to pick up a\n        random input from the all the inputs ever supplied to the benchmark via\n        this function.\n        '
        self._benchmark.add_input(*args, **kwargs)

    def benchmark(self, num_calling_threads=1, num_warmup_iters=10, num_iters=100, profiler_output_path=''):
        if False:
            print('Hello World!')
        '\n        Run a benchmark on the module.\n\n        Args:\n            num_warmup_iters (int): Warmup iters are used to make sure we run a module\n                a few times before actually measuring things. This way we avoid cold\n                caches and any other similar problems. This is the number of warmup\n                iterations for each of the thread in separate\n\n            num_iters (int): Number of iterations the benchmark should run with.\n                This number is separate from the warmup iterations. Also the number is\n                shared across all the threads. Once the num_iters iterations across all\n                the threads is reached, we will stop execution. Though total number of\n                iterations might be slightly larger. Which is reported as\n                stats.num_iters where stats is the result of this function\n\n            profiler_output_path (str): Location to save Autograd Profiler trace.\n                If not empty, Autograd Profiler will be enabled for the main benchmark\n                execution (but not the warmup phase). The full trace will be saved\n                into the file path provided by this argument\n\n\n        This function returns BenchmarkExecutionStats object which is defined via pybind11.\n        It currently has two fields:\n            - num_iters - number of actual iterations the benchmark have made\n            - avg_latency_ms - average time it took to infer on one input example in milliseconds\n        '
        config = torch._C.BenchmarkConfig()
        config.num_calling_threads = num_calling_threads
        config.num_warmup_iters = num_warmup_iters
        config.num_iters = num_iters
        config.profiler_output_path = profiler_output_path
        c_stats = self._benchmark.benchmark(config)
        return ExecutionStats(c_stats, config)