"""Benchmarks for `tf.data.Dataset.interleave()`."""
from tensorflow.python.data.benchmarks import benchmark_base
from tensorflow.python.data.experimental.ops import interleave_ops
from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.ops import dataset_ops
NON_PARALLEL = 'non_parallel'
EXPERIMENTAL_PARALLEL = 'experimental_parallel'
CORE_PARALLEL = 'core_parallel'

def _make_fake_dataset_fn(initial_delay_us, remainder_delay_us):
    if False:
        i = 10
        return i + 15
    'Returns a dataset that emulates a remote storage data source.\n\n  Returns a dataset factory which creates a dataset with 100 elements that\n  emulates the performance characteristic of a file-based dataset stored in a\n  remote storage. In particular, the first element will take an order of\n  magnitude longer to produce than the remaining elements (100ms vs. 1ms).\n\n  Args:\n    initial_delay_us: How long to wait before producing the first element.\n    remainder_delay_us: How long to wait before producing subsequent elements.\n  '

    def fake_dataset_fn(unused):
        if False:
            while True:
                i = 10
        'Returns a function that creates a dataset with the specified delays.'
        del unused

        def make_dataset(time_us, num_elements):
            if False:
                print('Hello World!')
            dataset = dataset_ops.Dataset.range(num_elements)
            if time_us > 0:
                dataset = dataset.apply(testing.sleep(time_us))
            return dataset
        if not initial_delay_us:
            return make_dataset(remainder_delay_us, 100)
        return make_dataset(initial_delay_us, 0).concatenate(make_dataset(remainder_delay_us, 100))
    return fake_dataset_fn

class ParallelInterleaveBenchmark(benchmark_base.DatasetBenchmarkBase):
    """Benchmarks for `tf.data.experimental.parallel_interleave()`."""

    def apply_interleave(self, interleave_version, dataset, interleave_fn, cycle_length, num_parallel_calls):
        if False:
            while True:
                i = 10
        if interleave_version == NON_PARALLEL:
            return dataset.interleave(interleave_fn, cycle_length=cycle_length)
        elif interleave_version == EXPERIMENTAL_PARALLEL:
            return dataset.apply(interleave_ops.parallel_interleave(interleave_fn, cycle_length=cycle_length))
        elif interleave_version == CORE_PARALLEL:
            if not num_parallel_calls:
                num_parallel_calls = cycle_length
            return dataset.interleave(interleave_fn, cycle_length=cycle_length, num_parallel_calls=num_parallel_calls)
        else:
            raise ValueError('Unknown version: ' + interleave_version)

    def make_dataset(self, interleave_version, initial_delay, remainder_delay, cycle_length, num_parallel_calls=None):
        if False:
            for i in range(10):
                print('nop')
        dataset = dataset_ops.Dataset.range(1).repeat()
        interleave_fn = _make_fake_dataset_fn(initial_delay, remainder_delay)
        return self.apply_interleave(interleave_version=interleave_version, dataset=dataset, interleave_fn=interleave_fn, cycle_length=cycle_length, num_parallel_calls=num_parallel_calls)

    def _benchmark(self, interleave_version, num_elements, benchmark_id, benchmark_label, initial_delay_us=0, remainder_delay_us=0, cycle_length=10, iters=100, num_parallel_calls=None, name=None):
        if False:
            print('Hello World!')
        dataset = self.make_dataset(interleave_version=interleave_version, initial_delay=initial_delay_us, remainder_delay=remainder_delay_us, cycle_length=cycle_length, num_parallel_calls=num_parallel_calls)
        self.run_and_report_benchmark(dataset=dataset, num_elements=num_elements, iters=iters, warmup=True, extras={'model_name': 'interleave.benchmark.%s.%d' % (benchmark_label, benchmark_id), 'parameters': '%d.%d.%d.%s' % (num_elements, cycle_length, iters, str(num_parallel_calls))}, name=name)

    def benchmark_remote_file_simulation(self):
        if False:
            print('Hello World!')
        for (i, version) in enumerate([EXPERIMENTAL_PARALLEL, CORE_PARALLEL]):
            self._benchmark(interleave_version=version, initial_delay_us=100 * 1000, remainder_delay_us=1000, num_elements=5000, name='remote_file_simulation_' + version, benchmark_id=i, benchmark_label='remote_file')

    def benchmark_fast_input(self):
        if False:
            i = 10
            return i + 15
        for (i, version) in enumerate([EXPERIMENTAL_PARALLEL, CORE_PARALLEL]):
            self._benchmark(interleave_version=version, num_elements=200000, name='fast_input_' + version, benchmark_id=i, benchmark_label='fast_input')

    def benchmark_single_cycle(self):
        if False:
            return 10
        for (i, version) in enumerate([NON_PARALLEL, EXPERIMENTAL_PARALLEL, CORE_PARALLEL]):
            self._benchmark(interleave_version=version, cycle_length=1, num_elements=200000, name='single_cycle_' + version, benchmark_id=i, benchmark_label='single_cycle')

    def benchmark_single_parallel_call(self):
        if False:
            return 10
        self._benchmark(interleave_version=CORE_PARALLEL, num_elements=200000, num_parallel_calls=1, name='single_parallel_call_' + CORE_PARALLEL, benchmark_id=1, benchmark_label='single_parallel_call')

    def benchmark_long_cycle(self):
        if False:
            while True:
                i = 10
        for (i, version) in enumerate([EXPERIMENTAL_PARALLEL, CORE_PARALLEL]):
            self._benchmark(interleave_version=version, cycle_length=1000, num_elements=100000, name='long_cycle_' + version, benchmark_id=i, benchmark_label='long_cycle')
if __name__ == '__main__':
    benchmark_base.test.main()