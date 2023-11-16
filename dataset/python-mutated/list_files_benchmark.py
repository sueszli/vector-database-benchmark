"""Benchmarks for `tf.data.Dataset.list_files()`."""
import os
import shutil
import tempfile
from tensorflow.python.data.benchmarks import benchmark_base
from tensorflow.python.data.ops import dataset_ops

class ListFilesBenchmark(benchmark_base.DatasetBenchmarkBase):
    """Benchmarks for `tf.data.Dataset.list_files()`."""

    def benchmark_nested_directories(self):
        if False:
            print('Hello World!')
        tmp_dir = tempfile.mkdtemp()
        width = 1024
        depth = 16
        for i in range(width):
            for j in range(depth):
                new_base = os.path.join(tmp_dir, str(i), *[str(dir_name) for dir_name in range(j)])
                os.makedirs(new_base)
                child_files = ['a.py', 'b.pyc'] if j < depth - 1 else ['c.txt', 'd.log']
                for f in child_files:
                    filename = os.path.join(new_base, f)
                    open(filename, 'w').close()
        patterns = [os.path.join(tmp_dir, os.path.join(*['**' for _ in range(depth)]), suffix) for suffix in ['*.txt', '*.log']]
        num_elements = width * 2
        dataset = dataset_ops.Dataset.list_files(patterns)
        self.run_and_report_benchmark(dataset=dataset, iters=3, num_elements=num_elements, extras={'model_name': 'list_files.benchmark.1', 'parameters': '%d.%d' % (width, depth)}, name='nested_directory(%d*%d)' % (width, depth))
        shutil.rmtree(tmp_dir, ignore_errors=True)
if __name__ == '__main__':
    benchmark_base.test.main()