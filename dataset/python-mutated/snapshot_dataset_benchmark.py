"""Benchmarks for `tf.data.experimental.snapshot()`."""
import os
import shutil
from tensorflow.python.data.benchmarks import benchmark_base
from tensorflow.python.data.experimental.ops import snapshot
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.platform import test

class SnapshotDatasetBenchmark(benchmark_base.DatasetBenchmarkBase):
    """Benchmarks for `tf.data.experimental.snapshot()`."""

    def _makeSnapshotDirectory(self):
        if False:
            for i in range(10):
                print('nop')
        tmp_dir = test.get_temp_dir()
        tmp_dir = os.path.join(tmp_dir, 'snapshot')
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.mkdir(tmp_dir)
        return tmp_dir

    def _createSimpleDataset(self, num_elements, tmp_dir=None, compression=snapshot.COMPRESSION_NONE):
        if False:
            for i in range(10):
                print('nop')
        if not tmp_dir:
            tmp_dir = self._makeSnapshotDirectory()
        dataset = dataset_ops.Dataset.from_tensor_slices([1.0])
        dataset = dataset.map(lambda x: gen_array_ops.broadcast_to(x, [50, 50, 3]))
        dataset = dataset.repeat(num_elements)
        dataset = dataset.apply(snapshot.legacy_snapshot(tmp_dir, compression=compression))
        return dataset

    def benchmarkWriteSnapshotGzipCompression(self):
        if False:
            print('Hello World!')
        num_elements = 500000
        dataset = self._createSimpleDataset(num_elements=num_elements, compression=snapshot.COMPRESSION_GZIP)
        self.run_and_report_benchmark(dataset=dataset, num_elements=num_elements, name='write_gzip', warmup=False, extras={'model_name': 'snapshot.benchmark.1', 'parameters': '%d' % num_elements}, iters=1)

    def benchmarkWriteSnapshotSnappyCompression(self):
        if False:
            print('Hello World!')
        num_elements = 500000
        dataset = self._createSimpleDataset(num_elements=num_elements, compression=snapshot.COMPRESSION_SNAPPY)
        self.run_and_report_benchmark(dataset=dataset, num_elements=num_elements, name='write_snappy', warmup=False, extras={'model_name': 'snapshot.benchmark.2', 'parameters': '%d' % num_elements}, iters=1)

    def benchmarkWriteSnapshotSimple(self):
        if False:
            while True:
                i = 10
        num_elements = 500000
        dataset = self._createSimpleDataset(num_elements=num_elements)
        self.run_and_report_benchmark(dataset=dataset, num_elements=num_elements, name='write_simple', warmup=False, extras={'model_name': 'snapshot.benchmark.3', 'parameters': '%d' % num_elements}, iters=1)

    def benchmarkPassthroughSnapshotSimple(self):
        if False:
            for i in range(10):
                print('nop')
        num_elements = 100000
        tmp_dir = self._makeSnapshotDirectory()
        dataset = self._createSimpleDataset(num_elements=num_elements, tmp_dir=tmp_dir)
        self.run_benchmark(dataset=dataset, num_elements=1, iters=1, warmup=False, apply_default_optimizations=True)
        self.run_and_report_benchmark(dataset=dataset, num_elements=num_elements, name='passthrough_simple', extras={'model_name': 'snapshot.benchmark.4', 'parameters': '%d' % num_elements})

    def benchmarkReadSnapshotSimple(self):
        if False:
            i = 10
            return i + 15
        num_elements = 100000
        tmp_dir = self._makeSnapshotDirectory()
        dataset = self._createSimpleDataset(num_elements=num_elements, tmp_dir=tmp_dir)
        self.run_benchmark(dataset=dataset, num_elements=num_elements, iters=1, warmup=False, apply_default_optimizations=True)
        self.run_and_report_benchmark(dataset=dataset, num_elements=num_elements, name='read_simple', extras={'model_name': 'snapshot.benchmark.5', 'parameters': '%d' % num_elements})

    def benchmarkReadSnapshotGzipCompression(self):
        if False:
            print('Hello World!')
        num_elements = 100000
        tmp_dir = self._makeSnapshotDirectory()
        dataset = self._createSimpleDataset(num_elements=num_elements, tmp_dir=tmp_dir, compression=snapshot.COMPRESSION_GZIP)
        self.run_benchmark(dataset=dataset, num_elements=num_elements, iters=1, warmup=False, apply_default_optimizations=True)
        self.run_and_report_benchmark(dataset=dataset, num_elements=num_elements, name='read_gzip', extras={'model_name': 'snapshot.benchmark.6', 'parameters': '%d' % num_elements})

    def benchmarkReadSnapshotSnappyCompression(self):
        if False:
            i = 10
            return i + 15
        num_elements = 100000
        tmp_dir = self._makeSnapshotDirectory()
        dataset = self._createSimpleDataset(num_elements=num_elements, tmp_dir=tmp_dir, compression=snapshot.COMPRESSION_SNAPPY)
        self.run_benchmark(dataset=dataset, num_elements=num_elements, iters=1, warmup=False, apply_default_optimizations=True)
        self.run_and_report_benchmark(dataset=dataset, num_elements=num_elements, name='read_snappy', extras={'model_name': 'snapshot.benchmark.7', 'parameters': '%d' % num_elements})
if __name__ == '__main__':
    benchmark_base.test.main()