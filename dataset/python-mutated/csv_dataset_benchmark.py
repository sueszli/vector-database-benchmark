"""Benchmarks for `tf.data.experimental.CsvDataset`."""
import os
import string
import tempfile
from tensorflow.python.data.benchmarks import benchmark_base
from tensorflow.python.data.experimental.ops import readers
from tensorflow.python.data.ops import readers as core_readers
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import googletest

class CsvDatasetBenchmark(benchmark_base.DatasetBenchmarkBase):
    """Benchmarks for `tf.data.experimental.CsvDataset`."""
    FLOAT_VAL = '1.23456E12'
    STR_VAL = string.ascii_letters * 10

    def _set_up(self, str_val):
        if False:
            print('Hello World!')
        gfile.MakeDirs(googletest.GetTempDir())
        self._temp_dir = tempfile.mkdtemp(dir=googletest.GetTempDir())
        self._num_cols = [4, 64, 256]
        self._num_per_iter = 5000
        self._filenames = []
        for n in self._num_cols:
            fn = os.path.join(self._temp_dir, 'file%d.csv' % n)
            with open(fn, 'w') as f:
                row = ','.join((str_val for _ in range(n)))
                f.write('\n'.join((row for _ in range(100))))
            self._filenames.append(fn)

    def _tear_down(self):
        if False:
            return 10
        gfile.DeleteRecursively(self._temp_dir)

    def _run_benchmark(self, dataset, num_cols, prefix, benchmark_id):
        if False:
            for i in range(10):
                print('nop')
        self.run_and_report_benchmark(dataset=dataset, num_elements=self._num_per_iter, name='%s_with_cols_%d' % (prefix, num_cols), iters=10, extras={'model_name': 'csv.benchmark.%d' % benchmark_id, 'parameters': '%d' % num_cols}, warmup=True)

    def benchmark_map_with_floats(self):
        if False:
            return 10
        self._set_up(self.FLOAT_VAL)
        for i in range(len(self._filenames)):
            num_cols = self._num_cols[i]
            kwargs = {'record_defaults': [[0.0]] * num_cols}
            dataset = core_readers.TextLineDataset(self._filenames[i]).repeat()
            dataset = dataset.map(lambda l: parsing_ops.decode_csv(l, **kwargs))
            self._run_benchmark(dataset=dataset, num_cols=num_cols, prefix='csv_float_map_decode_csv', benchmark_id=1)
        self._tear_down()

    def benchmark_map_with_strings(self):
        if False:
            for i in range(10):
                print('nop')
        self._set_up(self.STR_VAL)
        for i in range(len(self._filenames)):
            num_cols = self._num_cols[i]
            kwargs = {'record_defaults': [['']] * num_cols}
            dataset = core_readers.TextLineDataset(self._filenames[i]).repeat()
            dataset = dataset.map(lambda l: parsing_ops.decode_csv(l, **kwargs))
            self._run_benchmark(dataset=dataset, num_cols=num_cols, prefix='csv_strings_map_decode_csv', benchmark_id=2)
        self._tear_down()

    def benchmark_csv_dataset_with_floats(self):
        if False:
            print('Hello World!')
        self._set_up(self.FLOAT_VAL)
        for i in range(len(self._filenames)):
            num_cols = self._num_cols[i]
            kwargs = {'record_defaults': [[0.0]] * num_cols}
            dataset = core_readers.TextLineDataset(self._filenames[i]).repeat()
            dataset = readers.CsvDataset(self._filenames[i], **kwargs).repeat()
            self._run_benchmark(dataset=dataset, num_cols=num_cols, prefix='csv_float_fused_dataset', benchmark_id=3)
        self._tear_down()

    def benchmark_csv_dataset_with_strings(self):
        if False:
            print('Hello World!')
        self._set_up(self.STR_VAL)
        for i in range(len(self._filenames)):
            num_cols = self._num_cols[i]
            kwargs = {'record_defaults': [['']] * num_cols}
            dataset = core_readers.TextLineDataset(self._filenames[i]).repeat()
            dataset = readers.CsvDataset(self._filenames[i], **kwargs).repeat()
            self._run_benchmark(dataset=dataset, num_cols=num_cols, prefix='csv_strings_fused_dataset', benchmark_id=4)
        self._tear_down()
if __name__ == '__main__':
    benchmark_base.test.main()