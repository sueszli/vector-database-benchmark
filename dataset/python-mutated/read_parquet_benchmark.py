import ray
from ray.data.dataset import Dataset
from benchmark import Benchmark
from parquet_data_generator import generate_data
import shutil
import tempfile

def read_parquet(root: str, parallelism: int=-1, use_threads: bool=False, filter=None, columns=None) -> Dataset:
    if False:
        while True:
            i = 10
    return ray.data.read_parquet(paths=root, parallelism=parallelism, use_threads=use_threads, filter=filter, columns=columns).materialize()

def run_read_parquet_benchmark(benchmark: Benchmark):
    if False:
        for i in range(10):
            print('nop')
    for parallelism in [1, 2, 4]:
        for use_threads in [True, False]:
            test_name = f'read-parquet-downsampled-nyc-taxi-2009-{parallelism}-{use_threads}'
            benchmark.run_materialize_ds(test_name, read_parquet, root='s3://anonymous@air-example-data/ursa-labs-taxi-data/downsampled_2009_full_year_data.parquet', parallelism=parallelism, use_threads=use_threads)
    data_dirs = []
    total_rows = 1024 * 1024 * 8
    for num_files in [8, 128, 1024]:
        for compression in ['snappy', 'gzip']:
            data_dirs.append(tempfile.mkdtemp())
            generate_data(num_rows=total_rows, num_files=num_files, num_row_groups_per_file=16, compression=compression, data_dir=data_dirs[-1])
            test_name = f'read-parquet-random-data-{num_files}-{compression}'
            benchmark.run_materialize_ds(test_name, read_parquet, root=data_dirs[-1], parallelism=1)
    for dir in data_dirs:
        shutil.rmtree(dir)
    num_files = 1000
    num_row_groups_per_file = 2
    total_rows = num_files * num_row_groups_per_file
    compression = 'gzip'
    many_files_dir = 's3://air-example-data-2/read-many-parquet-files/'
    test_name = f'read-many-parquet-files-s3-{num_files}-{compression}'
    benchmark.run_materialize_ds(test_name, read_parquet, root=many_files_dir)
if __name__ == '__main__':
    ray.init()
    benchmark = Benchmark('read-parquet')
    run_read_parquet_benchmark(benchmark)
    benchmark.write_result()