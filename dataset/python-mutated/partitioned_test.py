from pathlib import Path
import shutil
import glob
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset
import vaex
path = Path(__file__).parent.parent
data_path = path / 'data'
countries = ['US', 'US', 'NL', 'FR', 'NL', 'NL']
years = [2020, 2021, 2020, 2020, 2019, 2020]
values = [1, 2, 3, 4, 5, 6]
table = pa.table({'country': countries, 'year': years, 'value': values})

def test_partitioning_basics_hive():
    if False:
        i = 10
        return i + 15
    shutil.rmtree(data_path / 'parquet_dataset_partitioned_hive', ignore_errors=True)
    pq.write_to_dataset(table, data_path / 'parquet_dataset_partitioned_hive', partition_cols=['year', 'country'])
    ds = pa.dataset.dataset(data_path / 'parquet_dataset_partitioned_hive', partitioning='hive')
    df = vaex.open(data_path / 'parquet_dataset_partitioned_hive', partitioning='hive')
    assert set(df.value.tolist()) == set(values)
    assert set(df.year.tolist()) == set(years)
    assert set(df.country.tolist()) == set(countries)

def test_partitioning_write_parquet():
    if False:
        i = 10
        return i + 15
    shutil.rmtree(data_path / 'parquet_dataset_partitioned_vaex', ignore_errors=True)
    df = vaex.from_arrow_table(table)
    df.export_partitioned(data_path / 'parquet_dataset_partitioned_vaex', ['country', 'year'])
    df = vaex.open(data_path / 'parquet_dataset_partitioned_vaex', partitioning='hive')
    assert len(glob.glob(str(data_path / 'parquet_dataset_partitioned_vaex/*/*/*.parquet'))) == 5
    assert len(glob.glob(str(data_path / 'parquet_dataset_partitioned_vaex/country=US/year=2020/*.parquet'))) == 1
    assert len(glob.glob(str(data_path / 'parquet_dataset_partitioned_vaex/country=NL/year=2020/*.parquet'))) == 1
    assert set(df.value.tolist()) == set(values)
    assert set(df.year.tolist()) == set(years)
    assert set(df.country.tolist()) == set(countries)

def test_partitioning_write_hdf5():
    if False:
        return 10
    shutil.rmtree(data_path / 'parquet_dataset_partitioned_vaex', ignore_errors=True)
    df = vaex.from_arrow_table(table)
    df.export_partitioned(data_path / 'parquet_dataset_partitioned_vaex_my_choice/{subdir}/{i}.hdf5', ['country'])
    assert len(glob.glob(str(data_path / 'parquet_dataset_partitioned_vaex_my_choice/*/*.hdf5'))) == 3
    assert len(glob.glob(str(data_path / 'parquet_dataset_partitioned_vaex_my_choice/country=US/[012].hdf5'))) == 1
    assert len(glob.glob(str(data_path / 'parquet_dataset_partitioned_vaex_my_choice/country=NL/[012].hdf5'))) == 1
    assert len(glob.glob(str(data_path / 'parquet_dataset_partitioned_vaex_my_choice/country=FR/[012].hdf5'))) == 1

def test_partitioning_write_directory():
    if False:
        while True:
            i = 10
    shutil.rmtree(data_path / 'parquet_dataset_partitioned_directory1', ignore_errors=True)
    shutil.rmtree(data_path / 'parquet_dataset_partitioned_directory2', ignore_errors=True)
    partitioning = pa.dataset.partitioning(pa.schema([('country', pa.string())]))
    df = vaex.from_arrow_table(table)
    df.export_partitioned(data_path / 'parquet_dataset_partitioned_directory1', ['country'], directory_format='{value}')
    assert len(glob.glob(str(data_path / 'parquet_dataset_partitioned_directory1/*/*.parquet'))) == 3
    assert len(glob.glob(str(data_path / 'parquet_dataset_partitioned_directory1/US/*.parquet'))) == 1
    assert len(glob.glob(str(data_path / 'parquet_dataset_partitioned_directory1/NL/*.parquet'))) == 1
    assert len(glob.glob(str(data_path / 'parquet_dataset_partitioned_directory1/FR/*.parquet'))) == 1
    assert set(df.value.tolist()) == set(values)
    assert set(df.year.tolist()) == set(years)
    assert set(df.country.tolist()) == set(countries)
    partitioning = pa.dataset.partitioning(pa.schema([('year', pa.int64()), ('country', pa.string())]))
    df.export_partitioned(data_path / 'parquet_dataset_partitioned_directory2', ['year', 'country'], directory_format='{value}')
    assert len(glob.glob(str(data_path / 'parquet_dataset_partitioned_directory2/*/*/*.parquet'))) == 5
    assert len(glob.glob(str(data_path / 'parquet_dataset_partitioned_directory2/2020/US/*.parquet'))) == 1
    assert len(glob.glob(str(data_path / 'parquet_dataset_partitioned_directory2/2020/NL/*.parquet'))) == 1
    df = vaex.open(data_path / 'parquet_dataset_partitioned_directory2', partitioning=partitioning)
    assert set(df.value.tolist()) == set(values)
    assert set(df.year.tolist()) == set(years)
    assert set(df.country.tolist()) == set(countries)