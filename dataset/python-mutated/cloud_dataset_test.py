import vaex
import pytest
import fsspec
fs_options = {'anonymous': 'true'}

@pytest.mark.slow
@pytest.mark.parametrize('base_url', ['gs://vaex-data', 's3://vaex'])
@pytest.mark.parametrize('cache', ['true', 'false'])
def test_cloud_dataset_basics(base_url, cache):
    if False:
        print('Hello World!')
    df = vaex.open(f'{base_url}/testing/xys.hdf5?cache={cache}', fs_options=fs_options)
    assert df.x.tolist() == [1, 2]
    assert df.y.tolist() == [3, 4]
    assert df.s.tolist() == ['5', '6']
    assert df.x.count() == 2
    assert df.x.sum() == 3

@pytest.mark.slow
@pytest.mark.parametrize('base_url', ['gs://vaex-data', 's3://vaex'])
@pytest.mark.parametrize('cache', ['true', 'false'])
@pytest.mark.parametrize('file_format', ['hdf5', 'arrow', 'parquet', 'csv', 'feather'])
def test_cloud_dataset_masked(base_url, file_format, cache):
    if False:
        print('Hello World!')
    kwargs = {}
    if file_format == 'csv':
        import pyarrow.csv
        kwargs = {'convert_options': pyarrow.csv.ConvertOptions(column_types={'s': pyarrow.string()}, strings_can_be_null=True)}
    df = vaex.open(f'{base_url}/testing/xys-masked.{file_format}?cache={cache}', fs_options=fs_options, **kwargs)
    assert df.x.tolist() == [1, None]
    assert df.y.tolist() == [None, 4]
    assert df.s.tolist() == ['5', None]
    assert df.x.count() == 1
    assert df.s.count() == 1
    assert df.x.sum() == 1

@pytest.mark.slow
@pytest.mark.parametrize('base_url', ['gs://vaex-data', 's3://vaex'])
def test_cloud_glob(base_url):
    if False:
        while True:
            i = 10
    assert set(vaex.file.glob(f'{base_url}/testing/*.hdf5', fs_options=fs_options)) >= {f'{base_url}/testing/xys-masked.hdf5', f'{base_url}/testing/xys.hdf5'}
    assert set(vaex.file.glob(f'{base_url}/testing/*.hdf5?anonymous=true')) >= {f'{base_url}/testing/xys-masked.hdf5?anonymous=true', f'{base_url}/testing/xys.hdf5?anonymous=true'}

@pytest.mark.parametrize('base_url', ['gs://vaex-data', 's3://vaex'])
def test_cloud_concat(base_url):
    if False:
        while True:
            i = 10
    df1 = vaex.open(f'{base_url}/testing/xys-masked.hdf5?cache=true', fs_options=fs_options)
    df2 = vaex.open(f'{base_url}/testing/xys.hdf5?cache=true', fs_options=fs_options)
    df = df1.concat(df2)
    assert df.x.tolist() == [1, None, 1, 2]

def test_fsspec_arrow():
    if False:
        i = 10
        return i + 15
    fs = fsspec.filesystem('s3', anon=True)
    df = vaex.open('vaex/testing/xys-masked.parquet', fs=fs)
    assert df.x.tolist() == [1, None]