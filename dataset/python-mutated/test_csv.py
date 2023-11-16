from __future__ import annotations
import gzip
import os
import warnings
from io import BytesIO, StringIO
from unittest import mock
import pytest
pd = pytest.importorskip('pandas')
dd = pytest.importorskip('dask.dataframe')
import fsspec
from fsspec.compression import compr
from packaging.version import Version
from tlz import partition_all, valmap
import dask
from dask.base import compute_as_if_collection
from dask.bytes.core import read_bytes
from dask.bytes.utils import compress
from dask.core import flatten
from dask.dataframe._compat import PANDAS_GE_140, PANDAS_GE_200, tm
from dask.dataframe.io.csv import _infer_block_size, auto_blocksize, block_mask, pandas_read_text, text_blocks_to_pandas
from dask.dataframe.optimize import optimize_dataframe_getitem
from dask.dataframe.utils import assert_eq, get_string_dtype, has_known_categories, pyarrow_strings_enabled
from dask.layers import DataFrameIOLayer
from dask.utils import filetext, filetexts, tmpdir, tmpfile
from dask.utils_test import hlg_layer
compression_fmts = [fmt for fmt in compr] + [None]

def normalize_text(s):
    if False:
        i = 10
        return i + 15
    return '\n'.join(map(str.strip, s.strip().split('\n')))

def parse_filename(path):
    if False:
        for i in range(10):
            print('nop')
    return os.path.split(path)[1]
csv_text = '\nname,amount\nAlice,100\nBob,-200\nCharlie,300\nDennis,400\nEdith,-500\nFrank,600\nAlice,200\nFrank,-200\nBob,600\nAlice,400\nFrank,200\nAlice,300\nEdith,600\n'.strip()
tsv_text = csv_text.replace(',', '\t')
tsv_text2 = '\nname   amount\nAlice    100\nBob     -200\nCharlie  300\nDennis   400\nEdith   -500\nFrank    600\nAlice    200\nFrank   -200\nBob      600\nAlice    400\nFrank    200\nAlice    300\nEdith    600\n'.strip()
timeseries = '\nDate,Open,High,Low,Close,Volume,Adj Close\n2015-08-28,198.50,199.839996,197.919998,199.240005,143298900,199.240005\n2015-08-27,197.020004,199.419998,195.210007,199.160004,266244700,199.160004\n2015-08-26,192.080002,194.789993,188.369995,194.679993,328058100,194.679993\n2015-08-25,195.429993,195.449997,186.919998,187.229996,353966700,187.229996\n2015-08-24,197.630005,197.630005,182.399994,189.550003,478672400,189.550003\n2015-08-21,201.729996,203.940002,197.520004,197.630005,328271500,197.630005\n2015-08-20,206.509995,208.289993,203.899994,204.009995,185865600,204.009995\n2015-08-19,209.089996,210.009995,207.350006,208.279999,167316300,208.279999\n2015-08-18,210.259995,210.679993,209.699997,209.929993,70043800,209.929993\n'.strip()
csv_files = {'2014-01-01.csv': b'name,amount,id\nAlice,100,1\nBob,200,2\nCharlie,300,3\n', '2014-01-02.csv': b'name,amount,id\n', '2014-01-03.csv': b'name,amount,id\nDennis,400,4\nEdith,500,5\nFrank,600,6\n'}
tsv_files = {k: v.replace(b',', b'\t') for (k, v) in csv_files.items()}
fwf_files = {'2014-01-01.csv': b'    name  amount  id\n   Alice     100   1\n     Bob     200   2\n Charlie     300   3\n', '2014-01-02.csv': b'    name  amount  id\n', '2014-01-03.csv': b'    name  amount  id\n  Dennis     400   4\n   Edith     500   5\n   Frank     600   6\n'}

def read_files(file_names=csv_files):
    if False:
        for i in range(10):
            print('nop')
    df = pd.concat([pd.read_csv(BytesIO(csv_files[k])) for k in sorted(file_names)])
    df = df.astype({'name': get_string_dtype(), 'amount': int, 'id': int})
    return df

def read_files_with(file_names, handler, **kwargs):
    if False:
        print('Hello World!')
    df = pd.concat([handler(n, **kwargs) for n in sorted(file_names)])
    df = df.astype({'name': get_string_dtype(), 'amount': int, 'id': int})
    return df
comment_header = b'# some header lines\n# that may be present\n# in a data file\n# before any data'
comment_footer = b'# some footer lines\n# that may be present\n# at the end of the file'
csv_units_row = b'str, int, int\n'
tsv_units_row = csv_units_row.replace(b',', b'\t')
csv_and_table = pytest.mark.parametrize('reader,files', [(pd.read_csv, csv_files), (pd.read_table, tsv_files), (pd.read_fwf, fwf_files)])

@csv_and_table
def test_pandas_read_text(reader, files):
    if False:
        return 10
    b = files['2014-01-01.csv']
    df = pandas_read_text(reader, b, b'', {})
    assert list(df.columns) == ['name', 'amount', 'id']
    assert len(df) == 3
    assert df.id.sum() == 1 + 2 + 3

@csv_and_table
def test_pandas_read_text_kwargs(reader, files):
    if False:
        print('Hello World!')
    b = files['2014-01-01.csv']
    df = pandas_read_text(reader, b, b'', {'usecols': ['name', 'id']})
    assert list(df.columns) == ['name', 'id']

@csv_and_table
def test_pandas_read_text_dtype_coercion(reader, files):
    if False:
        for i in range(10):
            print('nop')
    b = files['2014-01-01.csv']
    df = pandas_read_text(reader, b, b'', {}, {'amount': 'float'})
    assert df.amount.dtype == 'float'

@csv_and_table
def test_pandas_read_text_with_header(reader, files):
    if False:
        i = 10
        return i + 15
    b = files['2014-01-01.csv']
    (header, b) = b.split(b'\n', 1)
    header = header + b'\n'
    df = pandas_read_text(reader, b, header, {})
    assert list(df.columns) == ['name', 'amount', 'id']
    assert len(df) == 3
    assert df.id.sum() == 1 + 2 + 3

@csv_and_table
def test_text_blocks_to_pandas_simple(reader, files):
    if False:
        return 10
    blocks = [[files[k]] for k in sorted(files)]
    kwargs = {}
    head = pandas_read_text(reader, files['2014-01-01.csv'], b'', {})
    header = files['2014-01-01.csv'].split(b'\n')[0] + b'\n'
    df = text_blocks_to_pandas(reader, blocks, header, head, kwargs)
    assert isinstance(df, dd.DataFrame)
    assert list(df.columns) == ['name', 'amount', 'id']
    values = text_blocks_to_pandas(reader, blocks, header, head, kwargs)
    assert isinstance(values, dd.DataFrame)
    assert hasattr(values, 'dask')
    assert len(values.dask) == 6 if pyarrow_strings_enabled() else 3
    assert_eq(df.amount.sum(), 100 + 200 + 300 + 400 + 500 + 600)

@csv_and_table
def test_text_blocks_to_pandas_kwargs(reader, files):
    if False:
        for i in range(10):
            print('nop')
    blocks = [files[k] for k in sorted(files)]
    blocks = [[b] for b in blocks]
    kwargs = {'usecols': ['name', 'id']}
    head = pandas_read_text(reader, files['2014-01-01.csv'], b'', kwargs)
    header = files['2014-01-01.csv'].split(b'\n')[0] + b'\n'
    df = text_blocks_to_pandas(reader, blocks, header, head, kwargs)
    assert list(df.columns) == ['name', 'id']
    result = df.compute()
    assert (result.columns == df.columns).all()

@csv_and_table
def test_text_blocks_to_pandas_blocked(reader, files):
    if False:
        return 10
    expected = read_files()
    header = files['2014-01-01.csv'].split(b'\n')[0] + b'\n'
    blocks = []
    for k in sorted(files):
        b = files[k]
        lines = b.split(b'\n')
        blocks.append([b'\n'.join(bs) for bs in partition_all(2, lines)])
    df = text_blocks_to_pandas(reader, blocks, header, expected.head(), {})
    assert_eq(df.compute().reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False)
    expected2 = expected[['name', 'id']]
    df = text_blocks_to_pandas(reader, blocks, header, expected2.head(), {'usecols': ['name', 'id']})
    assert_eq(df.compute().reset_index(drop=True), expected2.reset_index(drop=True), check_dtype=False)

@pytest.mark.parametrize('dd_read,pd_read,files', [(dd.read_csv, pd.read_csv, csv_files), (dd.read_table, pd.read_table, tsv_files)])
def test_skiprows(dd_read, pd_read, files):
    if False:
        while True:
            i = 10
    files = {name: comment_header + b'\n' + content for (name, content) in files.items()}
    skip = len(comment_header.splitlines())
    with filetexts(files, mode='b'):
        df = dd_read('2014-01-*.csv', skiprows=skip)
        expected_df = read_files_with(files, pd_read, skiprows=skip)
        assert_eq(df, expected_df, check_dtype=False)

@pytest.mark.parametrize('dd_read,pd_read,files', [(dd.read_csv, pd.read_csv, csv_files), (dd.read_table, pd.read_table, tsv_files)])
def test_comment(dd_read, pd_read, files):
    if False:
        print('Hello World!')
    files = {name: comment_header + b'\n' + content.replace(b'\n', b'# just some comment\n', 1) for (name, content) in files.items()}
    with filetexts(files, mode='b'):
        df = dd_read('2014-01-*.csv', comment='#')
        expected_df = read_files_with(files, pd_read, comment='#')
        assert_eq(df, expected_df, check_dtype=False)

@pytest.mark.parametrize('dd_read,pd_read,files', [(dd.read_csv, pd.read_csv, csv_files), (dd.read_table, pd.read_table, tsv_files)])
def test_skipfooter(dd_read, pd_read, files):
    if False:
        i = 10
        return i + 15
    files = {name: content + b'\n' + comment_footer for (name, content) in files.items()}
    skip = len(comment_footer.splitlines())
    with filetexts(files, mode='b'):
        df = dd_read('2014-01-*.csv', skipfooter=skip, engine='python')
        expected_df = read_files_with(files, pd_read, skipfooter=skip, engine='python')
        assert_eq(df, expected_df, check_dtype=False)

@pytest.mark.parametrize('dd_read,pd_read,files,units', [(dd.read_csv, pd.read_csv, csv_files, csv_units_row), (dd.read_table, pd.read_table, tsv_files, tsv_units_row)])
def test_skiprows_as_list(dd_read, pd_read, files, units):
    if False:
        while True:
            i = 10
    files = {name: comment_header + b'\n' + content.replace(b'\n', b'\n' + units, 1) for (name, content) in files.items()}
    skip = [0, 1, 2, 3, 5]
    with filetexts(files, mode='b'):
        df = dd_read('2014-01-*.csv', skiprows=skip)
        expected_df = read_files_with(files, pd_read, skiprows=skip)
        assert_eq(df, expected_df, check_dtype=False)
csv_blocks = [[b'aa,bb\n1,1.0\n2,2.0', b'10,20\n30,40'], [b'aa,bb\n1,1.0\n2,2.0', b'10,20\n30,40']]
tsv_blocks = [[b'aa\tbb\n1\t1.0\n2\t2.0', b'10\t20\n30\t40'], [b'aa\tbb\n1\t1.0\n2\t2.0', b'10\t20\n30\t40']]

@pytest.mark.parametrize('reader,blocks', [(pd.read_csv, csv_blocks), (pd.read_table, tsv_blocks)])
def test_enforce_dtypes(reader, blocks):
    if False:
        while True:
            i = 10
    head = reader(BytesIO(blocks[0][0]), header=0)
    header = blocks[0][0].split(b'\n')[0] + b'\n'
    dfs = text_blocks_to_pandas(reader, blocks, header, head, {})
    dfs = dask.compute(dfs, scheduler='sync')
    assert all((df.dtypes.to_dict() == head.dtypes.to_dict() for df in dfs))

@pytest.mark.parametrize('reader,blocks', [(pd.read_csv, csv_blocks), (pd.read_table, tsv_blocks)])
def test_enforce_columns(reader, blocks):
    if False:
        print('Hello World!')
    blocks = [blocks[0], [blocks[1][0].replace(b'a', b'A'), blocks[1][1]]]
    head = reader(BytesIO(blocks[0][0]), header=0)
    header = blocks[0][0].split(b'\n')[0] + b'\n'
    with pytest.raises(ValueError):
        dfs = text_blocks_to_pandas(reader, blocks, header, head, {}, enforce=True)
        dask.compute(*dfs, scheduler='sync')

@pytest.mark.parametrize('dd_read,pd_read,text,sep', [(dd.read_csv, pd.read_csv, csv_text, ','), (dd.read_table, pd.read_table, tsv_text, '\t'), (dd.read_table, pd.read_table, tsv_text2, '\\s+')])
def test_read_csv(dd_read, pd_read, text, sep):
    if False:
        while True:
            i = 10
    with filetext(text) as fn:
        f = dd_read(fn, blocksize=30, lineterminator=os.linesep, sep=sep)
        assert list(f.columns) == ['name', 'amount']
        result = f.compute(scheduler='sync').reset_index(drop=True)
        assert_eq(result, pd_read(fn, sep=sep))

@pytest.mark.skipif(not PANDAS_GE_200, reason='dataframe.convert-string requires pandas>=2.0')
def test_read_csv_convert_string_config():
    if False:
        for i in range(10):
            print('nop')
    pytest.importorskip('pyarrow', reason='Requires pyarrow strings')
    with filetext(csv_text) as fn:
        df = pd.read_csv(fn)
        with dask.config.set({'dataframe.convert-string': True}):
            ddf = dd.read_csv(fn)
        df_pyarrow = df.astype({'name': 'string[pyarrow]'})
        assert_eq(df_pyarrow, ddf, check_index=False)

@pytest.mark.parametrize('dd_read,pd_read,text,skip', [(dd.read_csv, pd.read_csv, csv_text, 7), (dd.read_table, pd.read_table, tsv_text, [1, 13])])
def test_read_csv_large_skiprows(dd_read, pd_read, text, skip):
    if False:
        print('Hello World!')
    names = ['name', 'amount']
    with filetext(text) as fn:
        actual = dd_read(fn, skiprows=skip, names=names)
        assert_eq(actual, pd_read(fn, skiprows=skip, names=names))

@pytest.mark.parametrize('dd_read,pd_read,text,skip', [(dd.read_csv, pd.read_csv, csv_text, 7), (dd.read_table, pd.read_table, tsv_text, [1, 12])])
def test_read_csv_skiprows_only_in_first_partition(dd_read, pd_read, text, skip):
    if False:
        for i in range(10):
            print('nop')
    names = ['name', 'amount']
    with filetext(text) as fn:
        with pytest.warns(UserWarning, match='sample=blocksize'):
            actual = dd_read(fn, blocksize=200, skiprows=skip, names=names).compute()
            assert_eq(actual, pd_read(fn, skiprows=skip, names=names))
        with pytest.warns(UserWarning):
            with pytest.raises(ValueError):
                dd_read(fn, blocksize=30, skiprows=skip, names=names)

@pytest.mark.parametrize('dd_read,pd_read,files', [(dd.read_csv, pd.read_csv, csv_files), (dd.read_table, pd.read_table, tsv_files)])
def test_read_csv_files(dd_read, pd_read, files):
    if False:
        print('Hello World!')
    expected = read_files()
    with filetexts(files, mode='b'):
        df = dd_read('2014-01-*.csv')
        assert_eq(df, expected, check_dtype=False)
        fn = '2014-01-01.csv'
        df = dd_read(fn)
        expected2 = pd_read(BytesIO(files[fn]))
        assert_eq(df, expected2, check_dtype=False)

@pytest.mark.parametrize('dd_read,pd_read,files', [(dd.read_csv, pd.read_csv, csv_files), (dd.read_table, pd.read_table, tsv_files)])
def test_read_csv_files_list(dd_read, pd_read, files):
    if False:
        print('Hello World!')
    with filetexts(files, mode='b'):
        subset = sorted(files)[:2]
        sol = read_files(subset)
        res = dd_read(subset)
        assert_eq(res, sol, check_dtype=False)
        with pytest.raises(ValueError):
            dd_read([])

@pytest.mark.parametrize('dd_read,files', [(dd.read_csv, csv_files), (dd.read_table, tsv_files)])
def test_read_csv_include_path_column(dd_read, files):
    if False:
        i = 10
        return i + 15
    with filetexts(files, mode='b'):
        df = dd_read('2014-01-*.csv', include_path_column=True, converters={'path': parse_filename})
        filenames = df.path.compute().unique()
        assert '2014-01-01.csv' in filenames
        assert '2014-01-02.csv' not in filenames
        assert '2014-01-03.csv' in filenames

@pytest.mark.parametrize('dd_read,files', [(dd.read_csv, csv_files), (dd.read_table, tsv_files)])
def test_read_csv_include_path_column_as_str(dd_read, files):
    if False:
        print('Hello World!')
    with filetexts(files, mode='b'):
        df = dd_read('2014-01-*.csv', include_path_column='filename', converters={'filename': parse_filename})
        filenames = df.filename.compute().unique()
        assert '2014-01-01.csv' in filenames
        assert '2014-01-02.csv' not in filenames
        assert '2014-01-03.csv' in filenames

@pytest.mark.parametrize('dd_read,files', [(dd.read_csv, csv_files), (dd.read_table, tsv_files)])
def test_read_csv_include_path_column_with_duplicate_name(dd_read, files):
    if False:
        return 10
    with filetexts(files, mode='b'):
        with pytest.raises(ValueError):
            dd_read('2014-01-*.csv', include_path_column='name')

@pytest.mark.parametrize('dd_read,files', [(dd.read_csv, csv_files), (dd.read_table, tsv_files)])
def test_read_csv_include_path_column_is_dtype_category(dd_read, files):
    if False:
        print('Hello World!')
    with filetexts(files, mode='b'):
        df = dd_read('2014-01-*.csv', include_path_column=True)
        assert df.path.dtype == 'category'
        assert has_known_categories(df.path)
        dfs = dd_read('2014-01-*.csv', include_path_column=True)
        result = dfs.compute()
        assert result.path.dtype == 'category'
        assert has_known_categories(result.path)

@pytest.mark.parametrize('dd_read,files', [(dd.read_csv, csv_files), (dd.read_table, tsv_files)])
def test_read_csv_include_path_column_with_multiple_partitions_per_file(dd_read, files):
    if False:
        i = 10
        return i + 15
    with filetexts(files, mode='b'):
        df = dd_read('2014-01-*.csv', blocksize='10B', include_path_column=True)
        assert df.npartitions > 3
        assert df.path.dtype == 'category'
        assert has_known_categories(df.path)
        dfs = dd_read('2014-01-*.csv', blocksize='10B', include_path_column=True)
        result = dfs.compute()
        assert result.path.dtype == 'category'
        assert has_known_categories(result.path)

def test_read_csv_index():
    if False:
        while True:
            i = 10
    with filetext(csv_text) as fn:
        f = dd.read_csv(fn, blocksize=20).set_index('amount')
        result = f.compute(scheduler='sync')
        assert result.index.name == 'amount'
        blocks = compute_as_if_collection(dd.DataFrame, f.dask, f.__dask_keys__(), scheduler='sync')
        for (i, block) in enumerate(blocks):
            if i < len(f.divisions) - 2:
                assert (block.index < f.divisions[i + 1]).all()
            if i > 0:
                assert (block.index >= f.divisions[i]).all()
        expected = pd.read_csv(fn).set_index('amount')
        assert_eq(result, expected)

def test_read_csv_skiprows_range():
    if False:
        return 10
    with filetext(csv_text) as fn:
        f = dd.read_csv(fn, skiprows=range(5))
        result = f
        expected = pd.read_csv(fn, skiprows=range(5))
        assert_eq(result, expected)

def test_usecols():
    if False:
        i = 10
        return i + 15
    with filetext(timeseries) as fn:
        df = dd.read_csv(fn, blocksize=30, usecols=['High', 'Low'])
        df_select = df[['High']]
        expected = pd.read_csv(fn, usecols=['High', 'Low'])
        expected_select = expected[['High']]
        assert (df.compute().values == expected.values).all()
        assert (df_select.compute().values == expected_select.values).all()

def test_string_blocksize():
    if False:
        print('Hello World!')
    with filetext(timeseries) as fn:
        a = dd.read_csv(fn, blocksize='30B')
        b = dd.read_csv(fn, blocksize='30')
        assert a.npartitions == b.npartitions
        c = dd.read_csv(fn, blocksize='64MiB')
        assert c.npartitions == 1

def test_skipinitialspace():
    if False:
        return 10
    text = normalize_text('\n    name, amount\n    Alice,100\n    Bob,-200\n    Charlie,300\n    Dennis,400\n    Edith,-500\n    Frank,600\n    ')
    with filetext(text) as fn:
        df = dd.read_csv(fn, skipinitialspace=True, blocksize=20)
        assert 'amount' in df.columns
        assert df.amount.max().compute() == 600

def test_consistent_dtypes():
    if False:
        print('Hello World!')
    text = normalize_text('\n    name,amount\n    Alice,100.5\n    Bob,-200.5\n    Charlie,300\n    Dennis,400\n    Edith,-500\n    Frank,600\n    ')
    with filetext(text) as fn:
        df = dd.read_csv(fn, blocksize=30)
        assert df.amount.compute().dtype == float

def test_consistent_dtypes_2():
    if False:
        while True:
            i = 10
    text1 = normalize_text('\n    name,amount\n    Alice,100\n    Bob,-200\n    Charlie,300\n    ')
    text2 = normalize_text('\n    name,amount\n    1,400\n    2,-500\n    Frank,600\n    ')
    string_dtype = get_string_dtype()
    with filetexts({'foo.1.csv': text1, 'foo.2.csv': text2}):
        df = dd.read_csv('foo.*.csv', blocksize=25)
        assert df.name.dtype == string_dtype
        assert df.name.compute().dtype == string_dtype

def test_categorical_dtypes():
    if False:
        print('Hello World!')
    text1 = normalize_text('\n    fruit,count\n    apple,10\n    apple,25\n    pear,100\n    orange,15\n    ')
    text2 = normalize_text('\n    fruit,count\n    apple,200\n    banana,300\n    orange,400\n    banana,10\n    ')
    with filetexts({'foo.1.csv': text1, 'foo.2.csv': text2}):
        df = dd.read_csv('foo.*.csv', dtype={'fruit': 'category'}, blocksize=25)
        assert df.fruit.dtype == 'category'
        assert not has_known_categories(df.fruit)
        res = df.compute()
        assert res.fruit.dtype == 'category'
        assert sorted(res.fruit.cat.categories) == ['apple', 'banana', 'orange', 'pear']

def test_categorical_known():
    if False:
        while True:
            i = 10
    text1 = normalize_text('\n    A,B\n    a,a\n    b,b\n    a,a\n    ')
    text2 = normalize_text('\n    A,B\n    a,a\n    b,b\n    c,c\n    ')
    dtype = pd.api.types.CategoricalDtype(['a', 'b', 'c'], ordered=False)
    with filetexts({'foo.1.csv': text1, 'foo.2.csv': text2}):
        result = dd.read_csv('foo.*.csv', dtype={'A': 'category', 'B': 'category'})
        assert result.A.cat.known is False
        assert result.B.cat.known is False
        expected = pd.DataFrame({'A': pd.Categorical(['a', 'b', 'a', 'a', 'b', 'c'], categories=dtype.categories), 'B': pd.Categorical(['a', 'b', 'a', 'a', 'b', 'c'], categories=dtype.categories)}, index=[0, 1, 2, 0, 1, 2])
        assert_eq(result, expected)
        result = dd.read_csv('foo.*.csv', dtype={'A': dtype, 'B': 'category'})
        assert result.A.cat.known is True
        assert result.B.cat.known is False
        tm.assert_index_equal(result.A.cat.categories, dtype.categories)
        assert result.A.cat.ordered is False
        assert_eq(result, expected)
        dtype = pd.api.types.CategoricalDtype(['a', 'b', 'c'], ordered=True)
        result = dd.read_csv('foo.*.csv', dtype={'A': dtype, 'B': 'category'})
        expected['A'] = expected['A'].cat.as_ordered()
        assert result.A.cat.known is True
        assert result.B.cat.known is False
        assert result.A.cat.ordered is True
        assert_eq(result, expected)
        result = dd.read_csv('foo.*.csv', dtype=pd.api.types.CategoricalDtype(ordered=False))
        assert result.A.cat.known is False
        result = dd.read_csv('foo.*.csv', dtype='category')
        assert result.A.cat.known is False

@pytest.mark.slow
@pytest.mark.parametrize('compression', ['infer', 'gzip'])
def test_compression_multiple_files(compression):
    if False:
        for i in range(10):
            print('nop')
    with tmpdir() as tdir:
        f = gzip.open(os.path.join(tdir, 'a.csv.gz'), 'wb')
        f.write(csv_text.encode())
        f.close()
        f = gzip.open(os.path.join(tdir, 'b.csv.gz'), 'wb')
        f.write(csv_text.encode())
        f.close()
        with pytest.warns(UserWarning):
            df = dd.read_csv(os.path.join(tdir, '*.csv.gz'), compression=compression)
        assert len(df.compute()) == (len(csv_text.split('\n')) - 1) * 2

def test_empty_csv_file():
    if False:
        while True:
            i = 10
    with filetext('a,b') as fn:
        df = dd.read_csv(fn, header=0)
        assert len(df.compute()) == 0
        assert list(df.columns) == ['a', 'b']

def test_read_csv_no_sample():
    if False:
        while True:
            i = 10
    with filetexts(csv_files, mode='b') as fn:
        df = dd.read_csv(fn, sample=False)
        assert list(df.columns) == ['name', 'amount', 'id']

def test_read_csv_sensitive_to_enforce():
    if False:
        for i in range(10):
            print('nop')
    with filetexts(csv_files, mode='b'):
        a = dd.read_csv('2014-01-*.csv', enforce=True)
        b = dd.read_csv('2014-01-*.csv', enforce=False)
        assert a._name != b._name

@pytest.mark.parametrize('blocksize', [None, 10])
@pytest.mark.parametrize('fmt', compression_fmts)
def test_read_csv_compression(fmt, blocksize):
    if False:
        return 10
    if fmt and fmt not in compress:
        pytest.skip('compress function not provided for %s' % fmt)
    expected = read_files()
    suffix = {'gzip': '.gz', 'bz2': '.bz2', 'zip': '.zip', 'xz': '.xz'}.get(fmt, '')
    files2 = valmap(compress[fmt], csv_files) if fmt else csv_files
    renamed_files = {k + suffix: v for (k, v) in files2.items()}
    with filetexts(renamed_files, mode='b'):
        if fmt and blocksize:
            with pytest.warns(UserWarning):
                df = dd.read_csv('2014-01-*.csv' + suffix, blocksize=blocksize)
        else:
            df = dd.read_csv('2014-01-*.csv' + suffix, blocksize=blocksize)
        assert_eq(df.compute(scheduler='sync').reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False)

@pytest.mark.skip
def test_warn_non_seekable_files():
    if False:
        return 10
    files2 = valmap(compress['gzip'], csv_files)
    with filetexts(files2, mode='b'):
        with pytest.warns(UserWarning) as w:
            df = dd.read_csv('2014-01-*.csv', compression='gzip')
            assert df.npartitions == 3
        assert len(w) == 1
        msg = str(w[0].message)
        assert 'gzip' in msg
        assert 'blocksize=None' in msg
        with warnings.catch_warnings(record=True) as record:
            df = dd.read_csv('2014-01-*.csv', compression='gzip', blocksize=None)
        assert not record
        with pytest.raises(NotImplementedError):
            with pytest.warns(UserWarning):
                df = dd.read_csv('2014-01-*.csv', compression='foo')

def test_windows_line_terminator():
    if False:
        return 10
    text = 'a,b\r\n1,2\r\n2,3\r\n3,4\r\n4,5\r\n5,6\r\n6,7'
    with filetext(text) as fn:
        df = dd.read_csv(fn, blocksize=5, lineterminator='\r\n')
        assert df.b.sum().compute() == 2 + 3 + 4 + 5 + 6 + 7
        assert df.a.sum().compute() == 1 + 2 + 3 + 4 + 5 + 6

@pytest.mark.parametrize('header', [1, 2, 3])
def test_header_int(header):
    if False:
        for i in range(10):
            print('nop')
    text = 'id0,name0,x0,y0\nid,name,x,y\n1034,Victor,-0.25,0.84\n998,Xavier,-0.48,-0.13\n999,Zelda,0.00,0.47\n980,Alice,0.67,-0.98\n989,Zelda,-0.04,0.03\n'
    with filetexts({'test_header_int.csv': text}):
        df = dd.read_csv('test_header_int.csv', header=header, blocksize=64)
        expected = pd.read_csv('test_header_int.csv', header=header)
        assert_eq(df, expected, check_index=False)

def test_header_None():
    if False:
        return 10
    with filetexts({'.tmp.1.csv': '1,2', '.tmp.2.csv': '', '.tmp.3.csv': '3,4'}):
        df = dd.read_csv('.tmp.*.csv', header=None)
        expected = pd.DataFrame({0: [1, 3], 1: [2, 4]})
        assert_eq(df.compute().reset_index(drop=True), expected)

def test_auto_blocksize():
    if False:
        i = 10
        return i + 15
    assert isinstance(auto_blocksize(3000, 15), int)
    assert auto_blocksize(3000, 3) == 100
    assert auto_blocksize(5000, 2) == 250

def test__infer_block_size(monkeypatch):
    if False:
        return 10
    '\n    psutil returns a total memory of `None` on some systems\n    see https://github.com/dask/dask/pull/7601\n    '
    psutil = pytest.importorskip('psutil')

    class MockOutput:
        total = None

    def mock_virtual_memory():
        if False:
            print('Hello World!')
        return MockOutput
    monkeypatch.setattr(psutil, 'virtual_memory', mock_virtual_memory)
    assert _infer_block_size()

def test_auto_blocksize_max64mb():
    if False:
        print('Hello World!')
    blocksize = auto_blocksize(1000000000000, 3)
    assert blocksize == int(64000000.0)
    assert isinstance(blocksize, int)

def test_auto_blocksize_csv(monkeypatch):
    if False:
        i = 10
        return i + 15
    psutil = pytest.importorskip('psutil')
    total_memory = psutil.virtual_memory().total
    cpu_count = psutil.cpu_count()
    mock_read_bytes = mock.Mock(wraps=read_bytes)
    monkeypatch.setattr(dask.dataframe.io.csv, 'read_bytes', mock_read_bytes)
    expected_block_size = auto_blocksize(total_memory, cpu_count)
    with filetexts(csv_files, mode='b'):
        dd.read_csv('2014-01-01.csv')
        assert mock_read_bytes.called
        assert mock_read_bytes.call_args[1]['blocksize'] == expected_block_size

def test_head_partial_line_fix():
    if False:
        i = 10
        return i + 15
    files = {'.overflow1.csv': "a,b\n0,'abcdefghijklmnopqrstuvwxyz'\n1,'abcdefghijklmnopqrstuvwxyz'", '.overflow2.csv': 'a,b\n111111,-11111\n222222,-22222\n333333,-33333\n'}
    with filetexts(files):
        dd.read_csv('.overflow1.csv', sample=52)
        df = dd.read_csv('.overflow2.csv', sample=35)
        assert (df.dtypes == 'i8').all()

def test_read_csv_raises_on_no_files():
    if False:
        while True:
            i = 10
    fn = '.not.a.real.file.csv'
    try:
        dd.read_csv(fn)
        assert False
    except OSError as e:
        assert fn in str(e)

def test_read_csv_has_deterministic_name():
    if False:
        return 10
    with filetext(csv_text) as fn:
        a = dd.read_csv(fn)
        b = dd.read_csv(fn)
        assert a._name == b._name
        assert sorted(a.dask.keys(), key=str) == sorted(b.dask.keys(), key=str)
        assert isinstance(a._name, str)
        c = dd.read_csv(fn, skiprows=1, na_values=[0])
        assert a._name != c._name

def test_multiple_read_csv_has_deterministic_name():
    if False:
        print('Hello World!')
    with filetexts({'_foo.1.csv': csv_text, '_foo.2.csv': csv_text}):
        a = dd.read_csv('_foo.*.csv')
        b = dd.read_csv('_foo.*.csv')
        assert sorted(a.dask.keys(), key=str) == sorted(b.dask.keys(), key=str)

def test_read_csv_has_different_names_based_on_blocksize():
    if False:
        for i in range(10):
            print('nop')
    with filetext(csv_text) as fn:
        a = dd.read_csv(fn, blocksize='10kB')
        b = dd.read_csv(fn, blocksize='20kB')
        assert a._name != b._name

def test_csv_with_integer_names():
    if False:
        while True:
            i = 10
    with filetext('alice,1\nbob,2') as fn:
        df = dd.read_csv(fn, header=None)
        assert list(df.columns) == [0, 1]

def test_late_dtypes():
    if False:
        print('Hello World!')
    text = 'numbers,names,more_numbers,integers,dates\n'
    for _ in range(1000):
        text += '1,,2,3,2017-10-31 00:00:00\n'
    text += '1.5,bar,2.5,3,4998-01-01 00:00:00\n'
    date_msg = "\n\n-------------------------------------------------------------\n\nThe following columns also failed to properly parse as dates:\n\n- dates\n\nThis is usually due to an invalid value in that column. To\ndiagnose and fix it's recommended to drop these columns from the\n`parse_dates` keyword, and manually convert them to dates later\nusing `dd.to_datetime`."
    with filetext(text) as fn:
        sol = pd.read_csv(fn)
        msg = "Mismatched dtypes found in `pd.read_csv`/`pd.read_table`.\n\n+--------------+---------+----------+\n| Column       | Found   | Expected |\n+--------------+---------+----------+\n| more_numbers | float64 | int64    |\n| names        | object  | float64  |\n| numbers      | float64 | int64    |\n+--------------+---------+----------+\n\n- names\n  ValueError(.*)\n\nUsually this is due to dask's dtype inference failing, and\n*may* be fixed by specifying dtypes manually by adding:\n\ndtype={'more_numbers': 'float64',\n       'names': 'object',\n       'numbers': 'float64'}\n\nto the call to `read_csv`/`read_table`."
        with pytest.raises(ValueError) as e:
            dd.read_csv(fn, sample=50, parse_dates=['dates']).compute(scheduler='sync')
        assert e.match(msg + date_msg)
        with pytest.raises(ValueError) as e:
            dd.read_csv(fn, sample=50).compute(scheduler='sync')
        assert e.match(msg)
        msg = "Mismatched dtypes found in `pd.read_csv`/`pd.read_table`.\n\n+--------------+---------+----------+\n| Column       | Found   | Expected |\n+--------------+---------+----------+\n| more_numbers | float64 | int64    |\n| numbers      | float64 | int64    |\n+--------------+---------+----------+\n\nUsually this is due to dask's dtype inference failing, and\n*may* be fixed by specifying dtypes manually by adding:\n\ndtype={'more_numbers': 'float64',\n       'numbers': 'float64'}\n\nto the call to `read_csv`/`read_table`.\n\nAlternatively, provide `assume_missing=True` to interpret\nall unspecified integer columns as floats."
        with pytest.raises(ValueError) as e:
            dd.read_csv(fn, sample=50, dtype={'names': 'O'}).compute(scheduler='sync')
        assert str(e.value) == msg
        with pytest.raises(ValueError) as e:
            dd.read_csv(fn, sample=50, parse_dates=['dates'], dtype={'names': 'O'}).compute(scheduler='sync')
        assert str(e.value) == msg + date_msg
        msg = "Mismatched dtypes found in `pd.read_csv`/`pd.read_table`.\n\nThe following columns failed to properly parse as dates:\n\n- dates\n\nThis is usually due to an invalid value in that column. To\ndiagnose and fix it's recommended to drop these columns from the\n`parse_dates` keyword, and manually convert them to dates later\nusing `dd.to_datetime`."
        with pytest.raises(ValueError) as e:
            dd.read_csv(fn, sample=50, parse_dates=['dates'], dtype={'more_numbers': float, 'names': object, 'numbers': float}).compute(scheduler='sync')
        assert str(e.value) == msg
        res = dd.read_csv(fn, sample=50, dtype={'more_numbers': float, 'names': object, 'numbers': float})
        assert_eq(res, sol)

def test_assume_missing():
    if False:
        while True:
            i = 10
    text = 'numbers,names,more_numbers,integers\n'
    for _ in range(1000):
        text += '1,foo,2,3\n'
    text += '1.5,bar,2.5,3\n'
    with filetext(text) as fn:
        sol = pd.read_csv(fn)
        res = dd.read_csv(fn, sample=50, assume_missing=True)
        assert_eq(res, sol.astype({'integers': float}))
        res = dd.read_csv(fn, sample=50, assume_missing=True, dtype={'integers': 'int64'})
        assert_eq(res, sol)
        res = dd.read_csv(fn, sample=50, assume_missing=True, dtype=None)
        assert_eq(res, sol.astype({'integers': float}))
    text = 'numbers,integers\n'
    for _ in range(1000):
        text += '1,2\n'
    text += '1.5,2\n'
    with filetext(text) as fn:
        sol = pd.read_csv(fn)
        df = dd.read_csv(fn, sample=30, dtype='int64', assume_missing=True)
        assert df.numbers.dtype == 'int64'

def test_index_col():
    if False:
        while True:
            i = 10
    with filetext(csv_text) as fn:
        try:
            dd.read_csv(fn, blocksize=30, index_col='name')
            assert False
        except ValueError as e:
            assert 'set_index' in str(e)
        df = pd.read_csv(fn, index_col=False)
        ddf = dd.read_csv(fn, blocksize=30, index_col=False)
        assert_eq(df, ddf, check_index=False)

def test_read_csv_with_datetime_index_partitions_one():
    if False:
        for i in range(10):
            print('nop')
    with filetext(timeseries) as fn:
        df = pd.read_csv(fn, index_col=0, header=0, usecols=[0, 4], parse_dates=['Date'])
        ddf = dd.read_csv(fn, header=0, usecols=[0, 4], parse_dates=['Date'], blocksize=10000000).set_index('Date')
        assert_eq(df, ddf)
        ddf = dd.read_csv(fn, header=0, usecols=[0, 4], parse_dates=['Date']).set_index('Date')
        assert_eq(df, ddf)

def test_read_csv_with_datetime_index_partitions_n():
    if False:
        i = 10
        return i + 15
    with filetext(timeseries) as fn:
        df = pd.read_csv(fn, index_col=0, header=0, usecols=[0, 4], parse_dates=['Date'])
        ddf = dd.read_csv(fn, header=0, usecols=[0, 4], parse_dates=['Date'], blocksize=400).set_index('Date')
        assert_eq(df, ddf)
xfail_pandas_100 = pytest.mark.xfail(reason='https://github.com/dask/dask/issues/5787')

@pytest.mark.parametrize('encoding', [pytest.param('utf-16', marks=xfail_pandas_100), pytest.param('utf-16-le', marks=xfail_pandas_100), 'utf-16-be'])
def test_encoding_gh601(encoding):
    if False:
        i = 10
        return i + 15
    ar = pd.Series(range(0, 100))
    br = ar % 7
    cr = br * 3.3
    dr = br / 1.9836
    test_df = pd.DataFrame({'a': ar, 'b': br, 'c': cr, 'd': dr})
    with tmpfile('.csv') as fn:
        test_df.to_csv(fn, encoding=encoding, index=False)
        a = pd.read_csv(fn, encoding=encoding)
        d = dd.read_csv(fn, encoding=encoding, blocksize=1000)
        d = d.compute()
        d.index = range(len(d.index))
        assert_eq(d, a)

def test_read_csv_header_issue_823():
    if False:
        for i in range(10):
            print('nop')
    text = 'a b c-d\n1 2 3\n4 5 6'.replace(' ', '\t')
    with filetext(text) as fn:
        df = dd.read_csv(fn, sep='\t')
        assert_eq(df, pd.read_csv(fn, sep='\t'))
        df = dd.read_csv(fn, delimiter='\t')
        assert_eq(df, pd.read_csv(fn, delimiter='\t'))

def test_none_usecols():
    if False:
        i = 10
        return i + 15
    with filetext(csv_text) as fn:
        df = dd.read_csv(fn, usecols=None)
        assert_eq(df, pd.read_csv(fn, usecols=None))

def test_parse_dates_multi_column():
    if False:
        print('Hello World!')
    pdmc_text = normalize_text('\n    ID,date,time\n    10,2003-11-04,180036\n    11,2003-11-05,125640\n    12,2003-11-01,2519\n    13,2003-10-22,142559\n    14,2003-10-24,163113\n    15,2003-10-20,170133\n    16,2003-11-11,160448\n    17,2003-11-03,171759\n    18,2003-11-07,190928\n    19,2003-10-21,84623\n    20,2003-10-25,192207\n    21,2003-11-13,180156\n    22,2003-11-15,131037\n    ')
    with filetext(pdmc_text) as fn:
        ddf = dd.read_csv(fn, parse_dates=[['date', 'time']])
        df = pd.read_csv(fn, parse_dates=[['date', 'time']])
        assert (df.columns == ddf.columns).all()
        assert len(df) == len(ddf)

def test_read_csv_sep():
    if False:
        return 10
    sep_text = normalize_text('\n    name###amount\n    alice###100\n    bob###200\n    charlie###300')
    with filetext(sep_text) as fn:
        ddf = dd.read_csv(fn, sep='###', engine='python')
        df = pd.read_csv(fn, sep='###', engine='python')
        assert (df.columns == ddf.columns).all()
        assert len(df) == len(ddf)

def test_read_csv_slash_r():
    if False:
        return 10
    data = b'0,my\n1,data\n' * 1000 + b'2,foo\rbar'
    with filetext(data, mode='wb') as fn:
        dd.read_csv(fn, header=None, sep=',', lineterminator='\n', names=['a', 'b'], blocksize=200).compute(scheduler='sync')

def test_read_csv_singleton_dtype():
    if False:
        print('Hello World!')
    data = b'a,b\n1,2\n3,4\n5,6'
    with filetext(data, mode='wb') as fn:
        assert_eq(pd.read_csv(fn, dtype=float), dd.read_csv(fn, dtype=float))

@pytest.mark.skipif(not PANDAS_GE_140, reason='arrow engine available from 1.4')
def test_read_csv_arrow_engine():
    if False:
        i = 10
        return i + 15
    pytest.importorskip('pyarrow')
    sep_text = normalize_text('\n    a,b\n    1,2\n    ')
    with filetext(sep_text) as fn:
        assert_eq(pd.read_csv(fn, engine='pyarrow'), dd.read_csv(fn, engine='pyarrow'))

def test_robust_column_mismatch():
    if False:
        i = 10
        return i + 15
    files = csv_files.copy()
    k = sorted(files)[-1]
    files[k] = files[k].replace(b'name', b'Name')
    with filetexts(files, mode='b'):
        ddf = dd.read_csv('2014-01-*.csv', header=None, skiprows=1, names=['name', 'amount', 'id'])
        df = pd.read_csv('2014-01-01.csv')
        assert (df.columns == ddf.columns).all()
        assert_eq(ddf, ddf)

def test_different_columns_are_allowed():
    if False:
        while True:
            i = 10
    files = csv_files.copy()
    k = sorted(files)[-1]
    files[k] = files[k].replace(b'name', b'address')
    with filetexts(files, mode='b'):
        ddf = dd.read_csv('2014-01-*.csv')
        assert (ddf.columns == ['name', 'amount', 'id']).all()
        assert (ddf.compute().columns == ['name', 'amount', 'id', 'address']).all()

def test_error_if_sample_is_too_small():
    if False:
        while True:
            i = 10
    text = 'AAAAA,BBBBB,CCCCC,DDDDD,EEEEE\n1,2,3,4,5\n6,7,8,9,10\n11,12,13,14,15'
    with filetext(text) as fn:
        sample = 20
        with pytest.raises(ValueError):
            dd.read_csv(fn, sample=sample)
        assert_eq(dd.read_csv(fn, sample=sample, header=None), pd.read_csv(fn, header=None))
    skiptext = '# skip\n# these\n# lines\n'
    text = skiptext + text
    with filetext(text) as fn:
        sample = 20 + len(skiptext)
        with pytest.raises(ValueError):
            dd.read_csv(fn, sample=sample, skiprows=3)
        assert_eq(dd.read_csv(fn, sample=sample, header=None, skiprows=3), pd.read_csv(fn, header=None, skiprows=3))

def test_read_csv_names_not_none():
    if False:
        return 10
    text = 'Alice,100\nBob,-200\nCharlie,300\nDennis,400\nEdith,-500\nFrank,600\n'
    names = ['name', 'amount']
    with filetext(text) as fn:
        ddf = dd.read_csv(fn, names=names, blocksize=16)
        df = pd.read_csv(fn, names=names)
        assert_eq(df, ddf, check_index=False)

def test_to_csv():
    if False:
        for i in range(10):
            print('nop')
    df = pd.DataFrame({'x': ['a', 'b', 'c', 'd'], 'y': [1, 2, 3, 4]})
    for npartitions in [1, 2]:
        a = dd.from_pandas(df, npartitions)
        with tmpdir() as dn:
            a.to_csv(dn, index=False)
            result = dd.read_csv(os.path.join(dn, '*')).compute().reset_index(drop=True)
            assert_eq(result, df)
        with tmpdir() as dn:
            r = a.to_csv(dn, index=False, compute=False)
            paths = dask.compute(*r, scheduler='sync')
            assert paths == tuple((os.path.join(dn, f'{n}.part') for n in range(npartitions)))
            result = dd.read_csv(os.path.join(dn, '*')).compute().reset_index(drop=True)
            assert_eq(result, df)
        with tmpdir() as dn:
            fn = os.path.join(dn, 'data_*.csv')
            paths = a.to_csv(fn, index=False)
            assert paths == [os.path.join(dn, f'data_{n}.csv') for n in range(npartitions)]
            result = dd.read_csv(fn).compute().reset_index(drop=True)
            assert_eq(result, df)

def test_to_csv_multiple_files_cornercases():
    if False:
        while True:
            i = 10
    df = pd.DataFrame({'x': ['a', 'b', 'c', 'd'], 'y': [1, 2, 3, 4]})
    a = dd.from_pandas(df, 2)
    with tmpdir() as dn:
        with pytest.raises(ValueError):
            fn = os.path.join(dn, 'data_*_*.csv')
            a.to_csv(fn)
    df16 = pd.DataFrame({'x': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p'], 'y': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]})
    a = dd.from_pandas(df16, 16)
    with tmpdir() as dn:
        fn = os.path.join(dn, 'data_*.csv')
        a.to_csv(fn, index=False)
        result = dd.read_csv(fn).compute().reset_index(drop=True)
        assert_eq(result, df16)
    a = dd.from_pandas(df, 2)
    with tmpdir() as dn:
        a.to_csv(dn, index=False)
        fn = os.path.join(dn, 'data_*.csv')
        a.to_csv(fn, mode='w', index=False)
        result = dd.read_csv(fn).compute().reset_index(drop=True)
        assert_eq(result, df)
    a = dd.from_pandas(df16, 16)
    with tmpdir() as dn:
        a.to_csv(dn, index=False)
        fn = os.path.join(dn, 'data_*.csv')
        a.to_csv(fn, mode='w', index=False)
        result = dd.read_csv(fn).compute().reset_index(drop=True)
        assert_eq(result, df16)

def test_to_single_csv():
    if False:
        for i in range(10):
            print('nop')
    df = pd.DataFrame({'x': ['a', 'b', 'c', 'd'], 'y': [1, 2, 3, 4]})
    for npartitions in [1, 2]:
        a = dd.from_pandas(df, npartitions)
        with tmpdir() as dn:
            fn = os.path.join(dn, 'test.csv')
            a.to_csv(fn, index=False, single_file=True)
            result = dd.read_csv(fn).compute().reset_index(drop=True)
            assert_eq(result, df)
        with tmpdir() as dn:
            fn = os.path.join(dn, 'test.csv')
            r = a.to_csv(fn, index=False, compute=False, single_file=True)
            dask.compute(r, scheduler='sync')
            result = dd.read_csv(fn).compute().reset_index(drop=True)
            assert_eq(result, df)

def test_to_single_csv_with_name_function():
    if False:
        while True:
            i = 10
    df = pd.DataFrame({'x': ['a', 'b', 'c', 'd'], 'y': [1, 2, 3, 4]})
    a = dd.from_pandas(df, 1)
    with tmpdir() as dn:
        fn = os.path.join(dn, 'test.csv')
        with pytest.raises(ValueError, match='name_function is not supported under the single file mode'):
            a.to_csv(fn, name_function=lambda x: x, index=False, single_file=True)

def test_to_single_csv_with_header_first_partition_only():
    if False:
        while True:
            i = 10
    df = pd.DataFrame({'x': ['a', 'b', 'c', 'd'], 'y': [1, 2, 3, 4]})
    a = dd.from_pandas(df, 1)
    with tmpdir() as dn:
        fn = os.path.join(dn, 'test.csv')
        with pytest.raises(ValueError, match='header_first_partition_only cannot be False in the single file mode.'):
            a.to_csv(fn, index=False, header_first_partition_only=False, single_file=True)

def test_to_csv_with_single_file_and_exclusive_mode():
    if False:
        return 10
    df0 = pd.DataFrame({'x': ['a', 'b', 'c', 'd'], 'y': [1, 2, 3, 4]})
    df = dd.from_pandas(df0, npartitions=2)
    with tmpdir() as directory:
        csv_path = os.path.join(directory, 'test.csv')
        df.to_csv(csv_path, index=False, mode='x', single_file=True)
        result = dd.read_csv(os.path.join(directory, '*')).compute()
    assert_eq(result, df0, check_index=False)

def test_to_csv_single_file_exlusive_mode_no_overwrite():
    if False:
        print('Hello World!')
    df0 = pd.DataFrame({'x': ['a', 'b', 'c', 'd'], 'y': [1, 2, 3, 4]})
    df = dd.from_pandas(df0, npartitions=2)
    with tmpdir() as directory:
        csv_path = os.path.join(str(directory), 'test.csv')
        df.to_csv(csv_path, index=False, mode='x', single_file=True)
        assert os.path.exists(csv_path)
        with pytest.raises(FileExistsError):
            df.to_csv(csv_path, index=False, mode='x', single_file=True)
        df.to_csv(csv_path, index=False, mode='w', single_file=True)

def test_to_single_csv_gzip():
    if False:
        print('Hello World!')
    df = pd.DataFrame({'x': ['a', 'b', 'c', 'd'], 'y': [1, 2, 3, 4]})
    for npartitions in [1, 2]:
        a = dd.from_pandas(df, npartitions)
        with tmpdir() as dn:
            fn = os.path.join(dn, 'test.csv.gz')
            a.to_csv(fn, index=False, compression='gzip', single_file=True)
            result = pd.read_csv(fn, compression='gzip').reset_index(drop=True)
            assert_eq(result, df)

@pytest.mark.xfail(reason='to_csv does not support compression')
def test_to_csv_gzip():
    if False:
        return 10
    df = pd.DataFrame({'x': ['a', 'b', 'c', 'd'], 'y': [1, 2, 3, 4]}, index=[1.0, 2.0, 3.0, 4.0])
    for npartitions in [1, 2]:
        a = dd.from_pandas(df, npartitions)
        with tmpfile('csv') as fn:
            a.to_csv(fn, compression='gzip')
            result = pd.read_csv(fn, index_col=0, compression='gzip')
            tm.assert_frame_equal(result, df)

@pytest.mark.skipif(Version(fsspec.__version__) == Version('2023.9.1'), reason='https://github.com/dask/dask/issues/10515')
def test_to_csv_nodir():
    if False:
        i = 10
        return i + 15
    df0 = pd.DataFrame({'x': ['a', 'b', 'c', 'd'], 'y': [1, 2, 3, 4]}, index=[1.0, 2.0, 3.0, 4.0])
    df = dd.from_pandas(df0, npartitions=2)
    with tmpdir() as dir:
        dir0 = os.path.join(str(dir), 'createme')
        df.to_csv(dir0)
        assert 'createme' in os.listdir(dir)
        assert os.listdir(dir0)
        result = dd.read_csv(os.path.join(dir0, '*')).compute()
    assert (result.x.values == df0.x.values).all()

def test_to_csv_simple():
    if False:
        return 10
    df0 = pd.DataFrame({'x': ['a', 'b', 'c', 'd'], 'y': [1, 2, 3, 4]}, index=[1.0, 2.0, 3.0, 4.0])
    df = dd.from_pandas(df0, npartitions=2)
    with tmpdir() as dir:
        dir = str(dir)
        df.to_csv(dir)
        assert os.listdir(dir)
        result = dd.read_csv(os.path.join(dir, '*')).compute()
    assert (result.x.values == df0.x.values).all()

def test_to_csv_with_single_file_and_append_mode():
    if False:
        for i in range(10):
            print('nop')
    df0 = pd.DataFrame({'x': ['a', 'b'], 'y': [1, 2]})
    df1 = pd.DataFrame({'x': ['c', 'd'], 'y': [3, 4]})
    df = dd.from_pandas(df1, npartitions=2)
    with tmpdir() as dir:
        csv_path = os.path.join(dir, 'test.csv')
        df0.to_csv(csv_path, index=False)
        df.to_csv(csv_path, mode='a', header=False, index=False, single_file=True)
        result = dd.read_csv(os.path.join(dir, '*')).compute()
    expected = pd.concat([df0, df1])
    assert assert_eq(result, expected, check_index=False)

def test_to_csv_series():
    if False:
        return 10
    df0 = pd.Series(['a', 'b', 'c', 'd'], index=[1.0, 2.0, 3.0, 4.0])
    df = dd.from_pandas(df0, npartitions=2)
    with tmpdir() as dir:
        dir = str(dir)
        df.to_csv(dir, header=False)
        assert os.listdir(dir)
        result = dd.read_csv(os.path.join(dir, '*'), header=None, names=['x']).compute()
    assert (result.x == df0).all()

def test_to_csv_with_get():
    if False:
        print('Hello World!')
    from dask.multiprocessing import get as mp_get
    flag = [False]

    def my_get(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        flag[0] = True
        return mp_get(*args, **kwargs)
    df = pd.DataFrame({'x': ['a', 'b', 'c', 'd'], 'y': [1, 2, 3, 4]})
    ddf = dd.from_pandas(df, npartitions=2)
    with tmpdir() as dn:
        ddf.to_csv(dn, index=False, compute_kwargs={'scheduler': my_get})
        assert flag[0]
        result = dd.read_csv(os.path.join(dn, '*'))
        assert_eq(result, df, check_index=False)

def test_to_csv_warns_using_scheduler_argument():
    if False:
        i = 10
        return i + 15
    from dask.multiprocessing import get as mp_get
    df = pd.DataFrame({'x': ['a', 'b', 'c', 'd'], 'y': [1, 2, 3, 4]})
    ddf = dd.from_pandas(df, npartitions=2)

    def my_get(*args, **kwargs):
        if False:
            while True:
                i = 10
        return mp_get(*args, **kwargs)
    with tmpdir() as dn:
        with pytest.warns(FutureWarning):
            ddf.to_csv(dn, index=False, scheduler=my_get)

def test_to_csv_errors_using_multiple_scheduler_args():
    if False:
        return 10
    from dask.multiprocessing import get as mp_get
    df = pd.DataFrame({'x': ['a', 'b', 'c', 'd'], 'y': [1, 2, 3, 4]})
    ddf = dd.from_pandas(df, npartitions=2)

    def my_get(*args, **kwargs):
        if False:
            while True:
                i = 10
        return mp_get(*args, **kwargs)
    with tmpdir() as dn:
        with pytest.raises(ValueError) and pytest.warns(FutureWarning):
            ddf.to_csv(dn, index=False, scheduler=my_get, compute_kwargs={'scheduler': my_get})

def test_to_csv_keeps_all_non_scheduler_compute_kwargs():
    if False:
        i = 10
        return i + 15
    from dask.multiprocessing import get as mp_get

    def my_get(*args, **kwargs):
        if False:
            while True:
                i = 10
        assert kwargs['test_kwargs_passed'] == 'foobar'
        return mp_get(*args, **kwargs)
    df = pd.DataFrame({'x': ['a', 'b', 'c', 'd'], 'y': [1, 2, 3, 4]})
    ddf = dd.from_pandas(df, npartitions=2)
    with tmpdir() as dn:
        ddf.to_csv(dn, index=False, compute_kwargs={'scheduler': my_get, 'test_kwargs_passed': 'foobar'})

def test_to_csv_paths():
    if False:
        print('Hello World!')
    df = pd.DataFrame({'A': range(10)})
    ddf = dd.from_pandas(df, npartitions=2)
    paths = ddf.to_csv('foo*.csv')
    assert paths[0].endswith('foo0.csv')
    assert paths[1].endswith('foo1.csv')
    os.remove('foo0.csv')
    os.remove('foo1.csv')

@pytest.mark.parametrize('header, expected', [(False, ''), (True, 'x,y\n')])
def test_to_csv_header_empty_dataframe(header, expected):
    if False:
        print('Hello World!')
    dfe = pd.DataFrame({'x': [], 'y': []})
    ddfe = dd.from_pandas(dfe, npartitions=1)
    with tmpdir() as dn:
        ddfe.to_csv(os.path.join(dn, 'fooe*.csv'), index=False, header=header)
        assert not os.path.exists(os.path.join(dn, 'fooe1.csv'))
        filename = os.path.join(dn, 'fooe0.csv')
        with open(filename) as fp:
            line = fp.readline()
            assert line == expected
        os.remove(filename)

@pytest.mark.parametrize('header,header_first_partition_only,expected_first,expected_next', [(False, False, 'a,1\n', 'd,4\n'), (True, False, 'x,y\n', 'x,y\n'), (False, True, 'a,1\n', 'd,4\n'), (True, True, 'x,y\n', 'd,4\n'), (['aa', 'bb'], False, 'aa,bb\n', 'aa,bb\n'), (['aa', 'bb'], True, 'aa,bb\n', 'd,4\n')])
def test_to_csv_header(header, header_first_partition_only, expected_first, expected_next):
    if False:
        print('Hello World!')
    partition_count = 2
    df = pd.DataFrame({'x': ['a', 'b', 'c', 'd', 'e', 'f'], 'y': [1, 2, 3, 4, 5, 6]})
    ddf = dd.from_pandas(df, npartitions=partition_count)
    with tmpdir() as dn:
        ddf.to_csv(os.path.join(dn, 'fooa*.csv'), index=False, header=header, header_first_partition_only=header_first_partition_only)
        filename = os.path.join(dn, 'fooa0.csv')
        with open(filename) as fp:
            line = fp.readline()
            assert line == expected_first
        os.remove(filename)
        filename = os.path.join(dn, 'fooa1.csv')
        with open(filename) as fp:
            line = fp.readline()
            assert line == expected_next
        os.remove(filename)

def test_to_csv_line_ending():
    if False:
        for i in range(10):
            print('nop')
    df = pd.DataFrame({'x': [0]})
    ddf = dd.from_pandas(df, npartitions=1)
    expected = {b'0\r\n', b'0\n'}
    with tmpdir() as dn:
        ddf.to_csv(os.path.join(dn, 'foo*.csv'), header=False, index=False)
        filename = os.path.join(dn, 'foo0.csv')
        with open(filename, 'rb') as f:
            raw = f.read()
    assert raw in expected

@pytest.mark.parametrize('block_lists', [[[1, 2], [3], [4, 5, 6]], [], [[], [], [1], [], [1]], [list(range(i)) for i in range(10)]])
def test_block_mask(block_lists):
    if False:
        for i in range(10):
            print('nop')
    mask = list(block_mask(block_lists))
    assert len(mask) == len(list(flatten(block_lists)))

def test_reading_empty_csv_files_with_path():
    if False:
        i = 10
        return i + 15
    with tmpdir() as tdir:
        for (k, content) in enumerate(['0, 1, 2', '', '6, 7, 8']):
            with open(os.path.join(tdir, str(k) + '.csv'), 'w') as file:
                file.write(content)
        result = dd.read_csv(os.path.join(tdir, '*.csv'), include_path_column=True, converters={'path': parse_filename}, names=['A', 'B', 'C']).compute()
        df = pd.DataFrame({'A': [0, 6], 'B': [1, 7], 'C': [2, 8], 'path': ['0.csv', '2.csv']})
        df['path'] = df['path'].astype('category')
        assert_eq(result, df, check_index=False)

def test_read_csv_groupby_get_group(tmpdir):
    if False:
        while True:
            i = 10
    path = os.path.join(str(tmpdir), 'test.csv')
    df1 = pd.DataFrame([{'foo': 10, 'bar': 4}])
    df1.to_csv(path, index=False)
    ddf1 = dd.read_csv(path)
    ddfs = ddf1.groupby('foo')
    assert_eq(df1, ddfs.get_group(10).compute())

def test_csv_getitem_column_order(tmpdir):
    if False:
        i = 10
        return i + 15
    path = os.path.join(str(tmpdir), 'test.csv')
    columns = list('abcdefghijklmnopqrstuvwxyz')
    values = list(range(len(columns)))
    df1 = pd.DataFrame([{c: v for (c, v) in zip(columns, values)}])
    df1.to_csv(path)
    columns = list('hczzkylaape')
    df2 = dd.read_csv(path)[columns].head(1)
    assert_eq(df1[columns], df2)

@pytest.mark.skip_with_pyarrow_strings
def test_getitem_optimization_after_filter():
    if False:
        print('Hello World!')
    with filetext(timeseries) as fn:
        expect = pd.read_csv(fn)
        expect = expect[expect['High'] > 205.0][['Low']]
        ddf = dd.read_csv(fn)
        ddf = ddf[ddf['High'] > 205.0][['Low']]
        dsk = optimize_dataframe_getitem(ddf.dask, keys=[ddf._name])
        subgraph_rd = hlg_layer(dsk, 'read-csv')
        assert isinstance(subgraph_rd, DataFrameIOLayer)
        assert set(subgraph_rd.columns) == {'High', 'Low'}
        assert_eq(expect, ddf)

def test_csv_parse_fail(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    path = os.path.join(str(tmpdir), 'test.csv')
    data = b'a,b\n1,"hi\n"\n2,"oi\n"\n'
    expected = pd.read_csv(BytesIO(data))
    with open(path, 'wb') as f:
        f.write(data)
    with pytest.raises(ValueError, match='EOF encountered'):
        dd.read_csv(path, sample=13)
    df = dd.read_csv(path, sample=13, sample_rows=1)
    assert_eq(df, expected)

def test_csv_name_should_be_different_even_if_head_is_same(tmpdir):
    if False:
        return 10
    import random
    from shutil import copyfile
    old_csv_path = os.path.join(str(tmpdir), 'old.csv')
    new_csv_path = os.path.join(str(tmpdir), 'new_csv')
    with open(old_csv_path, 'w') as f:
        for _ in range(10):
            f.write(f'{random.randrange(1, 10 ** 9):09}, {random.randrange(1, 10 ** 9):09}, {random.randrange(1, 10 ** 9):09}\n')
    copyfile(old_csv_path, new_csv_path)
    with open(new_csv_path, 'a') as f:
        for _ in range(3):
            f.write(f'{random.randrange(1, 10 ** 9):09}, {random.randrange(1, 10 ** 9):09}, {random.randrange(1, 10 ** 9):09}\n')
    new_df = dd.read_csv(new_csv_path, header=None, delimiter=',', dtype=str, blocksize=None)
    old_df = dd.read_csv(old_csv_path, header=None, delimiter=',', dtype=str, blocksize=None)
    assert new_df.dask.keys() != old_df.dask.keys()

def test_select_with_include_path_column(tmpdir):
    if False:
        print('Hello World!')
    d = {'col1': [i for i in range(0, 100)], 'col2': [i for i in range(100, 200)]}
    df = pd.DataFrame(data=d)
    temp_path = str(tmpdir) + '/'
    for i in range(6):
        df.to_csv(f'{temp_path}file_{i}.csv', index=False)
    ddf = dd.read_csv(temp_path + '*.csv', include_path_column=True)
    assert_eq(ddf.col1, pd.concat([df.col1] * 6))

@pytest.mark.parametrize('use_names', [True, False])
def test_names_with_header_0(tmpdir, use_names):
    if False:
        while True:
            i = 10
    csv = StringIO('    city1,1992-09-13,10\n    city2,1992-09-13,14\n    city3,1992-09-13,98\n    city4,1992-09-13,13\n    city5,1992-09-13,45\n    city6,1992-09-13,64\n    ')
    if use_names:
        names = ['city', 'date', 'sales']
        usecols = ['city', 'sales']
    else:
        names = usecols = None
    path = os.path.join(str(tmpdir), 'input.csv')
    pd.read_csv(csv, header=None).to_csv(path, index=False, header=False)
    df = pd.read_csv(path, header=0, names=names, usecols=usecols)
    ddf = dd.read_csv(path, header=0, names=names, usecols=usecols, blocksize=60)
    assert_eq(df, ddf, check_index=False)