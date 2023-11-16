from __future__ import annotations
import decimal
import signal
import sys
import threading
import pytest
from dask.datasets import timeseries
dd = pytest.importorskip('dask.dataframe')
pyspark = pytest.importorskip('pyspark')
pa = pytest.importorskip('pyarrow')
pytest.importorskip('fastparquet')
import numpy as np
import pandas as pd
from dask.dataframe._compat import PANDAS_GE_150, PANDAS_GE_200
from dask.dataframe.utils import assert_eq
pytestmark = [pytest.mark.skipif(sys.platform != 'linux', reason='Unnecessary, and hard to get spark working on non-linux platforms'), pytest.mark.skipif(PANDAS_GE_200, reason="pyspark doesn't yet have support for pandas 2.0"), pytest.mark.skip_with_pyarrow_strings]
pdf = timeseries(freq='1h').compute()
pdf.index = pdf.index.tz_localize('UTC')
pdf = pdf.reset_index()

@pytest.fixture(scope='module')
def spark_session():
    if False:
        return 10
    prev = signal.getsignal(signal.SIGINT)
    spark = pyspark.sql.SparkSession.builder.master('local').appName('Dask Testing').config('spark.sql.session.timeZone', 'UTC').getOrCreate()
    yield spark
    spark.stop()
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, prev)

@pytest.mark.parametrize('npartitions', (1, 5, 10))
@pytest.mark.parametrize('engine', ('pyarrow', 'fastparquet'))
def test_roundtrip_parquet_spark_to_dask(spark_session, npartitions, tmpdir, engine):
    if False:
        for i in range(10):
            print('nop')
    tmpdir = str(tmpdir)
    sdf = spark_session.createDataFrame(pdf)
    sdf.repartition(npartitions).write.parquet(tmpdir, mode='overwrite')
    ddf = dd.read_parquet(tmpdir, engine=engine)
    ddf = ddf.assign(timestamp=ddf.timestamp.dt.tz_localize('UTC'))
    assert ddf.npartitions == npartitions
    assert_eq(ddf, pdf, check_index=False)

@pytest.mark.parametrize('engine', ('pyarrow', 'fastparquet'))
def test_roundtrip_hive_parquet_spark_to_dask(spark_session, tmpdir, engine):
    if False:
        while True:
            i = 10
    tmpdir = str(tmpdir)
    sdf = spark_session.createDataFrame(pdf)
    sdf.write.parquet(tmpdir, mode='overwrite', partitionBy='name')
    ddf = dd.read_parquet(tmpdir, engine=engine)
    ddf = ddf.assign(timestamp=ddf.timestamp.dt.tz_localize('UTC'))
    ddf = ddf.compute().sort_index(axis=1)
    ddf = ddf.assign(name=ddf.name.astype('str'))
    assert_eq(ddf, pdf.sort_index(axis=1), check_index=False)

@pytest.mark.parametrize('npartitions', (1, 5, 10))
@pytest.mark.parametrize('engine', ('pyarrow', 'fastparquet'))
def test_roundtrip_parquet_dask_to_spark(spark_session, npartitions, tmpdir, engine):
    if False:
        i = 10
        return i + 15
    tmpdir = str(tmpdir)
    ddf = dd.from_pandas(pdf, npartitions=npartitions)
    kwargs = {'times': 'int96'} if engine == 'fastparquet' else {}
    ddf.to_parquet(tmpdir, engine=engine, write_index=False, **kwargs)
    sdf = spark_session.read.parquet(tmpdir)
    sdf = sdf.toPandas()
    sdf = sdf.assign(timestamp=sdf.timestamp.dt.tz_localize('UTC'))
    assert_eq(sdf, ddf, check_index=False)

def test_roundtrip_parquet_spark_to_dask_extension_dtypes(spark_session, tmpdir):
    if False:
        i = 10
        return i + 15
    tmpdir = str(tmpdir)
    npartitions = 5
    size = 20
    pdf = pd.DataFrame({'a': range(size), 'b': np.random.random(size=size), 'c': [True, False] * (size // 2), 'd': ['alice', 'bob'] * (size // 2)})
    pdf = pdf.astype({'a': 'Int64', 'b': 'Float64', 'c': 'boolean', 'd': 'string'})
    assert all([pd.api.types.is_extension_array_dtype(dtype) for dtype in pdf.dtypes])
    sdf = spark_session.createDataFrame(pdf)
    sdf.repartition(npartitions).write.parquet(tmpdir, mode='overwrite')
    ddf = dd.read_parquet(tmpdir, engine='pyarrow', dtype_backend='numpy_nullable')
    assert all([pd.api.types.is_extension_array_dtype(dtype) for dtype in ddf.dtypes]), ddf.dtypes
    assert_eq(ddf, pdf, check_index=False)

@pytest.mark.skipif(not PANDAS_GE_150, reason='Requires pyarrow-backed nullable dtypes')
def test_read_decimal_dtype_pyarrow(spark_session, tmpdir):
    if False:
        for i in range(10):
            print('nop')
    tmpdir = str(tmpdir)
    npartitions = 3
    size = 6
    decimal_data = [decimal.Decimal('8093.234'), decimal.Decimal('8094.234'), decimal.Decimal('8095.234'), decimal.Decimal('8096.234'), decimal.Decimal('8097.234'), decimal.Decimal('8098.234')]
    pdf = pd.DataFrame({'a': range(size), 'b': decimal_data})
    sdf = spark_session.createDataFrame(pdf)
    sdf = sdf.withColumn('b', sdf['b'].cast(pyspark.sql.types.DecimalType(7, 3)))
    sdf.repartition(npartitions).write.parquet(tmpdir, mode='overwrite')
    ddf = dd.read_parquet(tmpdir, engine='pyarrow', dtype_backend='pyarrow')
    assert ddf.b.dtype.pyarrow_dtype == pa.decimal128(7, 3)
    assert ddf.b.compute().dtype.pyarrow_dtype == pa.decimal128(7, 3)
    expected = pdf.astype({'a': 'int64[pyarrow]', 'b': pd.ArrowDtype(pa.decimal128(7, 3))})
    assert_eq(ddf, expected, check_index=False)