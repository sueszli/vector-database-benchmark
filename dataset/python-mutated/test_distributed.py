from __future__ import annotations
import pytest
distributed = pytest.importorskip('distributed')
import asyncio
import os
import sys
from functools import partial
from operator import add
from distributed import Client, SchedulerPlugin, WorkerPlugin
from distributed.utils_test import cleanup
from distributed.utils_test import client as c
from distributed.utils_test import cluster, cluster_fixture, gen_cluster, loop, loop_in_thread, popen, varying
import dask
import dask.bag as db
from dask import compute, delayed, persist
from dask.base import compute_as_if_collection, get_scheduler
from dask.blockwise import Blockwise
from dask.delayed import Delayed
from dask.distributed import futures_of, wait
from dask.layers import ShuffleLayer, SimpleShuffleLayer
from dask.utils import get_named_args, get_scheduler_lock, tmpdir, tmpfile
from dask.utils_test import inc
if 'should_check_state' in get_named_args(gen_cluster):
    gen_cluster = partial(gen_cluster, should_check_state=False)
    cluster = partial(cluster, should_check_state=False)
pytestmark = pytest.mark.skipif(sys.platform == 'win32', reason='The teardown of distributed.utils_test.cluster_fixture fails on windows CI currently')
ignore_sync_scheduler_warning = pytest.mark.filterwarnings('ignore:Running on a single-machine scheduler when a distributed client is active might lead to unexpected results.')

def test_can_import_client():
    if False:
        i = 10
        return i + 15
    from dask.distributed import Client

def test_can_import_nested_things():
    if False:
        while True:
            i = 10
    from dask.distributed.protocol import dumps

@gen_cluster(client=True)
async def test_persist(c, s, a, b):
    x = delayed(inc)(1)
    (x2,) = persist(x)
    await wait(x2)
    assert x2.key in a.data or x2.key in b.data
    y = delayed(inc)(10)
    (y2, one) = persist(y, 1)
    await wait(y2)
    assert y2.key in a.data or y2.key in b.data

def test_persist_nested(c):
    if False:
        i = 10
        return i + 15
    a = delayed(1) + 5
    b = a + 1
    c = a + 2
    result = persist({'a': a, 'b': [1, 2, b]}, (c, 2), 4, [5])
    assert isinstance(result[0]['a'], Delayed)
    assert isinstance(result[0]['b'][2], Delayed)
    assert isinstance(result[1][0], Delayed)
    sol = ({'a': 6, 'b': [1, 2, 7]}, (8, 2), 4, [5])
    assert compute(*result) == sol
    res = persist([a, b], c, 4, [5], traverse=False)
    assert res[0][0] is a
    assert res[0][1] is b
    assert res[1].compute() == 8
    assert res[2:] == (4, [5])

def test_futures_to_delayed_dataframe(c):
    if False:
        print('Hello World!')
    pd = pytest.importorskip('pandas')
    dd = pytest.importorskip('dask.dataframe')
    df = pd.DataFrame({'x': [1, 2, 3]})
    futures = c.scatter([df, df])
    ddf = dd.from_delayed(futures)
    dd.utils.assert_eq(ddf.compute(), pd.concat([df, df], axis=0))
    assert isinstance(ddf.dask.layers[ddf._name], Blockwise)
    with pytest.raises(TypeError):
        ddf = dd.from_delayed([1, 2])

def test_from_delayed_dataframe(c):
    if False:
        for i in range(10):
            print('nop')
    pd = pytest.importorskip('pandas')
    dd = pytest.importorskip('dask.dataframe')
    df = pd.DataFrame({'x': range(20)})
    ddf = dd.from_pandas(df, npartitions=2)
    ddf = dd.from_delayed(ddf.to_delayed())
    dd.utils.assert_eq(ddf, df, scheduler=c)

@pytest.mark.parametrize('fuse', [True, False])
def test_fused_blockwise_dataframe_merge(c, fuse):
    if False:
        return 10
    pd = pytest.importorskip('pandas')
    dd = pytest.importorskip('dask.dataframe')
    size = 35
    df1 = pd.DataFrame({'x': range(size), 'y': range(size)})
    df2 = pd.DataFrame({'x': range(size), 'z': range(size)})
    ddf1 = dd.from_pandas(df1, npartitions=size) + 10
    ddf2 = dd.from_pandas(df2, npartitions=5) + 10
    df1 += 10
    df2 += 10
    with dask.config.set({'optimization.fuse.active': fuse}):
        ddfm = ddf1.merge(ddf2, on=['x'], how='left', shuffle='tasks')
        ddfm.head()
        dfm = ddfm.compute().sort_values('x')
    dd.utils.assert_eq(dfm, df1.merge(df2, on=['x'], how='left').sort_values('x'), check_index=False)

@pytest.mark.parametrize('on', ['a', ['a']])
@pytest.mark.parametrize('broadcast', [True, False])
def test_dataframe_broadcast_merge(c, on, broadcast):
    if False:
        for i in range(10):
            print('nop')
    pd = pytest.importorskip('pandas')
    dd = pytest.importorskip('dask.dataframe')
    pdfl = pd.DataFrame({'a': [1, 2] * 2, 'b_left': range(4)})
    pdfr = pd.DataFrame({'a': [2, 1], 'b_right': range(2)})
    dfl = dd.from_pandas(pdfl, npartitions=4)
    dfr = dd.from_pandas(pdfr, npartitions=2)
    ddfm = dd.merge(dfl, dfr, on=on, broadcast=broadcast, shuffle='tasks')
    dfm = ddfm.compute()
    dd.utils.assert_eq(dfm.sort_values('a'), pd.merge(pdfl, pdfr, on=on).sort_values('a'), check_index=False)

@pytest.mark.parametrize('computation', [None, 'compute_as_if_collection', 'dask.compute'])
@pytest.mark.parametrize('scheduler, use_distributed', [(None, True), ('sync', False)])
def test_default_scheduler_on_worker(c, computation, use_distributed, scheduler):
    if False:
        for i in range(10):
            print('nop')
    'Should a collection use its default scheduler or the distributed\n    scheduler when being computed within a task?\n    '
    pd = pytest.importorskip('pandas')
    dd = pytest.importorskip('dask.dataframe')

    class UpdateGraphCounter(SchedulerPlugin):

        async def start(self, scheduler):
            scheduler._update_graph_count = 0

        def update_graph(self, scheduler, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            scheduler._update_graph_count += 1
    c.register_plugin(UpdateGraphCounter())

    def foo():
        if False:
            i = 10
            return i + 15
        size = 10
        df = pd.DataFrame({'x': range(size), 'y': range(size)})
        ddf = dd.from_pandas(df, npartitions=2)
        if computation is None:
            ddf.compute(scheduler=scheduler)
        elif computation == 'dask.compute':
            dask.compute(ddf, scheduler=scheduler)
        elif computation == 'compute_as_if_collection':
            compute_as_if_collection(ddf.__class__, ddf.dask, list(ddf.dask), scheduler=scheduler)
        else:
            assert False
        return True
    res = c.submit(foo)
    assert res.result() is True
    num_update_graphs = c.run_on_scheduler(lambda dask_scheduler: dask_scheduler._update_graph_count)
    assert num_update_graphs == 2 if use_distributed else 1, num_update_graphs

def test_futures_to_delayed_bag(c):
    if False:
        print('Hello World!')
    L = [1, 2, 3]
    futures = c.scatter([L, L])
    b = db.from_delayed(futures)
    assert list(b) == L + L

def test_futures_to_delayed_array(c):
    if False:
        while True:
            i = 10
    da = pytest.importorskip('dask.array')
    from dask.array.utils import assert_eq
    np = pytest.importorskip('numpy')
    x = np.arange(5)
    futures = c.scatter([x, x])
    A = da.concatenate([da.from_delayed(f, shape=x.shape, dtype=x.dtype) for f in futures], axis=0)
    assert_eq(A.compute(), np.concatenate([x, x], axis=0))

@ignore_sync_scheduler_warning
@gen_cluster(client=True)
async def test_local_get_with_distributed_active(c, s, a, b):
    with dask.config.set(scheduler='sync'):
        x = delayed(inc)(1).persist()
    await asyncio.sleep(0.01)
    assert not s.tasks
    x = delayed(inc)(2).persist(scheduler='sync')
    await asyncio.sleep(0.01)
    assert not s.tasks

@pytest.mark.xfail_with_pyarrow_strings
def test_to_hdf_distributed(c):
    if False:
        print('Hello World!')
    pytest.importorskip('numpy')
    pytest.importorskip('pandas')
    from dask.dataframe.io.tests.test_hdf import test_to_hdf
    test_to_hdf()

@ignore_sync_scheduler_warning
@pytest.mark.parametrize('npartitions', [1, pytest.param(4, marks=pytest.mark.xfail(reason='HDF not multi-process safe', strict=False)), pytest.param(10, marks=pytest.mark.xfail(reason='HDF not multi-process safe', strict=False))])
@pytest.mark.xfail_with_pyarrow_strings
def test_to_hdf_scheduler_distributed(npartitions, c):
    if False:
        return 10
    pytest.importorskip('numpy')
    pytest.importorskip('pandas')
    from dask.dataframe.io.tests.test_hdf import test_to_hdf_schedulers
    test_to_hdf_schedulers(None, npartitions)

@gen_cluster(client=True)
async def test_serializable_groupby_agg(c, s, a, b):
    pd = pytest.importorskip('pandas')
    dd = pytest.importorskip('dask.dataframe')
    df = pd.DataFrame({'x': [1, 2, 3, 4], 'y': [1, 0, 1, 0]})
    ddf = dd.from_pandas(df, npartitions=2)
    result = ddf.groupby('y', sort=False).agg('count', split_out=2)
    agg0 = await c.compute(result.partitions[0])
    agg1 = await c.compute(result.partitions[1])
    dd.utils.assert_eq(pd.concat([agg0, agg1]), pd.DataFrame({'x': [2, 2], 'y': [0, 1]}).set_index('y'))

def test_futures_in_graph(c):
    if False:
        return 10
    (x, y) = (delayed(1), delayed(2))
    xx = delayed(add)(x, x)
    yy = delayed(add)(y, y)
    xxyy = delayed(add)(xx, yy)
    xxyy2 = c.persist(xxyy)
    xxyy3 = delayed(add)(xxyy2, 10)
    assert xxyy3.compute(scheduler='dask.distributed') == 1 + 1 + (2 + 2) + 10

def test_zarr_distributed_roundtrip(c):
    if False:
        return 10
    da = pytest.importorskip('dask.array')
    pytest.importorskip('zarr')
    with tmpdir() as d:
        a = da.zeros((3, 3), chunks=(1, 1))
        a.to_zarr(d)
        a2 = da.from_zarr(d)
        da.assert_eq(a, a2, scheduler=c)
        assert a2.chunks == a.chunks

def test_zarr_in_memory_distributed_err(c):
    if False:
        print('Hello World!')
    da = pytest.importorskip('dask.array')
    zarr = pytest.importorskip('zarr')
    chunks = (1, 1)
    a = da.ones((3, 3), chunks=chunks)
    z = zarr.zeros_like(a, chunks=chunks)
    with pytest.raises(RuntimeError, match='distributed scheduler'):
        a.to_zarr(z)

def test_scheduler_equals_client(c):
    if False:
        i = 10
        return i + 15
    x = delayed(lambda : 1)()
    assert x.compute(scheduler=c) == 1
    assert c.run_on_scheduler(lambda dask_scheduler: dask_scheduler.story(x.key))

@gen_cluster(client=True)
async def test_await(c, s, a, b):
    x = dask.delayed(inc)(1)
    x = await x.persist()
    assert x.key in s.tasks
    assert a.data or b.data
    assert all((f.done() for f in futures_of(x)))

def test_local_scheduler():
    if False:
        i = 10
        return i + 15

    async def f():
        x = dask.delayed(inc)(1)
        y = x + 1
        z = await y.persist()
        assert len(z.dask) == 1
    asyncio.run(f())

@gen_cluster(client=True)
async def test_annotations_blockwise_unpack(c, s, a, b):
    da = pytest.importorskip('dask.array')
    np = pytest.importorskip('numpy')
    from dask.array.utils import assert_eq
    scale = varying([ZeroDivisionError('one'), ZeroDivisionError('two'), 2, 2])

    def flaky_double(x):
        if False:
            print('Hello World!')
        return scale() * x

    def reliable_double(x):
        if False:
            for i in range(10):
                print('nop')
        return 2 * x
    x = da.ones(10, chunks=(5,))
    with dask.annotate(retries=2):
        y = x.map_blocks(flaky_double, meta=np.array((), dtype=np.float64))
    with dask.annotate(retries=0):
        z = y.map_blocks(reliable_double, meta=np.array((), dtype=np.float64))
    with dask.config.set(optimization__fuse__active=False):
        z = await c.compute(z)
    assert_eq(z, np.ones(10) * 4.0)

@pytest.mark.parametrize('io', ['ones', 'zeros', 'full'])
@pytest.mark.parametrize('fuse', [True, False, None])
def test_blockwise_array_creation(c, io, fuse):
    if False:
        return 10
    np = pytest.importorskip('numpy')
    da = pytest.importorskip('dask.array')
    chunks = (5, 2)
    shape = (10, 4)
    if io == 'ones':
        darr = da.ones(shape, chunks=chunks)
        narr = np.ones(shape)
    elif io == 'zeros':
        darr = da.zeros(shape, chunks=chunks)
        narr = np.zeros(shape)
    elif io == 'full':
        darr = da.full(shape, 10, chunks=chunks)
        narr = np.full(shape, 10)
    darr += 2
    narr += 2
    with dask.config.set({'optimization.fuse.active': fuse}):
        darr.compute()
        dsk = dask.array.optimize(darr.dask, darr.__dask_keys__())
        assert isinstance(dsk, dict) == (fuse is not False)
        da.assert_eq(darr, narr, scheduler=c)

@ignore_sync_scheduler_warning
@pytest.mark.parametrize('io', ['parquet-pyarrow', pytest.param('parquet-fastparquet', marks=pytest.mark.skip_with_pyarrow_strings), 'csv', pytest.param('hdf', marks=pytest.mark.flaky(reruns=5))])
@pytest.mark.parametrize('fuse', [True, False, None])
@pytest.mark.parametrize('from_futures', [True, False])
def test_blockwise_dataframe_io(c, tmpdir, io, fuse, from_futures):
    if False:
        print('Hello World!')
    pd = pytest.importorskip('pandas')
    dd = pytest.importorskip('dask.dataframe')
    df = pd.DataFrame({'x': [1, 2, 3] * 5, 'y': range(15)})
    if from_futures:
        parts = [df.iloc[:5], df.iloc[5:10], df.iloc[10:15]]
        futs = c.scatter(parts)
        ddf0 = dd.from_delayed(futs, meta=parts[0])
    else:
        ddf0 = dd.from_pandas(df, npartitions=3)
    if io.startswith('parquet'):
        if io == 'parquet-pyarrow':
            pytest.importorskip('pyarrow.parquet')
            engine = 'pyarrow'
        else:
            pytest.importorskip('fastparquet')
            engine = 'fastparquet'
        ddf0.to_parquet(str(tmpdir), engine=engine)
        ddf = dd.read_parquet(str(tmpdir), engine=engine)
    elif io == 'csv':
        ddf0.to_csv(str(tmpdir), index=False)
        ddf = dd.read_csv(os.path.join(str(tmpdir), '*'))
    elif io == 'hdf':
        pytest.importorskip('tables')
        fn = str(tmpdir.join('h5'))
        ddf0.to_hdf(fn, '/data*')
        ddf = dd.read_hdf(fn, '/data*')
    df = df[['x']] + 10
    ddf = ddf[['x']] + 10
    with dask.config.set({'optimization.fuse.active': fuse}):
        ddf.compute()
        dsk = dask.dataframe.optimize(ddf.dask, ddf.__dask_keys__())
        assert isinstance(dsk, dict) == bool(fuse)
        dd.assert_eq(ddf, df, check_index=False)

def test_blockwise_fusion_after_compute(c):
    if False:
        return 10
    pd = pytest.importorskip('pandas')
    dd = pytest.importorskip('dask.dataframe')
    df = pd.DataFrame({'x': [1, 2, 3] * 5})
    series = dd.from_pandas(df, npartitions=2)['x']
    result = series < 3
    series_len = len(series)
    assert series_len == 15
    assert df.x[result.compute()].sum() == 15

@gen_cluster(client=True)
async def test_blockwise_numpy_args(c, s, a, b):
    """Test pack/unpack of blockwise that includes a NumPy literal argument"""
    da = pytest.importorskip('dask.array')
    np = pytest.importorskip('numpy')

    def fn(x, dt):
        if False:
            while True:
                i = 10
        assert type(dt) is np.uint16
        return x.astype(dt)
    arr = da.blockwise(fn, 'x', da.ones(1000), 'x', np.uint16(42), None, dtype=np.uint16)
    res = await c.compute(arr.sum(), optimize_graph=False)
    assert res == 1000

@gen_cluster(client=True)
async def test_blockwise_numpy_kwargs(c, s, a, b):
    """Test pack/unpack of blockwise that includes a NumPy literal keyword argument"""
    da = pytest.importorskip('dask.array')
    np = pytest.importorskip('numpy')

    def fn(x, dt=None):
        if False:
            print('Hello World!')
        assert type(dt) is np.uint16
        return x.astype(dt)
    arr = da.blockwise(fn, 'x', da.ones(1000), 'x', dtype=np.uint16, dt=np.uint16(42))
    res = await c.compute(arr.sum(), optimize_graph=False)
    assert res == 1000

def test_blockwise_different_optimization(c):
    if False:
        i = 10
        return i + 15
    da = pytest.importorskip('dask.array')
    np = pytest.importorskip('numpy')
    u = da.from_array(np.arange(3))
    v = da.from_array(np.array([10 + 2j, 7 - 3j, 8 + 1j]))
    cv = v.conj()
    x = u * cv
    (cv,) = dask.optimize(cv)
    y = u * cv
    expected = np.array([0 + 0j, 7 + 3j, 16 - 2j])
    with dask.config.set({'optimization.fuse.active': False}):
        x_value = x.compute()
        y_value = y.compute()
    np.testing.assert_equal(x_value, expected)
    np.testing.assert_equal(y_value, expected)

@gen_cluster(client=True)
async def test_combo_of_layer_types(c, s, a, b):
    """Check pack/unpack of a HLG that has every type of Layers!"""
    da = pytest.importorskip('dask.array')
    dd = pytest.importorskip('dask.dataframe')
    np = pytest.importorskip('numpy')
    pd = pytest.importorskip('pandas')

    def add(x, y, z, extra_arg):
        if False:
            return 10
        return x + y + z + extra_arg
    y = c.submit(lambda x: x, 2)
    z = c.submit(lambda x: x, 3)
    x = da.blockwise(add, 'x', da.zeros((3,), chunks=(1,)), 'x', da.ones((3,), chunks=(1,)), 'x', y, None, concatenate=False, dtype=int, extra_arg=z)
    df = dd.from_pandas(pd.DataFrame({'a': np.arange(3)}), npartitions=3)
    df = df.shuffle('a', shuffle='tasks')
    df = df['a'].to_dask_array()
    res = x.sum() + df.sum()
    res = await c.compute(res, optimize_graph=False)
    assert res == 21

def test_blockwise_concatenate(c):
    if False:
        print('Hello World!')
    'Test a blockwise operation with concatenated axes'
    da = pytest.importorskip('dask.array')
    np = pytest.importorskip('numpy')

    def f(x, y):
        if False:
            print('Hello World!')
        da.assert_eq(y, [[0, 1, 2]])
        return x
    x = da.from_array(np.array([0, 1, 2]))
    y = da.from_array(np.array([[0, 1, 2]]))
    z = da.blockwise(f, 'i', x, 'i', y, 'ij', dtype=x.dtype, concatenate=True)
    c.compute(z, optimize_graph=False)
    da.assert_eq(z, x, scheduler=c)

@gen_cluster(client=True)
async def test_map_partitions_partition_info(c, s, a, b):
    dd = pytest.importorskip('dask.dataframe')
    pd = pytest.importorskip('pandas')
    ddf = dd.from_pandas(pd.DataFrame({'a': range(10)}), npartitions=2)
    res = await c.compute(ddf.map_partitions(lambda x, partition_info=None: partition_info))
    assert res[0] == {'number': 0, 'division': 0}
    assert res[1] == {'number': 1, 'division': 5}

@gen_cluster(client=True)
async def test_futures_in_subgraphs(c, s, a, b):
    """Copied from distributed (tests/test_client.py)"""
    dd = pytest.importorskip('dask.dataframe')
    pd = pytest.importorskip('pandas')
    ddf = dd.from_pandas(pd.DataFrame(dict(uid=range(50), enter_time=pd.date_range(start='2020-01-01', end='2020-09-01', periods=50, tz='UTC'))), npartitions=1)
    ddf = ddf[ddf.uid.isin(range(29))].persist()
    ddf['day'] = ddf.enter_time.dt.day_name()
    ddf = await c.submit(dd.categorical.categorize, ddf, columns=['day'], index=False)

@pytest.mark.parametrize('max_branch, expected_layer_type', [(32, SimpleShuffleLayer), (2, ShuffleLayer)])
@gen_cluster(client=True, nthreads=[('', 1)] * 2)
async def test_shuffle_priority(c, s, a, b, max_branch, expected_layer_type):
    pd = pytest.importorskip('pandas')
    dd = pytest.importorskip('dask.dataframe')

    class EnsureSplitsRunImmediatelyPlugin(WorkerPlugin):
        failure = False

        def setup(self, worker):
            if False:
                while True:
                    i = 10
            self.worker = worker

        def transition(self, key, start, finish, **kwargs):
            if False:
                print('Hello World!')
            if finish == 'executing' and (not all(('split' in ts.key for ts in self.worker.state.executing))):
                if any(('split' in ts.key for ts in list(self.worker.state.ready))):
                    EnsureSplitsRunImmediatelyPlugin.failure = True
                    raise RuntimeError('Split tasks are not prioritized')
    await c.register_plugin(EnsureSplitsRunImmediatelyPlugin())
    df = pd.DataFrame({'a': range(1000)})
    ddf = dd.from_pandas(df, npartitions=10)
    ddf2 = ddf.shuffle('a', shuffle='tasks', max_branch=max_branch)
    shuffle_layers = set(ddf2.dask.layers) - set(ddf.dask.layers)
    for layer_name in shuffle_layers:
        assert isinstance(ddf2.dask.layers[layer_name], expected_layer_type)
    await c.compute(ddf2)
    assert not EnsureSplitsRunImmediatelyPlugin.failure

@gen_cluster(client=True)
async def test_map_partitions_da_input(c, s, a, b):
    """Check that map_partitions can handle a dask array input"""
    np = pytest.importorskip('numpy')
    pd = pytest.importorskip('pandas')
    da = pytest.importorskip('dask.array')
    datasets = pytest.importorskip('dask.datasets')

    def f(d, a):
        if False:
            return 10
        assert isinstance(d, pd.DataFrame)
        assert isinstance(a, np.ndarray)
        return d
    df = datasets.timeseries(freq='1d').persist()
    arr = da.ones((1,), chunks=1).persist()
    await c.compute(df.map_partitions(f, arr, meta=df._meta))

def test_map_partitions_df_input():
    if False:
        print('Hello World!')
    '\n    Check that map_partitions can handle a delayed\n    partition of a dataframe input\n    '
    pd = pytest.importorskip('pandas')
    dd = pytest.importorskip('dask.dataframe')

    def f(d, a):
        if False:
            i = 10
            return i + 15
        assert isinstance(d, pd.DataFrame)
        assert isinstance(a, pd.DataFrame)
        return d

    def main():
        if False:
            while True:
                i = 10
        item_df = dd.from_pandas(pd.DataFrame({'a': range(10)}), npartitions=1)
        ddf = item_df.to_delayed()[0].persist()
        merged_df = dd.from_pandas(pd.DataFrame({'b': range(10)}), npartitions=1)
        merged_df = merged_df.shuffle(on='b', shuffle='tasks')
        merged_df.map_partitions(f, ddf, meta=merged_df, enforce_metadata=False).compute()
    with distributed.LocalCluster(scheduler_port=0, dashboard_address=':0', scheduler_kwargs={'dashboard': False}, asynchronous=False, n_workers=1, nthreads=1, processes=False) as cluster:
        with distributed.Client(cluster, asynchronous=False):
            main()

@pytest.mark.filterwarnings('ignore:Running on a single-machine scheduler when a distributed client is active might lead to unexpected results.')
@gen_cluster(client=True)
async def test_to_sql_engine_kwargs(c, s, a, b):
    pd = pytest.importorskip('pandas')
    dd = pytest.importorskip('dask.dataframe')
    pytest.importorskip('sqlalchemy')
    df = pd.DataFrame({'a': range(10), 'b': range(10)})
    df.index.name = 'index'
    ddf = dd.from_pandas(df, npartitions=1)
    with tmpfile() as f:
        uri = f'sqlite:///{f}'
        result = ddf.to_sql('test', uri, index=True, engine_kwargs={'echo': False}, compute=False)
        await c.compute(result)
        dd.utils.assert_eq(ddf, dd.read_sql_table('test', uri, 'index'), check_divisions=False)

@gen_cluster(client=True)
async def test_non_recursive_df_reduce(c, s, a, b):
    dd = pytest.importorskip('dask.dataframe')
    pd = pytest.importorskip('pandas')

    class SomeObject:

        def __init__(self, val):
            if False:
                i = 10
                return i + 15
            self.val = val
    N = 170
    series = pd.Series(data=[1] * N, index=range(2, N + 2))
    dask_series = dd.from_pandas(series, npartitions=34)
    result = dask_series.reduction(chunk=lambda x: x, aggregate=lambda x: SomeObject(x.sum().sum()), split_every=False, token='commit-dataset', meta=object)
    assert (await c.compute(result)).val == 170

def test_set_index_no_resursion_error(c):
    if False:
        i = 10
        return i + 15
    pytest.importorskip('dask.dataframe')
    try:
        ddf = dask.datasets.timeseries(start='2000-01-01', end='2000-07-01', freq='12h').reset_index().astype({'timestamp': str})
        ddf = ddf.set_index('timestamp', sorted=True)
        ddf.compute()
    except RecursionError:
        pytest.fail('dd.set_index triggered a recursion error')

def test_get_scheduler_without_distributed_raises():
    if False:
        for i in range(10):
            print('nop')
    msg = 'no Client'
    with pytest.raises(RuntimeError, match=msg):
        get_scheduler(scheduler='dask.distributed')
    with pytest.raises(RuntimeError, match=msg):
        get_scheduler(scheduler='distributed')

def test_get_scheduler_with_distributed_active(c):
    if False:
        print('Hello World!')
    assert get_scheduler() == c.get
    warning_message = 'Running on a single-machine scheduler when a distributed client is active might lead to unexpected results.'
    with pytest.warns(UserWarning, match=warning_message) as user_warnings_a:
        get_scheduler(scheduler='threads')
        get_scheduler(scheduler='sync')
    assert len(user_warnings_a) == 2

def test_get_scheduler_with_distributed_active_reset_config(c):
    if False:
        print('Hello World!')
    assert get_scheduler() == c.get
    with dask.config.set(scheduler='threads'):
        with pytest.warns(UserWarning):
            assert get_scheduler() != c.get
        with dask.config.set(scheduler=None):
            assert get_scheduler() == c.get

@pytest.mark.parametrize('scheduler, expected_classes', [(None, ('SerializableLock', 'SerializableLock', 'AcquirerProxy')), ('threads', ('SerializableLock', 'SerializableLock', 'SerializableLock')), ('processes', ('AcquirerProxy', 'AcquirerProxy', 'AcquirerProxy'))])
def test_get_scheduler_lock(scheduler, expected_classes):
    if False:
        i = 10
        return i + 15
    da = pytest.importorskip('dask.array', reason='Requires dask.array')
    db = pytest.importorskip('dask.bag', reason='Requires dask.bag')
    dd = pytest.importorskip('dask.dataframe', reason='Requires dask.dataframe')
    darr = da.ones((100,))
    ddf = dd.from_dask_array(darr, columns=['x'])
    dbag = db.range(100, npartitions=2)
    for (collection, expected) in zip((ddf, darr, dbag), expected_classes):
        res = get_scheduler_lock(collection, scheduler=scheduler)
        assert res.__class__.__name__ == expected

@pytest.mark.parametrize('multiprocessing_method', ['spawn', 'fork', 'forkserver'])
def test_get_scheduler_lock_distributed(c, multiprocessing_method):
    if False:
        return 10
    da = pytest.importorskip('dask.array', reason='Requires dask.array')
    dd = pytest.importorskip('dask.dataframe', reason='Requires dask.dataframe')
    darr = da.ones((100,))
    ddf = dd.from_dask_array(darr, columns=['x'])
    dbag = db.range(100, npartitions=2)
    with dask.config.set({'distributed.worker.multiprocessing-method': multiprocessing_method}):
        for collection in (ddf, darr, dbag):
            res = get_scheduler_lock(collection, scheduler='distributed')
            assert isinstance(res, distributed.lock.Lock)

@pytest.mark.skip_with_pyarrow_strings
@pytest.mark.parametrize('lock_param', [True, distributed.lock.Lock()])
def test_write_single_hdf(c, lock_param):
    if False:
        for i in range(10):
            print('nop')
    'https://github.com/dask/dask/issues/9972 and\n    https://github.com/dask/dask/issues/10315\n    '
    pytest.importorskip('dask.dataframe')
    pytest.importorskip('tables')
    with tmpfile(extension='hd5') as f:
        ddf = dask.datasets.timeseries(start='2000-01-01', end='2000-07-01', freq='12h')
        ddf.to_hdf(str(f), key='/ds_*', lock=lock_param)

@gen_cluster(config={'scheduler': 'sync'}, nthreads=[])
async def test_get_scheduler_default_client_config_interleaving(s):
    with pytest.warns(UserWarning):
        assert dask.base.get_scheduler() == dask.local.get_sync
        with dask.config.set(scheduler='threads'):
            assert dask.base.get_scheduler() == dask.threaded.get
            client = await Client(s.address, set_as_default=False, asynchronous=True)
            try:
                assert dask.base.get_scheduler() == dask.threaded.get
            finally:
                await client.close()
            client = await Client(s.address, set_as_default=True, asynchronous=True)
            try:
                assert dask.base.get_scheduler() == client.get
            finally:
                await client.close()
            assert dask.base.get_scheduler() == dask.threaded.get
            client = await Client(s.address, set_as_default=False, asynchronous=True)
            try:
                assert dask.base.get_scheduler() == dask.threaded.get
                with client.as_current():
                    sc = dask.base.get_scheduler()
                    assert sc == client.get
                assert dask.base.get_scheduler() == dask.threaded.get
            finally:
                await client.close()
            client = await Client(s.address, set_as_default=True, asynchronous=True)
            client2 = await Client(s.address, set_as_default=False, asynchronous=True)
            try:
                with client2.as_current():
                    assert dask.base.get_scheduler() == client2.get
                assert dask.base.get_scheduler() == client.get
            finally:
                await client.close()
                await client2.close()
            assert dask.base.get_scheduler() == dask.threaded.get
        assert dask.base.get_scheduler() == dask.local.get_sync
        client = await Client(s.address, set_as_default=True, asynchronous=True)
        try:
            assert dask.base.get_scheduler() == client.get
            with dask.config.set(scheduler='threads'):
                assert dask.base.get_scheduler() == dask.threaded.get
                with client.as_current():
                    assert dask.base.get_scheduler() == client.get
        finally:
            await client.close()

@gen_cluster(client=True)
async def test_bag_groupby_default(c, s, a, b):
    b = db.range(100, npartitions=10)
    b2 = b.groupby(lambda x: x % 13)
    assert not any(('partd' in k[0] for k in b2.dask))