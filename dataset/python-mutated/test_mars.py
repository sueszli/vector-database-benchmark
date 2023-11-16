import mars
import mars.dataframe as md
import pyarrow as pa
import pytest
import ray
from ray.data.tests.test_execution_optimizer import _check_usage_record

@pytest.fixture(scope='module')
def ray_start_regular(request):
    if False:
        i = 10
        return i + 15
    try:
        yield ray.init(num_cpus=16)
    finally:
        ray.shutdown()

def test_mars(ray_start_regular):
    if False:
        i = 10
        return i + 15
    import pandas as pd
    cluster = mars.new_cluster_in_ray(worker_num=2, worker_cpu=1)
    n = 10000
    pdf = pd.DataFrame({'a': list(range(n)), 'b': list(range(n, 2 * n))})
    df = md.DataFrame(pdf)
    ds = ray.data.from_mars(df)
    pd.testing.assert_frame_equal(ds.to_pandas(), df.to_pandas())
    ds2 = ds.filter(lambda row: row['a'] % 2 == 0)
    assert ds2.take(5) == [{'a': 2 * i, 'b': n + 2 * i} for i in range(5)]
    df2 = ds2.to_mars()
    pd.testing.assert_frame_equal(df2.head(5).to_pandas(), pd.DataFrame({'a': list(range(0, 10, 2)), 'b': list(range(n, n + 10, 2))}))
    pdf2 = pd.DataFrame({c: range(5) for c in 'abc'})
    ds3 = ray.data.from_arrow([pa.Table.from_pandas(pdf2) for _ in range(3)])
    df3 = ds3.to_mars()
    pd.testing.assert_frame_equal(df3.head(5).to_pandas(), pdf2)
    cluster.stop()

def test_from_mars_e2e(ray_start_regular, enable_optimizer):
    if False:
        for i in range(10):
            print('nop')
    import pandas as pd
    cluster = mars.new_cluster_in_ray(worker_num=2, worker_cpu=1)
    n = 10000
    pdf = pd.DataFrame({'a': list(range(n)), 'b': list(range(n, 2 * n))})
    df = md.DataFrame(pdf)
    ds = ray.data.from_mars(df)
    assert len(ds.take_all()) == len(df)
    pd.testing.assert_frame_equal(ds.to_pandas(), df.to_pandas())
    assert 'FromPandas' in ds.stats()
    assert ds._plan._logical_plan.dag.name == 'FromPandas'
    _check_usage_record(['FromPandas'])
    ds2 = ds.filter(lambda row: row['a'] % 2 == 0)
    assert ds2.take(5) == [{'a': 2 * i, 'b': n + 2 * i} for i in range(5)]
    assert 'Filter' in ds2.stats()
    assert ds2._plan._logical_plan.dag.name == 'Filter(<lambda>)'
    df2 = ds2.to_mars()
    pd.testing.assert_frame_equal(df2.head(5).to_pandas(), pd.DataFrame({'a': list(range(0, 10, 2)), 'b': list(range(n, n + 10, 2))}))
    _check_usage_record(['Filter', 'FromPandas'])
    pdf2 = pd.DataFrame({c: range(5) for c in 'abc'})
    ds3 = ray.data.from_arrow([pa.Table.from_pandas(pdf2) for _ in range(3)])
    assert len(ds3.take_all())
    df3 = ds3.to_mars()
    pd.testing.assert_frame_equal(df3.head(5).to_pandas(), pdf2)
    assert 'FromArrow' in ds3.stats()
    assert ds3._plan._logical_plan.dag.name == 'FromArrow'
    _check_usage_record(['FromArrow'])
    cluster.stop()
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', __file__]))