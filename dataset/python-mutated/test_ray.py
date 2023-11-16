import sys
import pytest
import ray
from .utils import pipeline, pipeline_with_level

def to_distributed(df):
    if False:
        print('Hello World!')
    return ray.data.from_pandas(df).repartition(2)

@pytest.fixture()
def sample_data(local_data):
    if False:
        for i in range(10):
            print('nop')
    (series, X_df) = local_data
    return (to_distributed(series), to_distributed(X_df))

@pytest.mark.skipif(sys.version_info < (3, 8), reason='requires python >= 3.8')
def test_ray_flow(horizon, sample_data, n_series):
    if False:
        i = 10
        return i + 15
    pipeline(*sample_data, n_series, horizon)

@pytest.mark.skipif(sys.version_info < (3, 8), reason='requires python >= 3.8')
def test_ray_flow_with_level(horizon, sample_data, n_series):
    if False:
        while True:
            i = 10
    pipeline_with_level(*sample_data, n_series, horizon)