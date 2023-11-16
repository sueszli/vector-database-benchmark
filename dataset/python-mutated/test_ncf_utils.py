import pytest
from unittest.mock import Mock
from recommenders.tuning.nni.ncf_utils import compute_test_results
from recommenders.datasets.movielens import MockMovielensSchema
DATA_SIZE = 1

@pytest.fixture(scope='module')
def mock_model():
    if False:
        return 10

    def mock_predict(*args, is_list=False):
        if False:
            print('Hello World!')
        'Mock model predict method'
        if is_list:
            return [0] * DATA_SIZE
        else:
            return 0
    mock_model = Mock()
    mock_model.predict.side_effect = mock_predict
    return mock_model

@pytest.fixture(scope='module')
def fake_movielens_df():
    if False:
        i = 10
        return i + 15
    return MockMovielensSchema.get_df(size=DATA_SIZE)

def test_compute_test_results__return_success(mock_model, fake_movielens_df):
    if False:
        print('Hello World!')
    mock_metric_func = 'lambda *args, **kwargs: 0'
    compute_test_results(mock_model, fake_movielens_df, fake_movielens_df, [mock_metric_func], [mock_metric_func])
    assert mock_model.predict.is_called