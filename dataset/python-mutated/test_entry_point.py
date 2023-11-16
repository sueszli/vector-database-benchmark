import pandas as pd
import pytest
from featuretools import dfs

@pytest.fixture
def pd_entry_point_dfs():
    if False:
        return 10
    cards_df = pd.DataFrame({'id': [1, 2, 3, 4, 5]})
    transactions_df = pd.DataFrame({'id': [1, 2, 3, 4, 5, 6], 'card_id': [1, 2, 1, 3, 4, 5], 'transaction_time': [10, 12, 13, 20, 21, 20], 'fraud': [True, False, True, False, True, True]})
    return (cards_df, transactions_df)

@pytest.fixture
def dask_entry_point_dfs(pd_entry_point_dfs):
    if False:
        return 10
    dd = pytest.importorskip('dask.dataframe', reason='Dask not installed, skipping')
    cards_df = dd.from_pandas(pd_entry_point_dfs[0], npartitions=2)
    transactions_df = dd.from_pandas(pd_entry_point_dfs[1], npartitions=2)
    return (cards_df, transactions_df)

@pytest.fixture(params=['pd_entry_point_dfs', 'dask_entry_point_dfs'])
def entry_points_dfs(request):
    if False:
        return 10
    return request.getfixturevalue(request.param)

class MockEntryPoint(object):

    def on_call(self, kwargs):
        if False:
            print('Hello World!')
        self.kwargs = kwargs

    def on_error(self, error, runtime):
        if False:
            for i in range(10):
                print('nop')
        self.error = error

    def on_return(self, return_value, runtime):
        if False:
            i = 10
            return i + 15
        self.return_value = return_value

    def load(self):
        if False:
            while True:
                i = 10
        return self

    def __call__(self):
        if False:
            i = 10
            return i + 15
        return self

class MockPkgResources(object):

    def __init__(self, entry_point):
        if False:
            print('Hello World!')
        self.entry_point = entry_point

    def iter_entry_points(self, name):
        if False:
            print('Hello World!')
        return [self.entry_point]

def test_entry_point(es, monkeypatch):
    if False:
        i = 10
        return i + 15
    entry_point = MockEntryPoint()
    monkeypatch.setitem(dfs.__globals__['entry_point'].__globals__, 'pkg_resources', MockPkgResources(entry_point))
    (fm, fl) = dfs(entityset=es, target_dataframe_name='customers')
    assert 'entityset' in entry_point.kwargs.keys()
    assert 'target_dataframe_name' in entry_point.kwargs.keys()
    assert (fm, fl) == entry_point.return_value

def test_entry_point_error(es, monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    entry_point = MockEntryPoint()
    monkeypatch.setitem(dfs.__globals__['entry_point'].__globals__, 'pkg_resources', MockPkgResources(entry_point))
    with pytest.raises(KeyError):
        dfs(entityset=es, target_dataframe_name='missing_dataframe')
    assert isinstance(entry_point.error, KeyError)

def test_entry_point_detect_arg(monkeypatch, entry_points_dfs):
    if False:
        print('Hello World!')
    cards_df = entry_points_dfs[0]
    transactions_df = entry_points_dfs[1]
    cards_df = pd.DataFrame({'id': [1, 2, 3, 4, 5]})
    transactions_df = pd.DataFrame({'id': [1, 2, 3, 4, 5, 6], 'card_id': [1, 2, 1, 3, 4, 5], 'transaction_time': [10, 12, 13, 20, 21, 20], 'fraud': [True, False, True, False, True, True]})
    dataframes = {'cards': (cards_df, 'id'), 'transactions': (transactions_df, 'id', 'transaction_time')}
    relationships = [('cards', 'id', 'transactions', 'card_id')]
    entry_point = MockEntryPoint()
    monkeypatch.setitem(dfs.__globals__['entry_point'].__globals__, 'pkg_resources', MockPkgResources(entry_point))
    (fm, fl) = dfs(dataframes, relationships, target_dataframe_name='cards')
    assert 'dataframes' in entry_point.kwargs.keys()
    assert 'relationships' in entry_point.kwargs.keys()
    assert 'target_dataframe_name' in entry_point.kwargs.keys()