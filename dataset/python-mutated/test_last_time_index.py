from datetime import datetime
import pandas as pd
import pytest
from woodwork.logical_types import Categorical, Datetime, Integer
from featuretools.entityset.entityset import LTI_COLUMN_NAME
from featuretools.tests.testing_utils import to_pandas
from featuretools.utils.gen_utils import Library, import_or_none
dd = import_or_none('dask.dataframe')
ps = import_or_none('pyspark.pandas')

@pytest.fixture
def values_es(es):
    if False:
        return 10
    es.normalize_dataframe('log', 'values', 'value', make_time_index=True, new_dataframe_time_index='value_time')
    return es

@pytest.fixture
def true_values_lti():
    if False:
        while True:
            i = 10
    true_values_lti = pd.Series([datetime(2011, 4, 10, 10, 41, 0), datetime(2011, 4, 9, 10, 31, 9), datetime(2011, 4, 9, 10, 31, 18), datetime(2011, 4, 9, 10, 31, 27), datetime(2011, 4, 10, 10, 40, 1), datetime(2011, 4, 10, 10, 41, 3), datetime(2011, 4, 9, 10, 30, 12), datetime(2011, 4, 10, 10, 41, 6), datetime(2011, 4, 9, 10, 30, 18), datetime(2011, 4, 9, 10, 30, 24), datetime(2011, 4, 10, 11, 10, 3)])
    return true_values_lti

@pytest.fixture
def true_sessions_lti():
    if False:
        return 10
    sessions_lti = pd.Series([datetime(2011, 4, 9, 10, 30, 24), datetime(2011, 4, 9, 10, 31, 27), datetime(2011, 4, 9, 10, 40, 0), datetime(2011, 4, 10, 10, 40, 1), datetime(2011, 4, 10, 10, 41, 6), datetime(2011, 4, 10, 11, 10, 3)])
    return sessions_lti

@pytest.fixture
def wishlist_df():
    if False:
        i = 10
        return i + 15
    wishlist_df = pd.DataFrame({'session_id': [0, 1, 2, 2, 3, 4, 5], 'datetime': [datetime(2011, 4, 9, 10, 30, 15), datetime(2011, 4, 9, 10, 31, 30), datetime(2011, 4, 9, 10, 30, 30), datetime(2011, 4, 9, 10, 35, 30), datetime(2011, 4, 10, 10, 41, 0), datetime(2011, 4, 10, 10, 39, 59), datetime(2011, 4, 10, 11, 10, 2)], 'product_id': ['coke zero', 'taco clock', 'coke zero', 'car', 'toothpaste', 'brown bag', 'coke zero']})
    return wishlist_df

@pytest.fixture
def extra_session_df(es):
    if False:
        for i in range(10):
            print('nop')
    row_values = {'customer_id': 2, 'device_name': 'PC', 'device_type': 0, 'id': 6}
    row = pd.DataFrame(row_values, index=pd.Index([6], name='id'))
    df = to_pandas(es['sessions'])
    df = pd.concat([df, row]).sort_index()
    if es.dataframe_type == Library.DASK:
        df = dd.from_pandas(df, npartitions=3)
    elif es.dataframe_type == Library.SPARK:
        df = df.astype('string')
        df = ps.from_pandas(df)
    return df

class TestLastTimeIndex(object):

    def test_leaf(self, es):
        if False:
            for i in range(10):
                print('nop')
        es.add_last_time_indexes()
        log = es['log']
        lti_name = log.ww.metadata.get('last_time_index')
        assert lti_name == LTI_COLUMN_NAME
        assert len(log[lti_name]) == 17
        log_df = to_pandas(log)
        for (v1, v2) in zip(log_df[lti_name], log_df['datetime']):
            assert pd.isnull(v1) and pd.isnull(v2) or v1 == v2

    def test_leaf_no_time_index(self, es):
        if False:
            i = 10
            return i + 15
        es.add_last_time_indexes()
        stores = es['stores']
        true_lti = pd.Series([None for x in range(6)], dtype='datetime64[ns]')
        assert len(true_lti) == len(stores[LTI_COLUMN_NAME])
        stores_lti = to_pandas(stores[LTI_COLUMN_NAME])
        for (v1, v2) in zip(stores_lti, true_lti):
            assert pd.isnull(v1) and pd.isnull(v2) or v1 == v2

    def test_parent(self, values_es, true_values_lti):
        if False:
            print('Hello World!')
        if values_es.dataframe_type != Library.PANDAS:
            pytest.xfail('possible issue with either normalize_dataframe or add_last_time_indexes')
        values_es.add_last_time_indexes()
        values = values_es['values']
        lti_name = values.ww.metadata.get('last_time_index')
        assert len(values[lti_name]) == 10
        sorted_lti = to_pandas(values[lti_name]).sort_index()
        for (v1, v2) in zip(sorted_lti, true_values_lti):
            assert pd.isnull(v1) and pd.isnull(v2) or v1 == v2

    def test_parent_some_missing(self, values_es, true_values_lti):
        if False:
            return 10
        if values_es.dataframe_type != Library.PANDAS:
            pytest.xfail('fails with Dask, tests needs to be reworked')
        values = values_es['values']
        row_values = {'value': [21.0], 'value_time': [pd.Timestamp('2011-04-10 11:10:02')]}
        row = pd.DataFrame(row_values, index=pd.Index([21]))
        df = pd.concat([values, row])
        df = df.sort_values(by='value')
        df.index.name = None
        values_es.replace_dataframe(dataframe_name='values', df=df)
        values_es.add_last_time_indexes()
        true_values_lti[10] = pd.Timestamp('2011-04-10 11:10:02')
        true_values_lti[11] = pd.Timestamp('2011-04-10 11:10:03')
        values = values_es['values']
        lti_name = values.ww.metadata.get('last_time_index')
        assert len(values[lti_name]) == 11
        sorted_lti = values[lti_name].sort_index()
        for (v1, v2) in zip(sorted_lti, true_values_lti):
            assert pd.isnull(v1) and pd.isnull(v2) or v1 == v2

    def test_parent_no_time_index(self, es, true_sessions_lti):
        if False:
            return 10
        es.add_last_time_indexes()
        sessions = es['sessions']
        lti_name = sessions.ww.metadata.get('last_time_index')
        assert len(sessions[lti_name]) == 6
        sorted_lti = to_pandas(sessions[lti_name]).sort_index()
        for (v1, v2) in zip(sorted_lti, true_sessions_lti):
            assert pd.isnull(v1) and pd.isnull(v2) or v1 == v2

    def test_parent_no_time_index_missing(self, es, extra_session_df, true_sessions_lti):
        if False:
            return 10
        es.replace_dataframe(dataframe_name='sessions', df=extra_session_df)
        es.add_last_time_indexes()
        true_sessions_lti[6] = pd.NaT
        sessions = es['sessions']
        lti_name = sessions.ww.metadata.get('last_time_index')
        assert len(sessions[lti_name]) == 7
        sorted_lti = to_pandas(sessions[lti_name]).sort_index()
        for (v1, v2) in zip(sorted_lti, true_sessions_lti):
            assert pd.isnull(v1) and pd.isnull(v2) or v1 == v2

    def test_multiple_children(self, es, wishlist_df, true_sessions_lti):
        if False:
            print('Hello World!')
        if es.dataframe_type == Library.SPARK:
            pytest.xfail('Cannot make index on a Spark DataFrame')
        if es.dataframe_type == Library.DASK:
            wishlist_df = dd.from_pandas(wishlist_df, npartitions=2)
        logical_types = {'session_id': Integer, 'datetime': Datetime, 'product_id': Categorical}
        es.add_dataframe(dataframe_name='wishlist_log', dataframe=wishlist_df, index='id', make_index=True, time_index='datetime', logical_types=logical_types)
        es.add_relationship('sessions', 'id', 'wishlist_log', 'session_id')
        es.add_last_time_indexes()
        sessions = es['sessions']
        true_sessions_lti[1] = pd.Timestamp('2011-4-9 10:31:30')
        true_sessions_lti[3] = pd.Timestamp('2011-4-10 10:41:00')
        lti_name = sessions.ww.metadata.get('last_time_index')
        assert len(sessions[lti_name]) == 6
        sorted_lti = to_pandas(sessions[lti_name]).sort_index()
        for (v1, v2) in zip(sorted_lti, true_sessions_lti):
            assert pd.isnull(v1) and pd.isnull(v2) or v1 == v2

    def test_multiple_children_right_missing(self, es, wishlist_df, true_sessions_lti):
        if False:
            i = 10
            return i + 15
        if es.dataframe_type == Library.SPARK:
            pytest.xfail('Cannot make index on a Spark DataFrame')
        wishlist_df.drop(4, inplace=True)
        if es.dataframe_type == Library.DASK:
            wishlist_df = dd.from_pandas(wishlist_df, npartitions=2)
        logical_types = {'session_id': Integer, 'datetime': Datetime, 'product_id': Categorical}
        es.add_dataframe(dataframe_name='wishlist_log', dataframe=wishlist_df, index='id', make_index=True, time_index='datetime', logical_types=logical_types)
        es.add_relationship('sessions', 'id', 'wishlist_log', 'session_id')
        es.add_last_time_indexes()
        sessions = es['sessions']
        true_sessions_lti[1] = pd.Timestamp('2011-4-9 10:31:30')
        lti_name = sessions.ww.metadata.get('last_time_index')
        assert len(sessions[lti_name]) == 6
        sorted_lti = to_pandas(sessions[lti_name]).sort_index()
        for (v1, v2) in zip(sorted_lti, true_sessions_lti):
            assert pd.isnull(v1) and pd.isnull(v2) or v1 == v2

    def test_multiple_children_left_missing(self, es, extra_session_df, wishlist_df, true_sessions_lti):
        if False:
            return 10
        if es.dataframe_type == Library.SPARK:
            pytest.xfail('Cannot make index on a Spark DataFrame')
        es.replace_dataframe(dataframe_name='sessions', df=extra_session_df)
        row_values = {'session_id': [6], 'datetime': [pd.Timestamp('2011-04-11 11:11:11')], 'product_id': ['toothpaste']}
        row = pd.DataFrame(row_values, index=pd.RangeIndex(start=7, stop=8))
        df = pd.concat([wishlist_df, row])
        if es.dataframe_type == Library.DASK:
            df = dd.from_pandas(df, npartitions=2)
        logical_types = {'session_id': Integer, 'datetime': Datetime, 'product_id': Categorical}
        es.add_dataframe(dataframe_name='wishlist_log', dataframe=df, index='id', make_index=True, time_index='datetime', logical_types=logical_types)
        es.add_relationship('sessions', 'id', 'wishlist_log', 'session_id')
        es.add_last_time_indexes()
        sessions = es['sessions']
        true_sessions_lti[1] = pd.Timestamp('2011-4-9 10:31:30')
        true_sessions_lti[3] = pd.Timestamp('2011-4-10 10:41:00')
        true_sessions_lti[6] = pd.Timestamp('2011-04-11 11:11:11')
        lti_name = sessions.ww.metadata.get('last_time_index')
        assert len(sessions[lti_name]) == 7
        sorted_lti = to_pandas(sessions[lti_name]).sort_index()
        for (v1, v2) in zip(sorted_lti, true_sessions_lti):
            assert pd.isnull(v1) and pd.isnull(v2) or v1 == v2

    def test_multiple_children_all_combined(self, es, extra_session_df, wishlist_df, true_sessions_lti):
        if False:
            return 10
        if es.dataframe_type == Library.SPARK:
            pytest.xfail('Cannot make index on a Spark DataFrame')
        es.replace_dataframe(dataframe_name='sessions', df=extra_session_df)
        row_values = {'session_id': [6], 'datetime': [pd.Timestamp('2011-04-11 11:11:11')], 'product_id': ['toothpaste']}
        row = pd.DataFrame(row_values, index=pd.RangeIndex(start=7, stop=8))
        df = pd.concat([wishlist_df, row])
        df.drop(4, inplace=True)
        if es.dataframe_type == Library.DASK:
            df = dd.from_pandas(df, npartitions=2)
        logical_types = {'session_id': Integer, 'datetime': Datetime, 'product_id': Categorical}
        es.add_dataframe(dataframe_name='wishlist_log', dataframe=df, index='id', make_index=True, time_index='datetime', logical_types=logical_types)
        es.add_relationship('sessions', 'id', 'wishlist_log', 'session_id')
        es.add_last_time_indexes()
        sessions = es['sessions']
        true_sessions_lti[1] = pd.Timestamp('2011-4-9 10:31:30')
        true_sessions_lti[6] = pd.Timestamp('2011-04-11 11:11:11')
        lti_name = sessions.ww.metadata.get('last_time_index')
        assert len(sessions[lti_name]) == 7
        sorted_lti = to_pandas(sessions[lti_name]).sort_index()
        for (v1, v2) in zip(sorted_lti, true_sessions_lti):
            assert pd.isnull(v1) and pd.isnull(v2) or v1 == v2

    def test_multiple_children_both_missing(self, es, extra_session_df, wishlist_df, true_sessions_lti):
        if False:
            while True:
                i = 10
        if es.dataframe_type == Library.SPARK:
            pytest.xfail('Cannot make index on a Spark DataFrame')
        sessions = es['sessions']
        if es.dataframe_type == Library.DASK:
            wishlist_df = dd.from_pandas(wishlist_df, npartitions=2)
        logical_types = {'session_id': Integer, 'datetime': Datetime, 'product_id': Categorical}
        es.replace_dataframe(dataframe_name='sessions', df=extra_session_df)
        es.add_dataframe(dataframe_name='wishlist_log', dataframe=wishlist_df, index='id', make_index=True, time_index='datetime', logical_types=logical_types)
        es.add_relationship('sessions', 'id', 'wishlist_log', 'session_id')
        es.add_last_time_indexes()
        sessions = es['sessions']
        true_sessions_lti[1] = pd.Timestamp('2011-4-9 10:31:30')
        true_sessions_lti[3] = pd.Timestamp('2011-4-10 10:41:00')
        true_sessions_lti[6] = pd.NaT
        lti_name = sessions.ww.metadata.get('last_time_index')
        assert len(sessions[lti_name]) == 7
        sorted_lti = to_pandas(sessions[lti_name]).sort_index()
        for (v1, v2) in zip(sorted_lti, true_sessions_lti):
            assert pd.isnull(v1) and pd.isnull(v2) or v1 == v2

    def test_grandparent(self, es):
        if False:
            return 10
        log = es['log']
        df = to_pandas(log)
        df['datetime'][5] = pd.Timestamp('2011-4-09 10:40:01')
        df = df.set_index('datetime', append=True).sort_index(level=[1, 0], kind='mergesort').reset_index('datetime', drop=False)
        if es.dataframe_type == Library.DASK:
            df = dd.from_pandas(df, npartitions=2)
        if es.dataframe_type == Library.SPARK:
            df = ps.from_pandas(df)
        es.replace_dataframe(dataframe_name='log', df=df)
        es.add_last_time_indexes()
        customers = es['customers']
        true_customers_lti = pd.Series([datetime(2011, 4, 9, 10, 40, 1), datetime(2011, 4, 10, 10, 41, 6), datetime(2011, 4, 10, 11, 10, 3)])
        lti_name = customers.ww.metadata.get('last_time_index')
        assert len(customers[lti_name]) == 3
        sorted_lti = to_pandas(customers).sort_values('id')[lti_name]
        for (v1, v2) in zip(sorted_lti, true_customers_lti):
            assert pd.isnull(v1) and pd.isnull(v2) or v1 == v2