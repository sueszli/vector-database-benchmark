from pandas import DataFrame, Series, Timestamp
from pandas.testing import assert_frame_equal
from pytest import fixture, mark
from superset.common.chart_data import ChartDataResultFormat, ChartDataResultType
from superset.common.query_context import QueryContext
from superset.common.query_context_processor import AGGREGATED_JOIN_COLUMN, QueryContextProcessor
from superset.connectors.base.models import BaseDatasource
from superset.constants import TimeGrain
query_context_processor = QueryContextProcessor(QueryContext(datasource=BaseDatasource(), queries=[], result_type=ChartDataResultType.COLUMNS, form_data={}, slice_=None, result_format=ChartDataResultFormat.CSV, cache_values={}))

@fixture
def make_join_column_producer():
    if False:
        print('Hello World!')

    def join_column_producer(row: Series, column_index: int) -> str:
        if False:
            i = 10
            return i + 15
        return 'CUSTOM_FORMAT'
    return join_column_producer

@mark.parametrize(('time_grain', 'expected'), [(TimeGrain.WEEK, '2020-W01'), (TimeGrain.MONTH, '2020-01'), (TimeGrain.QUARTER, '2020-Q1'), (TimeGrain.YEAR, '2020')])
def test_aggregated_join_column(time_grain: str, expected: str):
    if False:
        print('Hello World!')
    df = DataFrame({'ds': [Timestamp('2020-01-07')]})
    query_context_processor.add_aggregated_join_column(df, time_grain)
    result = DataFrame({'ds': [Timestamp('2020-01-07')], AGGREGATED_JOIN_COLUMN: [expected]})
    assert_frame_equal(df, result)

def test_aggregated_join_column_producer(make_join_column_producer):
    if False:
        for i in range(10):
            print('nop')
    df = DataFrame({'ds': [Timestamp('2020-01-07')]})
    query_context_processor.add_aggregated_join_column(df, TimeGrain.YEAR, make_join_column_producer)
    result = DataFrame({'ds': [Timestamp('2020-01-07')], AGGREGATED_JOIN_COLUMN: ['CUSTOM_FORMAT']})
    assert_frame_equal(df, result)