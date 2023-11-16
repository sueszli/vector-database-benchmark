from parameterized import parameterized
from sqlalchemy import column
from superset.constants import TimeGrain
from superset.db_engine_specs.elasticsearch import ElasticSearchEngineSpec
from tests.integration_tests.db_engine_specs.base_tests import TestDbEngineSpec

class TestElasticsearchDbEngineSpec(TestDbEngineSpec):

    @parameterized.expand([[TimeGrain.SECOND, "DATE_TRUNC('second', ts)"], [TimeGrain.MINUTE, "DATE_TRUNC('minute', ts)"], [TimeGrain.HOUR, "DATE_TRUNC('hour', ts)"], [TimeGrain.DAY, "DATE_TRUNC('day', ts)"], [TimeGrain.WEEK, "DATE_TRUNC('week', ts)"], [TimeGrain.MONTH, "DATE_TRUNC('month', ts)"], [TimeGrain.YEAR, "DATE_TRUNC('year', ts)"]])
    def test_time_grain_expressions(self, time_grain, expected_time_grain_expression):
        if False:
            return 10
        col = column('ts')
        col.type = 'DATETIME'
        actual = ElasticSearchEngineSpec.get_timestamp_expr(col=col, pdf=None, time_grain=time_grain)
        self.assertEqual(str(actual), expected_time_grain_expression)