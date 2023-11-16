from sqlalchemy import column
from superset.db_engine_specs.pinot import PinotEngineSpec
from tests.integration_tests.db_engine_specs.base_tests import TestDbEngineSpec

class TestPinotDbEngineSpec(TestDbEngineSpec):
    """Tests pertaining to our Pinot database support"""

    def test_pinot_time_expression_sec_one_1d_grain(self):
        if False:
            print('Hello World!')
        col = column('tstamp')
        expr = PinotEngineSpec.get_timestamp_expr(col, 'epoch_s', 'P1D')
        result = str(expr.compile())
        expected = "CAST(DATE_TRUNC('day', CAST(" + "DATETIMECONVERT(tstamp, '1:SECONDS:EPOCH', " + "'1:SECONDS:EPOCH', '1:SECONDS') AS TIMESTAMP)) AS TIMESTAMP)"
        self.assertEqual(result, expected)

    def test_pinot_time_expression_simple_date_format_1d_grain(self):
        if False:
            print('Hello World!')
        col = column('tstamp')
        expr = PinotEngineSpec.get_timestamp_expr(col, '%Y-%m-%d %H:%M:%S', 'P1D')
        result = str(expr.compile())
        expected = "CAST(DATE_TRUNC('day', CAST(tstamp AS TIMESTAMP)) AS TIMESTAMP)"
        self.assertEqual(result, expected)

    def test_pinot_time_expression_simple_date_format_10m_grain(self):
        if False:
            while True:
                i = 10
        col = column('tstamp')
        expr = PinotEngineSpec.get_timestamp_expr(col, '%Y-%m-%d %H:%M:%S', 'PT10M')
        result = str(expr.compile())
        expected = "CAST(ROUND(DATE_TRUNC('minute', CAST(tstamp AS " + 'TIMESTAMP)), 600000) AS TIMESTAMP)'
        self.assertEqual(result, expected)

    def test_pinot_time_expression_simple_date_format_1w_grain(self):
        if False:
            return 10
        col = column('tstamp')
        expr = PinotEngineSpec.get_timestamp_expr(col, '%Y-%m-%d %H:%M:%S', 'P1W')
        result = str(expr.compile())
        expected = "CAST(DATE_TRUNC('week', CAST(tstamp AS TIMESTAMP)) AS TIMESTAMP)"
        self.assertEqual(result, expected)

    def test_pinot_time_expression_sec_one_1m_grain(self):
        if False:
            while True:
                i = 10
        col = column('tstamp')
        expr = PinotEngineSpec.get_timestamp_expr(col, 'epoch_s', 'P1M')
        result = str(expr.compile())
        expected = "CAST(DATE_TRUNC('month', CAST(" + "DATETIMECONVERT(tstamp, '1:SECONDS:EPOCH', " + "'1:SECONDS:EPOCH', '1:SECONDS') AS TIMESTAMP)) AS TIMESTAMP)"
        self.assertEqual(result, expected)

    def test_invalid_get_time_expression_arguments(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(NotImplementedError):
            PinotEngineSpec.get_timestamp_expr(column('tstamp'), None, 'P0.25Y')
        with self.assertRaises(NotImplementedError):
            PinotEngineSpec.get_timestamp_expr(column('tstamp'), 'epoch_s', 'invalid_grain')