"""Unit tests for Superset"""
from unittest import mock
import pytest
from marshmallow import ValidationError
from tests.integration_tests.test_app import app
from superset.charts.schemas import ChartDataQueryContextSchema
from tests.integration_tests.base_tests import SupersetTestCase
from tests.integration_tests.fixtures.birth_names_dashboard import load_birth_names_dashboard_with_slices, load_birth_names_data
from tests.integration_tests.fixtures.query_context import get_query_context

class TestSchema(SupersetTestCase):

    @mock.patch('superset.common.query_context_factory.config', {**app.config, 'ROW_LIMIT': 5000})
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_query_context_limit_and_offset(self):
        if False:
            return 10
        self.login(username='admin')
        payload = get_query_context('birth_names')
        payload['queries'][0]['row_limit'] = -1
        payload['queries'][0]['row_offset'] = -1
        with self.assertRaises(ValidationError) as context:
            _ = ChartDataQueryContextSchema().load(payload)
        self.assertIn('row_limit', context.exception.messages['queries'][0])
        self.assertIn('row_offset', context.exception.messages['queries'][0])

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_query_context_null_timegrain(self):
        if False:
            i = 10
            return i + 15
        self.login(username='admin')
        payload = get_query_context('birth_names')
        payload['queries'][0]['extras']['time_grain_sqla'] = None
        _ = ChartDataQueryContextSchema().load(payload)

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_query_context_series_limit(self):
        if False:
            for i in range(10):
                print('nop')
        self.login(username='admin')
        payload = get_query_context('birth_names')
        payload['queries'][0]['timeseries_limit'] = 2
        payload['queries'][0]['timeseries_limit_metric'] = {'expressionType': 'SIMPLE', 'column': {'id': 334, 'column_name': 'gender', 'filterable': True, 'groupby': True, 'is_dttm': False, 'type': 'VARCHAR(16)', 'optionName': '_col_gender'}, 'aggregate': 'COUNT_DISTINCT', 'label': 'COUNT_DISTINCT(gender)'}
        _ = ChartDataQueryContextSchema().load(payload)