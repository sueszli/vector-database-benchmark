from __future__ import annotations
from unittest import mock
import pytest
from airflow.models import Connection
from airflow.providers.pagerduty.hooks.pagerduty import PagerdutyHook
from airflow.providers.pagerduty.hooks.pagerduty_events import PagerdutyEventsHook
from airflow.utils import db
pytestmark = pytest.mark.db_test
DEFAULT_CONN_ID = 'pagerduty_default'

@pytest.fixture(scope='class')
def pagerduty_connections():
    if False:
        while True:
            i = 10
    db.merge_conn(Connection(conn_id=DEFAULT_CONN_ID, conn_type='pagerduty', password='token', extra='{"routing_key": "integration_key"}'))
    db.merge_conn(Connection(conn_id='pagerduty_no_extra', conn_type='pagerduty', password='pagerduty_token_without_extra'))

class TestPagerdutyHook:

    def test_get_token_from_password(self, pagerduty_connections):
        if False:
            print('Hello World!')
        hook = PagerdutyHook(pagerduty_conn_id=DEFAULT_CONN_ID)
        assert hook.token == 'token', 'token initialised.'
        assert hook.routing_key == 'integration_key'

    def test_without_routing_key_extra(self):
        if False:
            print('Hello World!')
        hook = PagerdutyHook(pagerduty_conn_id='pagerduty_no_extra')
        assert hook.token == 'pagerduty_token_without_extra', 'token initialised.'
        assert hook.routing_key is None, 'default routing key skipped.'

    def test_token_parameter_override(self):
        if False:
            i = 10
            return i + 15
        hook = PagerdutyHook(token='pagerduty_param_token', pagerduty_conn_id=DEFAULT_CONN_ID)
        assert hook.token == 'pagerduty_param_token', 'token initialised.'

    def test_get_service(self, requests_mock):
        if False:
            i = 10
            return i + 15
        hook = PagerdutyHook(pagerduty_conn_id=DEFAULT_CONN_ID)
        mock_response_body = {'id': 'PZYX321', 'name': 'Apache Airflow', 'status': 'active', 'type': 'service', 'summary': 'Apache Airflow', 'self': 'https://api.pagerduty.com/services/PZYX321'}
        requests_mock.get('https://api.pagerduty.com/services/PZYX321', json={'service': mock_response_body})
        session = hook.get_session()
        resp = session.rget('/services/PZYX321')
        assert resp == mock_response_body

    @mock.patch.object(PagerdutyEventsHook, '__init__')
    @mock.patch.object(PagerdutyEventsHook, 'create_event')
    def test_create_event(self, events_hook_create_event, events_hook_init):
        if False:
            i = 10
            return i + 15
        events_hook_init.return_value = None
        hook = PagerdutyHook(pagerduty_conn_id=DEFAULT_CONN_ID)
        hook.create_event(summary='test', source='airflow_test', severity='error')
        events_hook_init.assert_called_with(integration_key='integration_key')
        events_hook_create_event.assert_called_with(summary='test', source='airflow_test', severity='error', action='trigger', dedup_key=None, custom_details=None, group=None, component=None, class_type=None, images=None, links=None)

    @mock.patch.object(PagerdutyEventsHook, 'create_event', mock.MagicMock(return_value=None))
    @mock.patch.object(PagerdutyEventsHook, '__init__')
    def test_create_event_override(self, events_hook_init):
        if False:
            print('Hello World!')
        events_hook_init.return_value = None
        hook = PagerdutyHook(pagerduty_conn_id=DEFAULT_CONN_ID)
        hook.create_event(routing_key='different_key', summary='test', source='airflow_test', severity='error')
        events_hook_init.assert_called_with(integration_key='different_key')