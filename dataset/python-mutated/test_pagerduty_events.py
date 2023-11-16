from __future__ import annotations
import pytest
from airflow.exceptions import AirflowProviderDeprecationWarning
from airflow.models import Connection
from airflow.providers.pagerduty.hooks.pagerduty import PagerdutyEventsHook
from airflow.utils import db
pytestmark = pytest.mark.db_test
DEFAULT_CONN_ID = 'pagerduty_events_default'

@pytest.fixture(scope='class')
def events_connections():
    if False:
        print('Hello World!')
    db.merge_conn(Connection(conn_id=DEFAULT_CONN_ID, conn_type='pagerduty_events', password='events_token'))

class TestPagerdutyEventsHook:

    def test_get_integration_key_from_password(self, events_connections):
        if False:
            return 10
        hook = PagerdutyEventsHook(pagerduty_events_conn_id=DEFAULT_CONN_ID)
        assert hook.integration_key == 'events_token', 'token initialised.'

    def test_token_parameter_override(self, events_connections):
        if False:
            return 10
        hook = PagerdutyEventsHook(integration_key='override_key', pagerduty_events_conn_id=DEFAULT_CONN_ID)
        assert hook.integration_key == 'override_key', 'token initialised.'

    def test_create_event(self, requests_mock, events_connections):
        if False:
            i = 10
            return i + 15
        hook = PagerdutyEventsHook(pagerduty_events_conn_id=DEFAULT_CONN_ID)
        mock_response_body = {'status': 'success', 'message': 'Event processed', 'dedup_key': 'samplekeyhere'}
        requests_mock.post('https://events.pagerduty.com/v2/enqueue', json=mock_response_body)
        resp = hook.create_event(summary='test', source='airflow_test', severity='error')
        assert resp == mock_response_body

    def test_create_change_event(self, requests_mock, events_connections):
        if False:
            print('Hello World!')
        hook = PagerdutyEventsHook(pagerduty_events_conn_id=DEFAULT_CONN_ID)
        change_event_id = 'change_event_id'
        mock_response_body = {'id': change_event_id}
        requests_mock.post('https://events.pagerduty.com/v2/change/enqueue', json=mock_response_body)
        resp = hook.create_change_event(summary='test', source='airflow')
        assert resp == change_event_id

    def test_send_event(self, requests_mock, events_connections):
        if False:
            print('Hello World!')
        hook = PagerdutyEventsHook(pagerduty_events_conn_id=DEFAULT_CONN_ID)
        dedup_key = 'samplekeyhere'
        mock_response_body = {'status': 'success', 'message': 'Event processed', 'dedup_key': dedup_key}
        requests_mock.post('https://events.pagerduty.com/v2/enqueue', json=mock_response_body)
        resp = hook.send_event(summary='test', source='airflow_test', severity='error', dedup_key=dedup_key)
        assert resp == dedup_key

    def test_create_event_deprecation_warning(self, requests_mock, events_connections):
        if False:
            i = 10
            return i + 15
        hook = PagerdutyEventsHook(pagerduty_events_conn_id=DEFAULT_CONN_ID)
        mock_response_body = {'status': 'success', 'message': 'Event processed', 'dedup_key': 'samplekeyhere'}
        requests_mock.post('https://events.pagerduty.com/v2/enqueue', json=mock_response_body)
        warning = 'This method will be deprecated. Please use the `PagerdutyEventsHook.send_event` to interact with the Events API'
        with pytest.warns(AirflowProviderDeprecationWarning, match=warning):
            hook.create_event(summary='test', source='airflow_test', severity='error')