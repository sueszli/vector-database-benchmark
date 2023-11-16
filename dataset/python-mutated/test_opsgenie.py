from __future__ import annotations
from unittest import mock
import pytest
from opsgenie_sdk import AlertApi, CloseAlertPayload, CreateAlertPayload
from opsgenie_sdk.exceptions import AuthenticationException
from airflow.models import Connection
from airflow.providers.opsgenie.hooks.opsgenie import OpsgenieAlertHook
from airflow.utils import db
pytestmark = pytest.mark.db_test

class TestOpsgenieAlertHook:
    conn_id = 'opsgenie_conn_id_test'
    opsgenie_alert_endpoint = 'https://api.opsgenie.com/v2/alerts'
    _create_alert_payload = {'message': 'An example alert message', 'alias': 'Life is too short for no alias', 'description': 'Every alert needs a description', 'responders': [{'id': '4513b7ea-3b91-438f-b7e4-e3e54af9147c', 'type': 'team'}, {'name': 'NOC', 'type': 'team'}, {'id': 'bb4d9938-c3c2-455d-aaab-727aa701c0d8', 'type': 'user'}, {'username': 'trinity@opsgenie.com', 'type': 'user'}, {'id': 'aee8a0de-c80f-4515-a232-501c0bc9d715', 'type': 'escalation'}, {'name': 'Nightwatch Escalation', 'type': 'escalation'}, {'id': '80564037-1984-4f38-b98e-8a1f662df552', 'type': 'schedule'}, {'name': 'First Responders Schedule', 'type': 'schedule'}], 'visible_to': [{'id': '4513b7ea-3b91-438f-b7e4-e3e54af9147c', 'type': 'team'}, {'name': 'rocket_team', 'type': 'team'}, {'id': 'bb4d9938-c3c2-455d-aaab-727aa701c0d8', 'type': 'user'}, {'username': 'trinity@opsgenie.com', 'type': 'user'}], 'actions': ['Restart', 'AnExampleAction'], 'tags': ['OverwriteQuietHours', 'Critical'], 'details': {'key1': 'value1', 'key2': 'value2'}, 'entity': 'An example entity', 'source': 'Airflow', 'priority': 'P1', 'user': 'Jesse', 'note': 'Write this down'}
    _mock_success_response_body = {'result': 'Request will be processed', 'took': 0.302, 'request_id': '43a29c5c-3dbf-4fa4-9c26-f4f71023e120'}

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        db.merge_conn(Connection(conn_id=self.conn_id, conn_type='opsgenie', host='https://api.opsgenie.com/', password='eb243592-faa2-4ba2-a551q-1afdf565c889'))

    def test_get_api_key(self):
        if False:
            print('Hello World!')
        hook = OpsgenieAlertHook(opsgenie_conn_id=self.conn_id)
        api_key = hook._get_api_key()
        assert 'eb243592-faa2-4ba2-a551q-1afdf565c889' == api_key

    def test_get_conn_defaults_host(self):
        if False:
            i = 10
            return i + 15
        hook = OpsgenieAlertHook()
        assert 'https://api.opsgenie.com' == hook.get_conn().api_client.configuration.host

    def test_get_conn_custom_host(self):
        if False:
            while True:
                i = 10
        conn_id = 'custom_host_opsgenie_test'
        db.merge_conn(Connection(conn_id=conn_id, conn_type='opsgenie', host='https://app.eu.opsgenie.com', password='eb243592-faa2-4ba2-a551q-1afdf565c889'))
        hook = OpsgenieAlertHook(conn_id)
        assert 'https://app.eu.opsgenie.com' == hook.get_conn().api_client.configuration.host

    def test_verify_api_key_set(self):
        if False:
            print('Hello World!')
        hook = OpsgenieAlertHook(opsgenie_conn_id=self.conn_id)
        assert hook.alert_api_instance.api_client.configuration.api_key.get('Authorization', None) == 'eb243592-faa2-4ba2-a551q-1afdf565c889'

    def test_create_alert_api_key_not_set(self):
        if False:
            while True:
                i = 10
        hook = OpsgenieAlertHook()
        with pytest.raises(AuthenticationException):
            hook.create_alert(payload=self._create_alert_payload)

    @mock.patch.object(AlertApi, 'create_alert')
    def test_create_alert_create_alert_payload(self, create_alert_mock):
        if False:
            print('Hello World!')
        hook = OpsgenieAlertHook(opsgenie_conn_id=self.conn_id)
        hook.create_alert(payload=self._create_alert_payload)
        create_alert_mock.assert_called_once_with(CreateAlertPayload(**self._create_alert_payload))

    @mock.patch.object(AlertApi, 'close_alert')
    def test_close_alert(self, close_alert_mock):
        if False:
            print('Hello World!')
        hook = OpsgenieAlertHook(opsgenie_conn_id=self.conn_id)
        pay_load = {'user': 'str', 'note': 'str', 'source': 'str'}
        identifier = 'identifier_example'
        identifier_type = 'id'
        kwargs = {'async_req': True}
        hook.close_alert(identifier=identifier, identifier_type=identifier_type, payload=pay_load, **kwargs)
        close_alert_mock.assert_called_once_with(identifier=identifier, identifier_type=identifier_type, close_alert_payload=CloseAlertPayload(**pay_load), **kwargs)

    @mock.patch.object(AlertApi, 'delete_alert')
    def test_delete_alert(self, delete_alert_mock):
        if False:
            while True:
                i = 10
        hook = OpsgenieAlertHook(opsgenie_conn_id=self.conn_id)
        identifier = 'identifier_example'
        identifier_type = 'id'
        user = 'some_user'
        source = 'airflow'
        hook.delete_alert(identifier=identifier, identifier_type=identifier_type, user=user, source=source)
        delete_alert_mock.assert_called_once_with(identifier=identifier, identifier_type=identifier_type, user=user, source=source)