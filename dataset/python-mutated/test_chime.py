from __future__ import annotations
import json
import pytest
from airflow.exceptions import AirflowException
from airflow.models import Connection
from airflow.providers.amazon.aws.hooks.chime import ChimeWebhookHook
from airflow.utils import db
pytestmark = pytest.mark.db_test

class TestChimeWebhookHook:
    _config = {'chime_conn_id': 'default-chime-webhook', 'webhook_endpoint': 'incomingwebhooks/abcd-1134-ZeDA?token=somechimetoken-111', 'message': 'your message here'}
    expected_payload_dict = {'Content': _config['message']}
    expected_payload = json.dumps(expected_payload_dict)

    def setup_method(self):
        if False:
            while True:
                i = 10
        db.merge_conn(Connection(conn_id='default-chime-webhook', conn_type='chime', host='hooks.chime.aws/incomingwebhooks/', password='abcd-1134-ZeDA?token=somechimetoken111', schema='https'))
        db.merge_conn(Connection(conn_id='chime-bad-url', conn_type='chime', host='https://hooks.chime.aws/', password='somebadurl', schema='https'))

    def test_get_webhook_endpoint_invalid_url(self):
        if False:
            return 10
        expected_message = 'Expected Chime webhook token in the form'
        hook = ChimeWebhookHook(chime_conn_id='chime-bad-url')
        with pytest.raises(AirflowException, match=expected_message):
            assert not hook.webhook_endpoint

    def test_get_webhook_endpoint_conn_id(self):
        if False:
            return 10
        conn_id = 'default-chime-webhook'
        hook = ChimeWebhookHook(chime_conn_id=conn_id)
        expected_webhook_endpoint = 'https://hooks.chime.aws/incomingwebhooks/abcd-1134-ZeDA?token=somechimetoken111'
        webhook_endpoint = hook._get_webhook_endpoint(conn_id)
        assert webhook_endpoint == expected_webhook_endpoint

    def test_build_chime_payload(self):
        if False:
            while True:
                i = 10
        hook = ChimeWebhookHook(self._config['chime_conn_id'])
        message = self._config['message']
        payload = hook._build_chime_payload(message)
        assert self.expected_payload == payload

    def test_build_chime_payload_message_length(self):
        if False:
            print('Hello World!')
        self._config.copy()
        message = 'c' * 4097
        hook = ChimeWebhookHook(self._config['chime_conn_id'])
        expected_message = 'Chime message must be 4096 characters or less.'
        with pytest.raises(AirflowException, match=expected_message):
            hook._build_chime_payload(message)