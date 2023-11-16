from __future__ import annotations
import json
import pytest
from airflow.exceptions import AirflowException
from airflow.models import Connection
from airflow.providers.discord.hooks.discord_webhook import DiscordWebhookHook
from airflow.utils import db
pytestmark = pytest.mark.db_test

class TestDiscordWebhookHook:
    _config = {'http_conn_id': 'default-discord-webhook', 'webhook_endpoint': 'webhooks/11111/some-discord-token_111', 'message': 'your message here', 'username': 'Airflow Webhook', 'avatar_url': 'https://static-cdn.avatars.com/my-avatar-path', 'tts': False, 'proxy': 'https://proxy.proxy.com:8888'}
    expected_payload_dict = {'username': _config['username'], 'avatar_url': _config['avatar_url'], 'tts': _config['tts'], 'content': _config['message']}
    expected_payload = json.dumps(expected_payload_dict)

    def setup_method(self):
        if False:
            return 10
        db.merge_conn(Connection(conn_id='default-discord-webhook', conn_type='discord', host='https://discordapp.com/api/', extra='{"webhook_endpoint": "webhooks/00000/some-discord-token_000"}'))

    def test_get_webhook_endpoint_manual_token(self):
        if False:
            return 10
        provided_endpoint = 'webhooks/11111/some-discord-token_111'
        hook = DiscordWebhookHook(webhook_endpoint=provided_endpoint)
        webhook_endpoint = hook._get_webhook_endpoint(None, provided_endpoint)
        assert webhook_endpoint == provided_endpoint

    def test_get_webhook_endpoint_invalid_url(self):
        if False:
            return 10
        provided_endpoint = 'https://discordapp.com/some-invalid-webhook-url'
        expected_message = 'Expected Discord webhook endpoint in the form of'
        with pytest.raises(AirflowException, match=expected_message):
            DiscordWebhookHook(webhook_endpoint=provided_endpoint)

    def test_get_webhook_endpoint_conn_id(self):
        if False:
            print('Hello World!')
        conn_id = 'default-discord-webhook'
        hook = DiscordWebhookHook(http_conn_id=conn_id)
        expected_webhook_endpoint = 'webhooks/00000/some-discord-token_000'
        webhook_endpoint = hook._get_webhook_endpoint(conn_id, None)
        assert webhook_endpoint == expected_webhook_endpoint

    def test_build_discord_payload(self):
        if False:
            for i in range(10):
                print('nop')
        hook = DiscordWebhookHook(**self._config)
        payload = hook._build_discord_payload()
        assert self.expected_payload == payload

    def test_build_discord_payload_message_length(self):
        if False:
            return 10
        config = self._config.copy()
        config['message'] = 'c' * 2001
        hook = DiscordWebhookHook(**config)
        expected_message = 'Discord message length must be 2000 or fewer characters'
        with pytest.raises(AirflowException, match=expected_message):
            hook._build_discord_payload()