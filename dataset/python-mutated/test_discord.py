from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
from airflow.models import Connection
from airflow.providers.discord.notifications.discord import DiscordNotifier
from airflow.utils import db
pytestmark = pytest.mark.db_test

@pytest.fixture(autouse=True)
def setup():
    if False:
        while True:
            i = 10
    db.merge_conn(Connection(conn_id='my_discord_conn_id', conn_type='discord', host='https://discordapp.com/api/', extra='{"webhook_endpoint": "webhooks/00000/some-discord-token_000"}'))

@patch('airflow.providers.discord.notifications.discord.DiscordWebhookHook.execute')
def test_discord_notifier_notify(mock_execute):
    if False:
        while True:
            i = 10
    notifier = DiscordNotifier(discord_conn_id='my_discord_conn_id', text='This is a test message', username='test_user', avatar_url='https://example.com/avatar.png', tts=False)
    context = MagicMock()
    notifier.notify(context)
    mock_execute.assert_called_once()
    assert notifier.hook.username == 'test_user'
    assert notifier.hook.message == 'This is a test message'
    assert notifier.hook.avatar_url == 'https://example.com/avatar.png'
    assert notifier.hook.tts is False