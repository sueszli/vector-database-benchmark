from __future__ import annotations
from unittest import mock
import pytest
from airflow.operators.empty import EmptyOperator
from airflow.providers.slack.notifications.slack_webhook import SlackWebhookNotifier, send_slack_webhook_notification
pytestmark = pytest.mark.db_test
DEFAULT_HOOKS_PARAMETERS = {'timeout': None, 'proxy': None, 'retry_handlers': None}

class TestSlackNotifier:

    def test_class_and_notifier_are_same(self):
        if False:
            print('Hello World!')
        assert send_slack_webhook_notification is SlackWebhookNotifier

    @mock.patch('airflow.providers.slack.notifications.slack_webhook.SlackWebhookHook')
    @pytest.mark.parametrize('slack_op_kwargs, hook_extra_kwargs', [pytest.param({}, DEFAULT_HOOKS_PARAMETERS, id='default-hook-parameters'), pytest.param({'timeout': 42, 'proxy': 'http://spam.egg', 'retry_handlers': []}, {'timeout': 42, 'proxy': 'http://spam.egg', 'retry_handlers': []}, id='with-extra-hook-parameters')])
    def test_slack_webhook_notifier(self, mock_slack_hook, slack_op_kwargs, hook_extra_kwargs):
        if False:
            i = 10
            return i + 15
        notifier = send_slack_webhook_notification(slack_webhook_conn_id='test_conn_id', text='foo-bar', blocks='spam-egg', attachments='baz-qux', unfurl_links=True, unfurl_media=False, **slack_op_kwargs)
        notifier.notify({})
        mock_slack_hook.return_value.send.assert_called_once_with(text='foo-bar', blocks='spam-egg', unfurl_links=True, unfurl_media=False, attachments='baz-qux')
        mock_slack_hook.assert_called_once_with(slack_webhook_conn_id='test_conn_id', **hook_extra_kwargs)

    @mock.patch('airflow.providers.slack.notifications.slack_webhook.SlackWebhookHook')
    def test_slack_webhook_templated(self, mock_slack_hook, dag_maker):
        if False:
            while True:
                i = 10
        with dag_maker('test_send_slack_webhook_notification_templated') as dag:
            EmptyOperator(task_id='task1')
        notifier = send_slack_webhook_notification(text='Who am I? {{ username }}', blocks=[{'type': 'header', 'text': {'type': 'plain_text', 'text': '{{ dag.dag_id }}'}}], attachments=[{'image_url': '{{ dag.dag_id }}.png'}])
        notifier({'dag': dag, 'username': 'not-a-root'})
        mock_slack_hook.return_value.send.assert_called_once_with(text='Who am I? not-a-root', blocks=[{'type': 'header', 'text': {'type': 'plain_text', 'text': 'test_send_slack_webhook_notification_templated'}}], attachments=[{'image_url': 'test_send_slack_webhook_notification_templated.png'}], unfurl_links=None, unfurl_media=None)