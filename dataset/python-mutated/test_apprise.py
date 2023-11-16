from __future__ import annotations
from unittest import mock
import pytest
from apprise import NotifyType
from airflow.operators.empty import EmptyOperator
from airflow.providers.apprise.notifications.apprise import AppriseNotifier, send_apprise_notification
pytestmark = pytest.mark.db_test

class TestAppriseNotifier:

    @mock.patch('airflow.providers.apprise.notifications.apprise.AppriseHook')
    def test_notifier(self, mock_apprise_hook, dag_maker):
        if False:
            while True:
                i = 10
        with dag_maker('test_notifier') as dag:
            EmptyOperator(task_id='task1')
        notifier = send_apprise_notification(body='DISK at 99%', notify_type=NotifyType.FAILURE)
        notifier({'dag': dag})
        mock_apprise_hook.return_value.notify.assert_called_once_with(body='DISK at 99%', notify_type=NotifyType.FAILURE, title=None, body_format=None, tag=None, attach=None, interpret_escapes=None, config=None)

    @mock.patch('airflow.providers.apprise.notifications.apprise.AppriseHook')
    def test_notifier_with_notifier_class(self, mock_apprise_hook, dag_maker):
        if False:
            print('Hello World!')
        with dag_maker('test_notifier') as dag:
            EmptyOperator(task_id='task1')
        notifier = AppriseNotifier(body='DISK at 99%', notify_type=NotifyType.FAILURE)
        notifier({'dag': dag})
        mock_apprise_hook.return_value.notify.assert_called_once_with(body='DISK at 99%', notify_type=NotifyType.FAILURE, title=None, body_format=None, tag=None, attach=None, interpret_escapes=None, config=None)

    @mock.patch('airflow.providers.apprise.notifications.apprise.AppriseHook')
    def test_notifier_templated(self, mock_apprise_hook, dag_maker):
        if False:
            while True:
                i = 10
        with dag_maker('test_notifier') as dag:
            EmptyOperator(task_id='task1')
        notifier = AppriseNotifier(notify_type=NotifyType.FAILURE, title='DISK at 99% {{dag.dag_id}}', body='System can crash soon {{dag.dag_id}}')
        context = {'dag': dag}
        notifier(context)
        mock_apprise_hook.return_value.notify.assert_called_once_with(notify_type=NotifyType.FAILURE, title='DISK at 99% test_notifier', body='System can crash soon test_notifier', body_format=None, tag=None, attach=None, interpret_escapes=None, config=None)