from __future__ import annotations
import json
from unittest import mock
from google.api_core.gapic_v1.method import DEFAULT
from google.cloud.monitoring_v3 import AlertPolicy, NotificationChannel
from airflow.providers.google.cloud.operators.stackdriver import StackdriverDeleteAlertOperator, StackdriverDeleteNotificationChannelOperator, StackdriverDisableAlertPoliciesOperator, StackdriverDisableNotificationChannelsOperator, StackdriverEnableAlertPoliciesOperator, StackdriverEnableNotificationChannelsOperator, StackdriverListAlertPoliciesOperator, StackdriverListNotificationChannelsOperator, StackdriverUpsertAlertOperator, StackdriverUpsertNotificationChannelOperator
TEST_TASK_ID = 'test-stackdriver-operator'
TEST_FILTER = 'filter'
TEST_ALERT_POLICY_1 = {'combiner': 'OR', 'name': 'projects/sd-project/alertPolicies/12345', 'enabled': True, 'display_name': 'test display', 'conditions': [{'condition_threshold': {'comparison': 'COMPARISON_GT', 'aggregations': [{'alignment_eriod': {'seconds': 60}, 'per_series_aligner': 'ALIGN_RATE'}]}, 'display_name': 'Condition display', 'name': 'projects/sd-project/alertPolicies/123/conditions/456'}]}
TEST_ALERT_POLICY_2 = {'combiner': 'OR', 'name': 'projects/sd-project/alertPolicies/6789', 'enabled': False, 'display_name': 'test display', 'conditions': [{'condition_threshold': {'comparison': 'COMPARISON_GT', 'aggregations': [{'alignment_period': {'seconds': 60}, 'per_series_aligner': 'ALIGN_RATE'}]}, 'display_name': 'Condition display', 'name': 'projects/sd-project/alertPolicies/456/conditions/789'}]}
TEST_NOTIFICATION_CHANNEL_1 = {'displayName': 'sd', 'enabled': True, 'labels': {'auth_token': 'top-secret', 'channel_name': '#channel'}, 'name': 'projects/sd-project/notificationChannels/12345', 'type': 'slack'}
TEST_NOTIFICATION_CHANNEL_2 = {'displayName': 'sd', 'enabled': False, 'labels': {'auth_token': 'top-secret', 'channel_name': '#channel'}, 'name': 'projects/sd-project/notificationChannels/6789', 'type': 'slack'}

class TestStackdriverListAlertPoliciesOperator:

    @mock.patch('airflow.providers.google.cloud.operators.stackdriver.StackdriverHook')
    def test_execute(self, mock_hook):
        if False:
            for i in range(10):
                print('nop')
        operator = StackdriverListAlertPoliciesOperator(task_id=TEST_TASK_ID, filter_=TEST_FILTER)
        mock_hook.return_value.list_alert_policies.return_value = [AlertPolicy(name='test-name')]
        result = operator.execute(context=mock.MagicMock())
        mock_hook.return_value.list_alert_policies.assert_called_once_with(project_id=None, filter_=TEST_FILTER, format_=None, order_by=None, page_size=None, retry=DEFAULT, timeout=None, metadata=())
        assert [{'combiner': 0, 'conditions': [], 'display_name': '', 'name': 'test-name', 'notification_channels': [], 'user_labels': {}}] == result

class TestStackdriverEnableAlertPoliciesOperator:

    @mock.patch('airflow.providers.google.cloud.operators.stackdriver.StackdriverHook')
    def test_execute(self, mock_hook):
        if False:
            for i in range(10):
                print('nop')
        operator = StackdriverEnableAlertPoliciesOperator(task_id=TEST_TASK_ID, filter_=TEST_FILTER)
        operator.execute(context=mock.MagicMock())
        mock_hook.return_value.enable_alert_policies.assert_called_once_with(project_id=None, filter_=TEST_FILTER, retry=DEFAULT, timeout=None, metadata=())

class TestStackdriverDisableAlertPoliciesOperator:

    @mock.patch('airflow.providers.google.cloud.operators.stackdriver.StackdriverHook')
    def test_execute(self, mock_hook):
        if False:
            print('Hello World!')
        operator = StackdriverDisableAlertPoliciesOperator(task_id=TEST_TASK_ID, filter_=TEST_FILTER)
        operator.execute(context=mock.MagicMock())
        mock_hook.return_value.disable_alert_policies.assert_called_once_with(project_id=None, filter_=TEST_FILTER, retry=DEFAULT, timeout=None, metadata=())

class TestStackdriverUpsertAlertsOperator:

    @mock.patch('airflow.providers.google.cloud.operators.stackdriver.StackdriverHook')
    def test_execute(self, mock_hook):
        if False:
            while True:
                i = 10
        operator = StackdriverUpsertAlertOperator(task_id=TEST_TASK_ID, alerts=json.dumps({'policies': [TEST_ALERT_POLICY_1, TEST_ALERT_POLICY_2]}))
        operator.execute(context=mock.MagicMock())
        mock_hook.return_value.upsert_alert.assert_called_once_with(alerts=json.dumps({'policies': [TEST_ALERT_POLICY_1, TEST_ALERT_POLICY_2]}), project_id=None, retry=DEFAULT, timeout=None, metadata=())

class TestStackdriverDeleteAlertOperator:

    @mock.patch('airflow.providers.google.cloud.operators.stackdriver.StackdriverHook')
    def test_execute(self, mock_hook):
        if False:
            i = 10
            return i + 15
        operator = StackdriverDeleteAlertOperator(task_id=TEST_TASK_ID, name='test-alert')
        operator.execute(context=mock.MagicMock())
        mock_hook.return_value.delete_alert_policy.assert_called_once_with(name='test-alert', retry=DEFAULT, timeout=None, metadata=())

class TestStackdriverListNotificationChannelsOperator:

    @mock.patch('airflow.providers.google.cloud.operators.stackdriver.StackdriverHook')
    def test_execute(self, mock_hook):
        if False:
            i = 10
            return i + 15
        operator = StackdriverListNotificationChannelsOperator(task_id=TEST_TASK_ID, filter_=TEST_FILTER)
        mock_hook.return_value.list_notification_channels.return_value = [NotificationChannel(name='test-123')]
        result = operator.execute(context=mock.MagicMock())
        mock_hook.return_value.list_notification_channels.assert_called_once_with(project_id=None, filter_=TEST_FILTER, format_=None, order_by=None, page_size=None, retry=DEFAULT, timeout=None, metadata=())
        assert result in [[{'description': '', 'display_name': '', 'labels': {}, 'name': 'test-123', 'type_': '', 'user_labels': {}, 'verification_status': 0}], [{'description': '', 'display_name': '', 'labels': {}, 'mutation_records': [], 'name': 'test-123', 'type_': '', 'user_labels': {}, 'verification_status': 0}]]

class TestStackdriverEnableNotificationChannelsOperator:

    @mock.patch('airflow.providers.google.cloud.operators.stackdriver.StackdriverHook')
    def test_execute(self, mock_hook):
        if False:
            for i in range(10):
                print('nop')
        operator = StackdriverEnableNotificationChannelsOperator(task_id=TEST_TASK_ID, filter_=TEST_FILTER)
        operator.execute(context=mock.MagicMock())
        mock_hook.return_value.enable_notification_channels.assert_called_once_with(project_id=None, filter_=TEST_FILTER, retry=DEFAULT, timeout=None, metadata=())

class TestStackdriverDisableNotificationChannelsOperator:

    @mock.patch('airflow.providers.google.cloud.operators.stackdriver.StackdriverHook')
    def test_execute(self, mock_hook):
        if False:
            i = 10
            return i + 15
        operator = StackdriverDisableNotificationChannelsOperator(task_id=TEST_TASK_ID, filter_=TEST_FILTER)
        operator.execute(context=mock.MagicMock())
        mock_hook.return_value.disable_notification_channels.assert_called_once_with(project_id=None, filter_=TEST_FILTER, retry=DEFAULT, timeout=None, metadata=())

class TestStackdriverUpsertChannelOperator:

    @mock.patch('airflow.providers.google.cloud.operators.stackdriver.StackdriverHook')
    def test_execute(self, mock_hook):
        if False:
            while True:
                i = 10
        operator = StackdriverUpsertNotificationChannelOperator(task_id=TEST_TASK_ID, channels=json.dumps({'channels': [TEST_NOTIFICATION_CHANNEL_1, TEST_NOTIFICATION_CHANNEL_2]}))
        operator.execute(context=mock.MagicMock())
        mock_hook.return_value.upsert_channel.assert_called_once_with(channels=json.dumps({'channels': [TEST_NOTIFICATION_CHANNEL_1, TEST_NOTIFICATION_CHANNEL_2]}), project_id=None, retry=DEFAULT, timeout=None, metadata=())

class TestStackdriverDeleteNotificationChannelOperator:

    @mock.patch('airflow.providers.google.cloud.operators.stackdriver.StackdriverHook')
    def test_execute(self, mock_hook):
        if False:
            print('Hello World!')
        operator = StackdriverDeleteNotificationChannelOperator(task_id=TEST_TASK_ID, name='test-channel')
        operator.execute(context=mock.MagicMock())
        mock_hook.return_value.delete_notification_channel.assert_called_once_with(name='test-channel', retry=DEFAULT, timeout=None, metadata=())