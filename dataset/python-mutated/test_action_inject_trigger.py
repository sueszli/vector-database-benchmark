from __future__ import absolute_import
import mock
from st2tests.base import BaseActionTestCase
from inject_trigger import InjectTriggerAction

class InjectTriggerActionTestCase(BaseActionTestCase):
    action_cls = InjectTriggerAction

    @mock.patch('st2common.services.datastore.BaseDatastoreService.get_api_client')
    def test_inject_trigger_only_trigger_no_payload(self, mock_get_api_client):
        if False:
            for i in range(10):
                print('nop')
        mock_api_client = mock.Mock()
        mock_get_api_client.return_value = mock_api_client
        action = self.get_action_instance()
        action.run(trigger='dummy_pack.trigger1')
        mock_api_client.webhooks.post_generic_webhook.assert_called_with(trigger='dummy_pack.trigger1', payload={}, trace_tag=None)
        mock_api_client.webhooks.post_generic_webhook.reset()

    @mock.patch('st2common.services.datastore.BaseDatastoreService.get_api_client')
    def test_inject_trigger_trigger_and_payload(self, mock_get_api_client):
        if False:
            i = 10
            return i + 15
        mock_api_client = mock.Mock()
        mock_get_api_client.return_value = mock_api_client
        action = self.get_action_instance()
        action.run(trigger='dummy_pack.trigger2', payload={'foo': 'bar'})
        mock_api_client.webhooks.post_generic_webhook.assert_called_with(trigger='dummy_pack.trigger2', payload={'foo': 'bar'}, trace_tag=None)
        mock_api_client.webhooks.post_generic_webhook.reset()

    @mock.patch('st2common.services.datastore.BaseDatastoreService.get_api_client')
    def test_inject_trigger_trigger_payload_trace_tag(self, mock_get_api_client):
        if False:
            print('Hello World!')
        mock_api_client = mock.Mock()
        mock_get_api_client.return_value = mock_api_client
        action = self.get_action_instance()
        action.run(trigger='dummy_pack.trigger3', payload={'foo': 'bar'}, trace_tag='Tag1')
        mock_api_client.webhooks.post_generic_webhook.assert_called_with(trigger='dummy_pack.trigger3', payload={'foo': 'bar'}, trace_tag='Tag1')