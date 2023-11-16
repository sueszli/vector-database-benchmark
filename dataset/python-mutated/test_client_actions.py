from __future__ import absolute_import
import json
import logging
import mock
import unittest2
from tests import base
from st2client import client
from st2client.utils import httpclient
LOG = logging.getLogger(__name__)
EXECUTION = {'id': 12345, 'action': {'ref': 'mock.foobar'}, 'status': 'failed', 'result': 'non-empty'}
ENTRYPOINT = 'version: 1.0description: A basic workflow that runs an arbitrary linux command.input:  - cmd  - timeouttasks:  task1:    action: core.local cmd=<% ctx(cmd) %> timeout=<% ctx(timeout) %>    next:      - when: <% succeeded() %>        publish:          - stdout: <% result().stdout %>          - stderr: <% result().stderr %>output:  - stdout: <% ctx(stdout) %>'

class TestActionResourceManager(unittest2.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        super(TestActionResourceManager, cls).setUpClass()
        cls.client = client.Client()

    @mock.patch.object(httpclient.HTTPClient, 'get', mock.MagicMock(return_value=base.FakeResponse(json.dumps(ENTRYPOINT), 200, 'OK')))
    def test_get_action_entry_point_by_ref(self):
        if False:
            i = 10
            return i + 15
        actual_entrypoint = self.client.actions.get_entrypoint(EXECUTION['action']['ref'])
        actual_entrypoint = json.loads(actual_entrypoint)
        endpoint = '/actions/views/entry_point/%s' % EXECUTION['action']['ref']
        httpclient.HTTPClient.get.assert_called_with(endpoint)
        self.assertEqual(ENTRYPOINT, actual_entrypoint)

    @mock.patch.object(httpclient.HTTPClient, 'get', mock.MagicMock(return_value=base.FakeResponse(json.dumps(ENTRYPOINT), 200, 'OK')))
    def test_get_action_entry_point_by_id(self):
        if False:
            while True:
                i = 10
        actual_entrypoint = self.client.actions.get_entrypoint(EXECUTION['id'])
        actual_entrypoint = json.loads(actual_entrypoint)
        endpoint = '/actions/views/entry_point/%s' % EXECUTION['id']
        httpclient.HTTPClient.get.assert_called_with(endpoint)
        self.assertEqual(ENTRYPOINT, actual_entrypoint)

    @mock.patch.object(httpclient.HTTPClient, 'get', mock.MagicMock(return_value=base.FakeResponse(json.dumps({}), 404, '404 Client Error: Not Found')))
    def test_get_non_existent_action_entry_point(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegexp(Exception, '404 Client Error: Not Found'):
            self.client.actions.get_entrypoint('nonexistentpack.nonexistentaction')
        endpoint = '/actions/views/entry_point/%s' % 'nonexistentpack.nonexistentaction'
        httpclient.HTTPClient.get.assert_called_with(endpoint)