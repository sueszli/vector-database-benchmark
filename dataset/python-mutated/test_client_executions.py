from __future__ import absolute_import
import json
import logging
import warnings
import mock
import unittest2
from tests import base
from st2client import client
from st2client import models
from st2client.utils import httpclient
LOG = logging.getLogger(__name__)
RUNNER = {'enabled': True, 'name': 'marathon', 'runner_parameters': {'var1': {'type': 'string'}}}
ACTION = {'ref': 'mock.foobar', 'runner_type': 'marathon', 'name': 'foobar', 'parameters': {}, 'enabled': True, 'entry_point': '', 'pack': 'mocke'}
EXECUTION = {'id': 12345, 'action': {'ref': 'mock.foobar'}, 'status': 'failed', 'result': 'non-empty'}

class TestExecutionResourceManager(unittest2.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        super(TestExecutionResourceManager, cls).setUpClass()
        cls.client = client.Client()

    @mock.patch.object(models.ResourceManager, 'get_by_id', mock.MagicMock(return_value=models.Execution(**EXECUTION)))
    @mock.patch.object(models.ResourceManager, 'get_by_ref_or_id', mock.MagicMock(return_value=models.Action(**ACTION)))
    @mock.patch.object(models.ResourceManager, 'get_by_name', mock.MagicMock(return_value=models.RunnerType(**RUNNER)))
    @mock.patch.object(httpclient.HTTPClient, 'post', mock.MagicMock(return_value=base.FakeResponse(json.dumps(EXECUTION), 200, 'OK')))
    def test_rerun_with_no_params(self):
        if False:
            while True:
                i = 10
        self.client.executions.re_run(EXECUTION['id'], tasks=['foobar'])
        endpoint = '/executions/%s/re_run' % EXECUTION['id']
        data = {'tasks': ['foobar'], 'reset': ['foobar'], 'parameters': {}, 'delay': 0}
        httpclient.HTTPClient.post.assert_called_with(endpoint, data)

    @mock.patch.object(models.ResourceManager, 'get_by_id', mock.MagicMock(return_value=models.Execution(**EXECUTION)))
    @mock.patch.object(models.ResourceManager, 'get_by_ref_or_id', mock.MagicMock(return_value=models.Action(**ACTION)))
    @mock.patch.object(models.ResourceManager, 'get_by_name', mock.MagicMock(return_value=models.RunnerType(**RUNNER)))
    @mock.patch.object(httpclient.HTTPClient, 'post', mock.MagicMock(return_value=base.FakeResponse(json.dumps(EXECUTION), 200, 'OK')))
    def test_rerun_with_params(self):
        if False:
            return 10
        params = {'var1': 'testing...'}
        self.client.executions.re_run(EXECUTION['id'], tasks=['foobar'], parameters=params)
        endpoint = '/executions/%s/re_run' % EXECUTION['id']
        data = {'tasks': ['foobar'], 'reset': ['foobar'], 'parameters': params, 'delay': 0}
        httpclient.HTTPClient.post.assert_called_with(endpoint, data)

    @mock.patch.object(models.ResourceManager, 'get_by_id', mock.MagicMock(return_value=models.Execution(**EXECUTION)))
    @mock.patch.object(models.ResourceManager, 'get_by_ref_or_id', mock.MagicMock(return_value=models.Action(**ACTION)))
    @mock.patch.object(models.ResourceManager, 'get_by_name', mock.MagicMock(return_value=models.RunnerType(**RUNNER)))
    @mock.patch.object(httpclient.HTTPClient, 'post', mock.MagicMock(return_value=base.FakeResponse(json.dumps(EXECUTION), 200, 'OK')))
    def test_rerun_with_delay(self):
        if False:
            i = 10
            return i + 15
        self.client.executions.re_run(EXECUTION['id'], tasks=['foobar'], delay=100)
        endpoint = '/executions/%s/re_run' % EXECUTION['id']
        data = {'tasks': ['foobar'], 'reset': ['foobar'], 'parameters': {}, 'delay': 100}
        httpclient.HTTPClient.post.assert_called_with(endpoint, data)

    @mock.patch.object(models.ResourceManager, 'get_by_id', mock.MagicMock(return_value=models.Execution(**EXECUTION)))
    @mock.patch.object(models.ResourceManager, 'get_by_ref_or_id', mock.MagicMock(return_value=models.Action(**ACTION)))
    @mock.patch.object(models.ResourceManager, 'get_by_name', mock.MagicMock(return_value=models.RunnerType(**RUNNER)))
    @mock.patch.object(httpclient.HTTPClient, 'put', mock.MagicMock(return_value=base.FakeResponse(json.dumps(EXECUTION), 200, 'OK')))
    def test_pause(self):
        if False:
            i = 10
            return i + 15
        self.client.executions.pause(EXECUTION['id'])
        endpoint = '/executions/%s' % EXECUTION['id']
        data = {'status': 'pausing'}
        httpclient.HTTPClient.put.assert_called_with(endpoint, data)

    @mock.patch.object(models.ResourceManager, 'get_by_id', mock.MagicMock(return_value=models.Execution(**EXECUTION)))
    @mock.patch.object(models.ResourceManager, 'get_by_ref_or_id', mock.MagicMock(return_value=models.Action(**ACTION)))
    @mock.patch.object(models.ResourceManager, 'get_by_name', mock.MagicMock(return_value=models.RunnerType(**RUNNER)))
    @mock.patch.object(httpclient.HTTPClient, 'put', mock.MagicMock(return_value=base.FakeResponse(json.dumps(EXECUTION), 200, 'OK')))
    def test_resume(self):
        if False:
            while True:
                i = 10
        self.client.executions.resume(EXECUTION['id'])
        endpoint = '/executions/%s' % EXECUTION['id']
        data = {'status': 'resuming'}
        httpclient.HTTPClient.put.assert_called_with(endpoint, data)

    @mock.patch.object(models.core.Resource, 'get_url_path_name', mock.MagicMock(return_value='executions'))
    @mock.patch.object(httpclient.HTTPClient, 'get', mock.MagicMock(return_value=base.FakeResponse(json.dumps([EXECUTION]), 200, 'OK')))
    def test_get_children(self):
        if False:
            return 10
        self.client.executions.get_children(EXECUTION['id'])
        endpoint = '/executions/%s/children' % EXECUTION['id']
        data = {'depth': -1}
        httpclient.HTTPClient.get.assert_called_with(url=endpoint, params=data)

    @mock.patch.object(models.ResourceManager, 'get_all', mock.MagicMock(return_value=[models.Execution(**EXECUTION)]))
    @mock.patch.object(warnings, 'warn')
    def test_st2client_liveactions_has_been_deprecated_and_emits_warning(self, mock_warn):
        if False:
            print('Hello World!')
        self.assertEqual(mock_warn.call_args, None)
        self.client.liveactions.get_all()
        expected_msg = 'st2client.liveactions has been renamed'
        self.assertTrue(len(mock_warn.call_args_list) >= 1)
        self.assertIn(expected_msg, mock_warn.call_args_list[0][0][0])
        self.assertEqual(mock_warn.call_args_list[0][0][1], DeprecationWarning)