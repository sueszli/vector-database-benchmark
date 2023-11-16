from __future__ import absolute_import
import mock
import json
from tests import base
from tests.base import BaseCLITestCase
from st2client.utils import httpclient
from st2client.commands.action import LIVEACTION_STATUS_RUNNING
from st2client.commands.action import LIVEACTION_STATUS_SUCCEEDED
from st2client.commands.action import LIVEACTION_STATUS_FAILED
from st2client.commands.action import LIVEACTION_STATUS_TIMED_OUT
from st2client.shell import Shell
__all__ = ['ActionExecutionTailCommandTestCase']
MOCK_LIVEACTION_1_RUNNING = {'id': 'idfoo1', 'status': LIVEACTION_STATUS_RUNNING}
MOCK_LIVEACTION_1_SUCCEEDED = {'id': 'idfoo1', 'status': LIVEACTION_STATUS_SUCCEEDED}
MOCK_LIVEACTION_2_FAILED = {'id': 'idfoo2', 'status': LIVEACTION_STATUS_FAILED}
MOCK_LIVEACTION_3_RUNNING = {'id': 'idfoo3', 'status': LIVEACTION_STATUS_RUNNING}
MOCK_LIVEACTION_3_CHILD_1_RUNNING = {'id': 'idchild1', 'context': {'parent': {'execution_id': 'idfoo3'}, 'chain': {'name': 'task_1'}}, 'status': LIVEACTION_STATUS_RUNNING}
MOCK_LIVEACTION_3_CHILD_1_SUCCEEDED = {'id': 'idchild1', 'context': {'parent': {'execution_id': 'idfoo3'}, 'chain': {'name': 'task_1'}}, 'status': LIVEACTION_STATUS_SUCCEEDED}
MOCK_LIVEACTION_3_CHILD_1_OUTPUT_1 = {'execution_id': 'idchild1', 'timestamp': '1505732598', 'output_type': 'stdout', 'data': 'line ac 4\n'}
MOCK_LIVEACTION_3_CHILD_1_OUTPUT_2 = {'execution_id': 'idchild1', 'timestamp': '1505732598', 'output_type': 'stderr', 'data': 'line ac 5\n'}
MOCK_LIVEACTION_3_CHILD_2_RUNNING = {'id': 'idchild2', 'context': {'parent': {'execution_id': 'idfoo3'}, 'chain': {'name': 'task_2'}}, 'status': LIVEACTION_STATUS_RUNNING}
MOCK_LIVEACTION_3_CHILD_2_FAILED = {'id': 'idchild2', 'context': {'parent': {'execution_id': 'idfoo3'}, 'chain': {'name': 'task_2'}}, 'status': LIVEACTION_STATUS_FAILED}
MOCK_LIVEACTION_3_CHILD_2_OUTPUT_1 = {'execution_id': 'idchild2', 'timestamp': '1505732598', 'output_type': 'stdout', 'data': 'line ac 100\n'}
MOCK_LIVEACTION_3_SUCCEDED = {'id': 'idfoo3', 'status': LIVEACTION_STATUS_SUCCEEDED}
MOCK_LIVEACTION_4_RUNNING = {'id': 'idfoo4', 'status': LIVEACTION_STATUS_RUNNING}
MOCK_LIVEACTION_4_CHILD_1_RUNNING = {'id': 'idorquestachild1', 'context': {'orquesta': {'task_name': 'task_1'}, 'parent': {'execution_id': 'idfoo4'}}, 'status': LIVEACTION_STATUS_RUNNING}
MOCK_LIVEACTION_4_CHILD_1_1_RUNNING = {'id': 'idorquestachild1_1', 'context': {'orquesta': {'task_name': 'task_1'}, 'parent': {'execution_id': 'idorquestachild1'}}, 'status': LIVEACTION_STATUS_RUNNING}
MOCK_LIVEACTION_4_CHILD_1_SUCCEEDED = {'id': 'idorquestachild1', 'context': {'orquesta': {'task_name': 'task_1'}, 'parent': {'execution_id': 'idfoo4'}}, 'status': LIVEACTION_STATUS_SUCCEEDED}
MOCK_LIVEACTION_4_CHILD_1_1_SUCCEEDED = {'id': 'idorquestachild1_1', 'context': {'orquesta': {'task_name': 'task_1'}, 'parent': {'execution_id': 'idorquestachild1'}}, 'status': LIVEACTION_STATUS_SUCCEEDED}
MOCK_LIVEACTION_4_CHILD_1_OUTPUT_1 = {'execution_id': 'idorquestachild1', 'timestamp': '1505732598', 'output_type': 'stdout', 'data': 'line orquesta 4\n'}
MOCK_LIVEACTION_4_CHILD_1_OUTPUT_2 = {'execution_id': 'idorquestachild1', 'timestamp': '1505732598', 'output_type': 'stderr', 'data': 'line orquesta 5\n'}
MOCK_LIVEACTION_4_CHILD_1_1_OUTPUT_1 = {'execution_id': 'idorquestachild1_1', 'timestamp': '1505732598', 'output_type': 'stdout', 'data': 'line orquesta 4\n'}
MOCK_LIVEACTION_4_CHILD_1_1_OUTPUT_2 = {'execution_id': 'idorquestachild1_1', 'timestamp': '1505732598', 'output_type': 'stderr', 'data': 'line orquesta 5\n'}
MOCK_LIVEACTION_4_CHILD_2_RUNNING = {'id': 'idorquestachild2', 'context': {'orquesta': {'task_name': 'task_2'}, 'parent': {'execution_id': 'idfoo4'}}, 'status': LIVEACTION_STATUS_RUNNING}
MOCK_LIVEACTION_4_CHILD_2_TIMED_OUT = {'id': 'idorquestachild2', 'context': {'orquesta': {'task_name': 'task_2'}, 'parent': {'execution_id': 'idfoo4'}}, 'status': LIVEACTION_STATUS_TIMED_OUT}
MOCK_LIVEACTION_4_CHILD_2_OUTPUT_1 = {'execution_id': 'idorquestachild2', 'timestamp': '1505732598', 'output_type': 'stdout', 'data': 'line orquesta 100\n'}
MOCK_LIVEACTION_4_SUCCEDED = {'id': 'idfoo4', 'status': LIVEACTION_STATUS_SUCCEEDED}
MOCK_OUTPUT_1 = {'execution_id': 'idfoo3', 'timestamp': '1505732598', 'output_type': 'stdout', 'data': 'line 1\n'}
MOCK_OUTPUT_2 = {'execution_id': 'idfoo3', 'timestamp': '1505732598', 'output_type': 'stderr', 'data': 'line 2\n'}

class ActionExecutionTailCommandTestCase(BaseCLITestCase):
    capture_output = True

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(ActionExecutionTailCommandTestCase, self).__init__(*args, **kwargs)
        self.shell = Shell()

    @mock.patch.object(httpclient.HTTPClient, 'get', mock.MagicMock(return_value=base.FakeResponse(json.dumps(MOCK_LIVEACTION_1_SUCCEEDED), 200, 'OK')))
    def test_tail_simple_execution_already_finished_succeeded(self):
        if False:
            return 10
        argv = ['execution', 'tail', 'idfoo1']
        self.assertEqual(self.shell.run(argv), 0)
        stdout = self.stdout.getvalue()
        stderr = self.stderr.getvalue()
        self.assertIn('Execution idfoo1 has completed (status=succeeded)', stdout)
        self.assertEqual(stderr, '')

    @mock.patch.object(httpclient.HTTPClient, 'get', mock.MagicMock(return_value=base.FakeResponse(json.dumps(MOCK_LIVEACTION_2_FAILED), 200, 'OK')))
    def test_tail_simple_execution_already_finished_failed(self):
        if False:
            for i in range(10):
                print('nop')
        argv = ['execution', 'tail', 'idfoo2']
        self.assertEqual(self.shell.run(argv), 0)
        stdout = self.stdout.getvalue()
        stderr = self.stderr.getvalue()
        self.assertIn('Execution idfoo2 has completed (status=failed)', stdout)
        self.assertEqual(stderr, '')

    @mock.patch.object(httpclient.HTTPClient, 'get', mock.MagicMock(return_value=base.FakeResponse(json.dumps(MOCK_LIVEACTION_1_RUNNING), 200, 'OK')))
    @mock.patch('st2client.client.StreamManager', autospec=True)
    def test_tail_simple_execution_running_no_data_produced(self, mock_stream_manager):
        if False:
            for i in range(10):
                print('nop')
        argv = ['execution', 'tail', 'idfoo1']
        MOCK_EVENTS = [MOCK_LIVEACTION_1_SUCCEEDED]
        mock_cls = mock.Mock()
        mock_cls.listen = mock.Mock()
        mock_listen_generator = mock.Mock()
        mock_listen_generator.return_value = MOCK_EVENTS
        mock_cls.listen.side_effect = mock_listen_generator
        mock_stream_manager.return_value = mock_cls
        self.assertEqual(self.shell.run(argv), 0)
        self.assertEqual(mock_listen_generator.call_count, 1)
        stdout = self.stdout.getvalue()
        stderr = self.stderr.getvalue()
        expected_result = '\nExecution idfoo1 has completed (status=succeeded).\n'
        self.assertEqual(stdout, expected_result)
        self.assertEqual(stderr, '')

    @mock.patch.object(httpclient.HTTPClient, 'get', mock.MagicMock(return_value=base.FakeResponse(json.dumps(MOCK_LIVEACTION_3_RUNNING), 200, 'OK')))
    @mock.patch('st2client.client.StreamManager', autospec=True)
    def test_tail_simple_execution_running_with_data(self, mock_stream_manager):
        if False:
            return 10
        argv = ['execution', 'tail', 'idfoo3']
        MOCK_EVENTS = [MOCK_LIVEACTION_3_RUNNING, MOCK_OUTPUT_1, MOCK_OUTPUT_2, MOCK_LIVEACTION_3_SUCCEDED]
        mock_cls = mock.Mock()
        mock_cls.listen = mock.Mock()
        mock_listen_generator = mock.Mock()
        mock_listen_generator.return_value = MOCK_EVENTS
        mock_cls.listen.side_effect = mock_listen_generator
        mock_stream_manager.return_value = mock_cls
        self.assertEqual(self.shell.run(argv), 0)
        self.assertEqual(mock_listen_generator.call_count, 1)
        stdout = self.stdout.getvalue()
        stderr = self.stderr.getvalue()
        expected_result = '\nExecution idfoo3 has started.\n\nline 1\nline 2\n\nExecution idfoo3 has completed (status=succeeded).\n'.lstrip()
        self.assertEqual(stdout, expected_result)
        self.assertEqual(stderr, '')

    @mock.patch.object(httpclient.HTTPClient, 'get', mock.MagicMock(return_value=base.FakeResponse(json.dumps(MOCK_LIVEACTION_3_RUNNING), 200, 'OK')))
    @mock.patch('st2client.client.StreamManager', autospec=True)
    def test_tail_action_chain_workflow_execution(self, mock_stream_manager):
        if False:
            i = 10
            return i + 15
        argv = ['execution', 'tail', 'idfoo3']
        MOCK_EVENTS = [MOCK_LIVEACTION_3_RUNNING, MOCK_LIVEACTION_3_CHILD_1_RUNNING, MOCK_LIVEACTION_3_CHILD_1_OUTPUT_1, MOCK_LIVEACTION_3_CHILD_1_OUTPUT_2, MOCK_LIVEACTION_3_CHILD_1_SUCCEEDED, MOCK_LIVEACTION_3_CHILD_2_RUNNING, MOCK_LIVEACTION_3_CHILD_2_OUTPUT_1, MOCK_LIVEACTION_3_CHILD_2_FAILED, MOCK_LIVEACTION_3_SUCCEDED]
        mock_cls = mock.Mock()
        mock_cls.listen = mock.Mock()
        mock_listen_generator = mock.Mock()
        mock_listen_generator.return_value = MOCK_EVENTS
        mock_cls.listen.side_effect = mock_listen_generator
        mock_stream_manager.return_value = mock_cls
        self.assertEqual(self.shell.run(argv), 0)
        self.assertEqual(mock_listen_generator.call_count, 1)
        stdout = self.stdout.getvalue()
        stderr = self.stderr.getvalue()
        expected_result = '\nExecution idfoo3 has started.\n\nChild execution (task=task_1) idchild1 has started.\n\nline ac 4\nline ac 5\n\nChild execution (task=task_1) idchild1 has finished (status=succeeded).\nChild execution (task=task_2) idchild2 has started.\n\nline ac 100\n\nChild execution (task=task_2) idchild2 has finished (status=failed).\n\nExecution idfoo3 has completed (status=succeeded).\n'.lstrip()
        self.assertEqual(stdout, expected_result)
        self.assertEqual(stderr, '')

    @mock.patch.object(httpclient.HTTPClient, 'get', mock.MagicMock(return_value=base.FakeResponse(json.dumps(MOCK_LIVEACTION_4_RUNNING), 200, 'OK')))
    @mock.patch('st2client.client.StreamManager', autospec=True)
    def test_tail_orquesta_workflow_execution(self, mock_stream_manager):
        if False:
            while True:
                i = 10
        argv = ['execution', 'tail', 'idfoo4']
        MOCK_EVENTS = [MOCK_LIVEACTION_4_RUNNING, MOCK_LIVEACTION_4_CHILD_1_RUNNING, MOCK_LIVEACTION_4_CHILD_1_OUTPUT_1, MOCK_LIVEACTION_4_CHILD_1_OUTPUT_2, MOCK_LIVEACTION_4_CHILD_1_SUCCEEDED, MOCK_LIVEACTION_4_CHILD_2_RUNNING, MOCK_LIVEACTION_4_CHILD_2_OUTPUT_1, MOCK_LIVEACTION_4_CHILD_2_TIMED_OUT, MOCK_LIVEACTION_4_SUCCEDED]
        mock_cls = mock.Mock()
        mock_cls.listen = mock.Mock()
        mock_listen_generator = mock.Mock()
        mock_listen_generator.return_value = MOCK_EVENTS
        mock_cls.listen.side_effect = mock_listen_generator
        mock_stream_manager.return_value = mock_cls
        self.assertEqual(self.shell.run(argv), 0)
        self.assertEqual(mock_listen_generator.call_count, 1)
        stdout = self.stdout.getvalue()
        stderr = self.stderr.getvalue()
        expected_result = '\nExecution idfoo4 has started.\n\nChild execution (task=task_1) idorquestachild1 has started.\n\nline orquesta 4\nline orquesta 5\n\nChild execution (task=task_1) idorquestachild1 has finished (status=succeeded).\nChild execution (task=task_2) idorquestachild2 has started.\n\nline orquesta 100\n\nChild execution (task=task_2) idorquestachild2 has finished (status=timeout).\n\nExecution idfoo4 has completed (status=succeeded).\n'.lstrip()
        self.assertEqual(stdout, expected_result)
        self.assertEqual(stderr, '')

    @mock.patch.object(httpclient.HTTPClient, 'get', mock.MagicMock(return_value=base.FakeResponse(json.dumps(MOCK_LIVEACTION_4_RUNNING), 200, 'OK')))
    @mock.patch('st2client.client.StreamManager', autospec=True)
    def test_tail_double_nested_orquesta_workflow_execution(self, mock_stream_manager):
        if False:
            for i in range(10):
                print('nop')
        argv = ['execution', 'tail', 'idfoo4']
        MOCK_EVENTS = [MOCK_LIVEACTION_4_RUNNING, MOCK_LIVEACTION_4_CHILD_1_RUNNING, MOCK_LIVEACTION_4_CHILD_1_1_RUNNING, MOCK_LIVEACTION_4_CHILD_1_1_OUTPUT_1, MOCK_LIVEACTION_4_CHILD_1_1_OUTPUT_2, MOCK_LIVEACTION_3_RUNNING, MOCK_LIVEACTION_3_CHILD_1_RUNNING, MOCK_LIVEACTION_3_CHILD_1_OUTPUT_1, MOCK_LIVEACTION_3_CHILD_1_OUTPUT_2, MOCK_LIVEACTION_3_CHILD_1_SUCCEEDED, MOCK_LIVEACTION_3_SUCCEDED, MOCK_LIVEACTION_4_CHILD_1_1_SUCCEEDED, MOCK_LIVEACTION_4_CHILD_1_SUCCEEDED, MOCK_LIVEACTION_4_CHILD_2_RUNNING, MOCK_LIVEACTION_4_CHILD_2_OUTPUT_1, MOCK_LIVEACTION_4_CHILD_2_TIMED_OUT, MOCK_LIVEACTION_4_SUCCEDED]
        mock_cls = mock.Mock()
        mock_cls.listen = mock.Mock()
        mock_listen_generator = mock.Mock()
        mock_listen_generator.return_value = MOCK_EVENTS
        mock_cls.listen.side_effect = mock_listen_generator
        mock_stream_manager.return_value = mock_cls
        self.assertEqual(self.shell.run(argv), 0)
        self.assertEqual(mock_listen_generator.call_count, 1)
        stdout = self.stdout.getvalue()
        stderr = self.stderr.getvalue()
        expected_result = '\nExecution idfoo4 has started.\n\nChild execution (task=task_1) idorquestachild1 has started.\n\nChild execution (task=task_1) idorquestachild1_1 has started.\n\nline orquesta 4\nline orquesta 5\n\nChild execution (task=task_1) idorquestachild1_1 has finished (status=succeeded).\n\nChild execution (task=task_1) idorquestachild1 has finished (status=succeeded).\nChild execution (task=task_2) idorquestachild2 has started.\n\nline orquesta 100\n\nChild execution (task=task_2) idorquestachild2 has finished (status=timeout).\n\nExecution idfoo4 has completed (status=succeeded).\n'.lstrip()
        self.assertEqual(stdout, expected_result)
        self.assertEqual(stderr, '')

    @mock.patch.object(httpclient.HTTPClient, 'get', mock.MagicMock(return_value=base.FakeResponse(json.dumps(MOCK_LIVEACTION_4_CHILD_2_RUNNING), 200, 'OK')))
    @mock.patch('st2client.client.StreamManager', autospec=True)
    def test_tail_child_execution_directly(self, mock_stream_manager):
        if False:
            return 10
        argv = ['execution', 'tail', 'idfoo4']
        MOCK_EVENTS = [MOCK_LIVEACTION_4_CHILD_2_RUNNING, MOCK_LIVEACTION_4_CHILD_2_OUTPUT_1, MOCK_LIVEACTION_3_CHILD_1_RUNNING, MOCK_LIVEACTION_4_CHILD_1_SUCCEEDED, MOCK_LIVEACTION_4_CHILD_2_TIMED_OUT]
        mock_cls = mock.Mock()
        mock_cls.listen = mock.Mock()
        mock_listen_generator = mock.Mock()
        mock_listen_generator.return_value = MOCK_EVENTS
        mock_cls.listen.side_effect = mock_listen_generator
        mock_stream_manager.return_value = mock_cls
        self.assertEqual(self.shell.run(argv), 0)
        self.assertEqual(mock_listen_generator.call_count, 1)
        stdout = self.stdout.getvalue()
        stderr = self.stderr.getvalue()
        expected_result = '\nChild execution (task=task_2) idorquestachild2 has started.\n\nline orquesta 100\n\nChild execution (task=task_2) idorquestachild2 has finished (status=timeout).\n'.lstrip()
        self.assertEqual(stdout, expected_result)
        self.assertEqual(stderr, '')