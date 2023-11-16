import json
from io import BytesIO
from unittest import TestCase, main
from nni.runtime import msg_dispatcher_base
from nni.runtime.msg_dispatcher import MsgDispatcher
from nni.runtime.tuner_command_channel.legacy import *
from nni.tuner import Tuner
from nni.utils import extract_scalar_reward

class NaiveTuner(Tuner):

    def __init__(self):
        if False:
            return 10
        self.param = 0
        self.trial_results = []
        self.search_space = None
        self._accept_customized_trials()

    def generate_parameters(self, parameter_id, **kwargs):
        if False:
            print('Hello World!')
        self.param += 2
        return {'param': self.param, 'trial_results': self.trial_results, 'search_space': self.search_space}

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        if False:
            i = 10
            return i + 15
        reward = extract_scalar_reward(value)
        self.trial_results.append((parameter_id, parameters['param'], reward, kwargs.get('customized')))

    def update_search_space(self, search_space):
        if False:
            i = 10
            return i + 15
        self.search_space = search_space
_in_buf = BytesIO()
_out_buf = BytesIO()

def _reverse_io():
    if False:
        i = 10
        return i + 15
    _in_buf.seek(0)
    _out_buf.seek(0)
    _set_out_file(_in_buf)
    _set_in_file(_out_buf)

def _restore_io():
    if False:
        print('Hello World!')
    _in_buf.seek(0)
    _out_buf.seek(0)
    _set_in_file(_in_buf)
    _set_out_file(_out_buf)

class MsgDispatcherTestCase(TestCase):

    def test_msg_dispatcher(self):
        if False:
            while True:
                i = 10
        _reverse_io()
        send(CommandType.RequestTrialJobs, '2')
        send(CommandType.ReportMetricData, '{"parameter_id":0,"type":"PERIODICAL","value":"10"}')
        send(CommandType.ReportMetricData, '{"parameter_id":1,"type":"FINAL","value":"11"}')
        send(CommandType.UpdateSearchSpace, '{"name":"SS0"}')
        send(CommandType.RequestTrialJobs, '1')
        send(CommandType.KillTrialJob, 'null')
        _restore_io()
        tuner = NaiveTuner()
        dispatcher = MsgDispatcher('ws://_unittest_placeholder_', tuner)
        dispatcher._channel = LegacyCommandChannel()
        msg_dispatcher_base._worker_fast_exit_on_terminate = False
        dispatcher.run()
        e = dispatcher.worker_exceptions[0]
        self.assertIs(type(e), AssertionError)
        self.assertEqual(e.args[0], 'Unsupported command: CommandType.KillTrialJob')
        _reverse_io()
        self._assert_params(0, 2, [], None)
        self._assert_params(1, 4, [], None)
        self._assert_params(2, 6, [[1, 4, 11, False]], {'name': 'SS0'})
        self.assertEqual(len(_out_buf.read()), 0)

    def _assert_params(self, parameter_id, param, trial_results, search_space):
        if False:
            i = 10
            return i + 15
        (command, data) = receive()
        self.assertIs(command, CommandType.NewTrialJob)
        data = json.loads(data)
        self.assertEqual(data['parameter_id'], parameter_id)
        self.assertEqual(data['parameter_source'], 'algorithm')
        self.assertEqual(data['parameters']['param'], param)
        self.assertEqual(data['parameters']['trial_results'], trial_results)
        self.assertEqual(data['parameters']['search_space'], search_space)
if __name__ == '__main__':
    main()