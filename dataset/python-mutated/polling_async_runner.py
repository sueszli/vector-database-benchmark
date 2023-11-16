from __future__ import absolute_import
try:
    import simplejson as json
except:
    import json
from st2common.runners.base import PollingAsyncActionRunner
from st2common.constants.action import LIVEACTION_STATUS_RUNNING
RAISE_PROPERTY = 'raise'

def get_runner():
    if False:
        i = 10
        return i + 15
    return PollingAsyncTestRunner()

class PollingAsyncTestRunner(PollingAsyncActionRunner):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(PollingAsyncTestRunner, self).__init__(runner_id='1')
        self.pre_run_called = False
        self.run_called = False
        self.post_run_called = False

    def pre_run(self):
        if False:
            for i in range(10):
                print('nop')
        self.pre_run_called = True

    def run(self, action_params):
        if False:
            while True:
                i = 10
        self.run_called = True
        result = {}
        if self.runner_parameters.get(RAISE_PROPERTY, False):
            raise Exception('Raise required.')
        else:
            result = {'ran': True, 'action_params': action_params}
        return (LIVEACTION_STATUS_RUNNING, json.dumps(result), {'id': 'foo'})

    def post_run(self, status, result):
        if False:
            i = 10
            return i + 15
        self.post_run_called = True