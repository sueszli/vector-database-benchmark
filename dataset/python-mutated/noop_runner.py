from __future__ import absolute_import
import uuid
from st2common import log as logging
from st2common.runners.base import ActionRunner
from st2common.runners.base import get_metadata as get_runner_metadata
from st2common.constants.action import LIVEACTION_STATUS_SUCCEEDED
import st2common.util.jsonify as jsonify
__all__ = ['NoopRunner', 'get_runner', 'get_metadata']
LOG = logging.getLogger(__name__)

class NoopRunner(ActionRunner):
    """
    Runner which does absolutely nothing. No-op action.
    """
    KEYS_TO_TRANSFORM = ['stdout', 'stderr']

    def __init__(self, runner_id):
        if False:
            return 10
        super(NoopRunner, self).__init__(runner_id=runner_id)

    def pre_run(self):
        if False:
            for i in range(10):
                print('nop')
        super(NoopRunner, self).pre_run()

    def run(self, action_parameters):
        if False:
            while True:
                i = 10
        LOG.info('Executing action via NoopRunner: %s', self.runner_id)
        LOG.info('[Action info] name: %s, Id: %s', self.action_name, str(self.execution_id))
        result = {'failed': False, 'succeeded': True, 'return_code': 0}
        status = LIVEACTION_STATUS_SUCCEEDED
        return (status, jsonify.json_loads(result, NoopRunner.KEYS_TO_TRANSFORM), None)

def get_runner():
    if False:
        return 10
    return NoopRunner(str(uuid.uuid4()))

def get_metadata():
    if False:
        while True:
            i = 10
    return get_runner_metadata('noop_runner')[0]