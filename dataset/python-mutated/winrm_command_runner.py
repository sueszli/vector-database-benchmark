from __future__ import absolute_import
import uuid
from st2common import log as logging
from st2common.runners.base import get_metadata as get_runner_metadata
from winrm_runner.winrm_base import WinRmBaseRunner
__all__ = ['WinRmCommandRunner', 'get_runner', 'get_metadata']
LOG = logging.getLogger(__name__)
RUNNER_COMMAND = 'cmd'

class WinRmCommandRunner(WinRmBaseRunner):

    def run(self, action_parameters):
        if False:
            print('Hello World!')
        cmd_command = self.runner_parameters[RUNNER_COMMAND]
        return self.run_cmd(cmd_command)

def get_runner():
    if False:
        return 10
    return WinRmCommandRunner(str(uuid.uuid4()))

def get_metadata():
    if False:
        print('Hello World!')
    metadata = get_runner_metadata('winrm_runner')
    metadata = [runner for runner in metadata if runner['runner_module'] == __name__.split('.')[-1]][0]
    return metadata