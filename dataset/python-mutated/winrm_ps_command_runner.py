from __future__ import absolute_import
import uuid
from st2common import log as logging
from st2common.runners.base import get_metadata as get_runner_metadata
from winrm_runner.winrm_base import WinRmBaseRunner
__all__ = ['WinRmPsCommandRunner', 'get_runner', 'get_metadata']
LOG = logging.getLogger(__name__)
RUNNER_COMMAND = 'cmd'

class WinRmPsCommandRunner(WinRmBaseRunner):

    def run(self, action_parameters):
        if False:
            while True:
                i = 10
        powershell_command = self.runner_parameters[RUNNER_COMMAND]
        return self.run_ps(powershell_command)

def get_runner():
    if False:
        i = 10
        return i + 15
    return WinRmPsCommandRunner(str(uuid.uuid4()))

def get_metadata():
    if False:
        while True:
            i = 10
    metadata = get_runner_metadata('winrm_runner')
    metadata = [runner for runner in metadata if runner['runner_module'] == __name__.split('.')[-1]][0]
    return metadata