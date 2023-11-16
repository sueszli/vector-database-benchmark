from __future__ import absolute_import
import uuid
from st2common import log as logging
from st2common.runners.base import ShellRunnerMixin
from st2common.runners.base import get_metadata as get_runner_metadata
from winrm_runner.winrm_base import WinRmBaseRunner
__all__ = ['WinRmPsScriptRunner', 'get_runner', 'get_metadata']
LOG = logging.getLogger(__name__)

class WinRmPsScriptRunner(WinRmBaseRunner, ShellRunnerMixin):

    def run(self, action_parameters):
        if False:
            return 10
        if not self.entry_point:
            raise ValueError('Missing entry_point action metadata attribute')
        with open(self.entry_point, 'r') as script_file:
            ps_script = script_file.read()
        (positional_args, named_args) = self._get_script_args(action_parameters)
        named_args = self._transform_named_args(named_args)
        ps_params = self.create_ps_params_string(positional_args, named_args)
        return self.run_ps(ps_script, ps_params)

def get_runner():
    if False:
        while True:
            i = 10
    return WinRmPsScriptRunner(str(uuid.uuid4()))

def get_metadata():
    if False:
        while True:
            i = 10
    metadata = get_runner_metadata('winrm_runner')
    metadata = [runner for runner in metadata if runner['runner_module'] == __name__.split('.')[-1]][0]
    return metadata