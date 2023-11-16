from __future__ import absolute_import
import uuid
from st2common.models.system.action import ShellCommandAction
from st2common.runners.base import get_metadata as get_runner_metadata
from local_runner.base import BaseLocalShellRunner
from local_runner.base import RUNNER_COMMAND
__all__ = ['LocalShellCommandRunner', 'get_runner', 'get_metadata']

class LocalShellCommandRunner(BaseLocalShellRunner):

    def run(self, action_parameters):
        if False:
            print('Hello World!')
        if self.entry_point:
            raise ValueError('entry_point is only valid for local-shell-script runner')
        command = self.runner_parameters.get(RUNNER_COMMAND, None)
        action = ShellCommandAction(name=self.action_name, action_exec_id=str(self.liveaction_id), command=command, user=self._user, env_vars=self._env, sudo=self._sudo, timeout=self._timeout, sudo_password=self._sudo_password)
        return self._run(action=action)

def get_runner():
    if False:
        return 10
    return LocalShellCommandRunner(str(uuid.uuid4()))

def get_metadata():
    if False:
        i = 10
        return i + 15
    metadata = get_runner_metadata('local_runner')
    metadata = [runner for runner in metadata if runner['runner_module'] == __name__.split('.')[-1]][0]
    return metadata