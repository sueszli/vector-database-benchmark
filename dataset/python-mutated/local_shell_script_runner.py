from __future__ import absolute_import
import uuid
from st2common.models.system.action import ShellScriptAction
from st2common.runners.base import GitWorktreeActionRunner
from st2common.runners.base import get_metadata as get_runner_metadata
from local_runner.base import BaseLocalShellRunner
__all__ = ['LocalShellScriptRunner', 'get_runner', 'get_metadata']

class LocalShellScriptRunner(BaseLocalShellRunner, GitWorktreeActionRunner):

    def run(self, action_parameters):
        if False:
            return 10
        if not self.entry_point:
            raise ValueError('Missing entry_point action metadata attribute')
        script_local_path_abs = self.entry_point
        (positional_args, named_args) = self._get_script_args(action_parameters)
        named_args = self._transform_named_args(named_args)
        action = ShellScriptAction(name=self.action_name, action_exec_id=str(self.liveaction_id), script_local_path_abs=script_local_path_abs, named_args=named_args, positional_args=positional_args, user=self._user, env_vars=self._env, sudo=self._sudo, timeout=self._timeout, cwd=self._cwd, sudo_password=self._sudo_password)
        return self._run(action=action)

def get_runner():
    if False:
        i = 10
        return i + 15
    return LocalShellScriptRunner(str(uuid.uuid4()))

def get_metadata():
    if False:
        i = 10
        return i + 15
    metadata = get_runner_metadata('local_runner')
    metadata = [runner for runner in metadata if runner['runner_module'] == __name__.split('.')[-1]][0]
    return metadata