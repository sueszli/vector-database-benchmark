from __future__ import absolute_import
import uuid
from oslo_config import cfg
from st2common import log as logging
from st2common.runners.paramiko_ssh_runner import RUNNER_COMMAND
from st2common.runners.paramiko_ssh_runner import BaseParallelSSHRunner
from st2common.runners.base import get_metadata as get_runner_metadata
from st2common.models.system.paramiko_command_action import ParamikoRemoteCommandAction
__all__ = ['ParamikoRemoteCommandRunner', 'get_runner', 'get_metadata']
LOG = logging.getLogger(__name__)

class ParamikoRemoteCommandRunner(BaseParallelSSHRunner):

    def run(self, action_parameters):
        if False:
            i = 10
            return i + 15
        remote_action = self._get_remote_action(action_parameters)
        LOG.debug('Executing remote command action.', extra={'_action_params': remote_action})
        result = self._run(remote_action)
        LOG.debug('Executed remote_action.', extra={'_result': result})
        status = self._get_result_status(result, cfg.CONF.ssh_runner.allow_partial_failure)
        return (status, result, None)

    def _run(self, remote_action):
        if False:
            i = 10
            return i + 15
        command = remote_action.get_full_command_string()
        return self._parallel_ssh_client.run(command, timeout=remote_action.get_timeout())

    def _get_remote_action(self, action_paramaters):
        if False:
            while True:
                i = 10
        if self.entry_point:
            msg = 'Action "%s" specified "entry_point" attribute. Perhaps wanted to use "remote-shell-script" runner?' % self.action_name
            raise Exception(msg)
        command = self.runner_parameters.get(RUNNER_COMMAND, None)
        env_vars = self._get_env_vars()
        return ParamikoRemoteCommandAction(self.action_name, str(self.liveaction_id), command, env_vars=env_vars, on_behalf_user=self._on_behalf_user, user=self._username, password=self._password, private_key=self._private_key, passphrase=self._passphrase, hosts=self._hosts, parallel=self._parallel, sudo=self._sudo, sudo_password=self._sudo_password, timeout=self._timeout, cwd=self._cwd)

def get_runner():
    if False:
        i = 10
        return i + 15
    return ParamikoRemoteCommandRunner(str(uuid.uuid4()))

def get_metadata():
    if False:
        i = 10
        return i + 15
    metadata = get_runner_metadata('remote_runner')
    metadata = [runner for runner in metadata if runner['runner_module'] == __name__.split('.')[-1]][0]
    return metadata