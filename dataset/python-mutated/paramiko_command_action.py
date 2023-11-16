from __future__ import absolute_import
import os
import pwd
from st2common import log as logging
from st2common.models.system.action import RemoteAction
from st2common.models.system.action import SUDO_COMMON_OPTIONS
from st2common.util.shell import quote_unix
__all__ = ['ParamikoRemoteCommandAction']
LOG = logging.getLogger(__name__)
LOGGED_USER_USERNAME = pwd.getpwuid(os.getuid())[0]

class ParamikoRemoteCommandAction(RemoteAction):

    def get_full_command_string(self):
        if False:
            i = 10
            return i + 15
        env_str = self._get_env_vars_export_string()
        cwd = self.get_cwd()
        if self.sudo:
            if env_str:
                command = quote_unix('%s && cd %s && %s' % (env_str, cwd, self.command))
            else:
                command = quote_unix('cd %s && %s' % (cwd, self.command))
            sudo_arguments = ' '.join(self._get_common_sudo_arguments())
            command = 'sudo %s -- bash -c %s' % (sudo_arguments, command)
            if self.sudo_password:
                command = 'set +o history ; echo -e %s | %s' % (quote_unix('%s\n' % self.sudo_password), command)
        elif env_str:
            command = '%s && cd %s && %s' % (env_str, cwd, self.command)
        else:
            command = 'cd %s && %s' % (cwd, self.command)
        LOG.debug('Command to run on remote host will be: %s', command)
        return command

    def _get_common_sudo_arguments(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Retrieve a list of flags which are passed to sudo on every invocation.\n\n        :rtype: ``list``\n        '
        flags = []
        if self.sudo_password:
            flags.append('-S')
        flags = flags + SUDO_COMMON_OPTIONS
        return flags