from __future__ import annotations
DOCUMENTATION = '\n    name: sudo\n    short_description: Substitute User DO\n    description:\n        - This become plugin allows your remote/login user to execute commands as another user via the sudo utility.\n    author: ansible (@core)\n    version_added: "2.8"\n    options:\n        become_user:\n            description: User you \'become\' to execute the task\n            default: root\n            ini:\n              - section: privilege_escalation\n                key: become_user\n              - section: sudo_become_plugin\n                key: user\n            vars:\n              - name: ansible_become_user\n              - name: ansible_sudo_user\n            env:\n              - name: ANSIBLE_BECOME_USER\n              - name: ANSIBLE_SUDO_USER\n            keyword:\n              - name: become_user\n        become_exe:\n            description: Sudo executable\n            default: sudo\n            ini:\n              - section: privilege_escalation\n                key: become_exe\n              - section: sudo_become_plugin\n                key: executable\n            vars:\n              - name: ansible_become_exe\n              - name: ansible_sudo_exe\n            env:\n              - name: ANSIBLE_BECOME_EXE\n              - name: ANSIBLE_SUDO_EXE\n            keyword:\n              - name: become_exe\n        become_flags:\n            description: Options to pass to sudo\n            default: -H -S -n\n            ini:\n              - section: privilege_escalation\n                key: become_flags\n              - section: sudo_become_plugin\n                key: flags\n            vars:\n              - name: ansible_become_flags\n              - name: ansible_sudo_flags\n            env:\n              - name: ANSIBLE_BECOME_FLAGS\n              - name: ANSIBLE_SUDO_FLAGS\n            keyword:\n              - name: become_flags\n        become_pass:\n            description: Password to pass to sudo\n            required: False\n            vars:\n              - name: ansible_become_password\n              - name: ansible_become_pass\n              - name: ansible_sudo_pass\n            env:\n              - name: ANSIBLE_BECOME_PASS\n              - name: ANSIBLE_SUDO_PASS\n            ini:\n              - section: sudo_become_plugin\n                key: password\n'
import re
import shlex
from ansible.plugins.become import BecomeBase

class BecomeModule(BecomeBase):
    name = 'sudo'
    fail = ('Sorry, try again.',)
    missing = ('Sorry, a password is required to run sudo', 'sudo: a password is required')

    def build_become_command(self, cmd, shell):
        if False:
            print('Hello World!')
        super(BecomeModule, self).build_become_command(cmd, shell)
        if not cmd:
            return cmd
        becomecmd = self.get_option('become_exe') or self.name
        flags = self.get_option('become_flags') or ''
        prompt = ''
        if self.get_option('become_pass'):
            self.prompt = '[sudo via ansible, key=%s] password:' % self._id
            if flags:
                reflag = []
                for flag in shlex.split(flags):
                    if flag in ('-n', '--non-interactive'):
                        continue
                    elif not flag.startswith('--'):
                        flag = re.sub('^(-\\w*)n(\\w*.*)', '\\1\\2', flag)
                    reflag.append(flag)
                flags = shlex.join(reflag)
            prompt = '-p "%s"' % self.prompt
        user = self.get_option('become_user') or ''
        if user:
            user = '-u %s' % user
        return ' '.join([becomecmd, flags, prompt, user, self._build_success_command(cmd, shell)])