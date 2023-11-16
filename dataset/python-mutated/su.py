from __future__ import annotations
DOCUMENTATION = '\n    name: su\n    short_description: Substitute User\n    description:\n        - This become plugin allows your remote/login user to execute commands as another user via the su utility.\n    author: ansible (@core)\n    version_added: "2.8"\n    options:\n        become_user:\n            description: User you \'become\' to execute the task\n            default: root\n            ini:\n              - section: privilege_escalation\n                key: become_user\n              - section: su_become_plugin\n                key: user\n            vars:\n              - name: ansible_become_user\n              - name: ansible_su_user\n            env:\n              - name: ANSIBLE_BECOME_USER\n              - name: ANSIBLE_SU_USER\n            keyword:\n              - name: become_user\n        become_exe:\n            description: Su executable\n            default: su\n            ini:\n              - section: privilege_escalation\n                key: become_exe\n              - section: su_become_plugin\n                key: executable\n            vars:\n              - name: ansible_become_exe\n              - name: ansible_su_exe\n            env:\n              - name: ANSIBLE_BECOME_EXE\n              - name: ANSIBLE_SU_EXE\n            keyword:\n              - name: become_exe\n        become_flags:\n            description: Options to pass to su\n            default: \'\'\n            ini:\n              - section: privilege_escalation\n                key: become_flags\n              - section: su_become_plugin\n                key: flags\n            vars:\n              - name: ansible_become_flags\n              - name: ansible_su_flags\n            env:\n              - name: ANSIBLE_BECOME_FLAGS\n              - name: ANSIBLE_SU_FLAGS\n            keyword:\n              - name: become_flags\n        become_pass:\n            description: Password to pass to su\n            required: False\n            vars:\n              - name: ansible_become_password\n              - name: ansible_become_pass\n              - name: ansible_su_pass\n            env:\n              - name: ANSIBLE_BECOME_PASS\n              - name: ANSIBLE_SU_PASS\n            ini:\n              - section: su_become_plugin\n                key: password\n        prompt_l10n:\n            description:\n                - List of localized strings to match for prompt detection\n                - If empty we\'ll use the built in one\n                - Do NOT add a colon (:) to your custom entries. Ansible adds a colon at the end of each prompt;\n                  if you add another one in your string, your prompt will fail with a "Timeout" error.\n            default: []\n            type: list\n            elements: string\n            ini:\n              - section: su_become_plugin\n                key: localized_prompts\n            vars:\n              - name: ansible_su_prompt_l10n\n            env:\n              - name: ANSIBLE_SU_PROMPT_L10N\n'
import re
import shlex
from ansible.module_utils.common.text.converters import to_bytes
from ansible.plugins.become import BecomeBase

class BecomeModule(BecomeBase):
    name = 'su'
    fail = ('Authentication failure',)
    SU_PROMPT_LOCALIZATIONS = ['Password', '암호', 'パスワード', 'Adgangskode', 'Contraseña', 'Contrasenya', 'Hasło', 'Heslo', 'Jelszó', 'Lösenord', 'Mật khẩu', 'Mot de passe', 'Parola', 'Parool', 'Pasahitza', 'Passord', 'Passwort', 'Salasana', 'Sandi', 'Senha', 'Wachtwoord', 'ססמה', 'Лозинка', 'Парола', 'Пароль', 'गुप्तशब्द', 'शब्दकूट', 'సంకేతపదము', 'හස්පදය', '密码', '密碼', '口令']

    def check_password_prompt(self, b_output):
        if False:
            return 10
        ' checks if the expected password prompt exists in b_output '
        prompts = self.get_option('prompt_l10n') or self.SU_PROMPT_LOCALIZATIONS
        b_password_string = b'|'.join((b"(\\w+\\'s )?" + to_bytes(p) for p in prompts))
        b_password_string = b_password_string + to_bytes(u' ?(:|：) ?')
        b_su_prompt_localizations_re = re.compile(b_password_string, flags=re.IGNORECASE)
        return bool(b_su_prompt_localizations_re.match(b_output))

    def build_become_command(self, cmd, shell):
        if False:
            print('Hello World!')
        super(BecomeModule, self).build_become_command(cmd, shell)
        self.prompt = True
        if not cmd:
            return cmd
        exe = self.get_option('become_exe') or self.name
        flags = self.get_option('become_flags') or ''
        user = self.get_option('become_user') or ''
        success_cmd = self._build_success_command(cmd, shell)
        return '%s %s %s -c %s' % (exe, flags, user, shlex.quote(success_cmd))