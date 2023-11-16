from __future__ import annotations
import json
import re
from ansible.errors import AnsibleConnectionFailure
from ansible.module_utils.common.text.converters import to_text, to_bytes
from ansible.plugins.terminal import TerminalBase
from ansible.utils.display import Display
display = Display()

class TerminalModule(TerminalBase):
    terminal_stdout_re = [re.compile(b'[\\r\\n]?[\\w\\+\\-\\.:\\/\\[\\]]+(?:\\([^\\)]+\\)){0,3}(?:[>#]) ?$')]
    terminal_stderr_re = [re.compile(b'% ?Error'), re.compile(b'% ?Bad secret'), re.compile(b'[\\r\\n%] Bad passwords'), re.compile(b'invalid input', re.I), re.compile(b'(?:incomplete|ambiguous) command', re.I), re.compile(b'connection timed out', re.I), re.compile(b'[^\\r\\n]+ not found'), re.compile(b"'[^']' +returned error code: ?\\d+"), re.compile(b'Bad mask', re.I), re.compile(b'% ?(\\S+) ?overlaps with ?(\\S+)', re.I), re.compile(b'[%\\S] ?Error: ?[\\s]+', re.I), re.compile(b'[%\\S] ?Informational: ?[\\s]+', re.I), re.compile(b'Command authorization failed')]

    def on_open_shell(self):
        if False:
            print('Hello World!')
        try:
            self._exec_cli_command(b'terminal length 0')
        except AnsibleConnectionFailure:
            raise AnsibleConnectionFailure('unable to set terminal parameters')
        try:
            self._exec_cli_command(b'terminal width 512')
            try:
                self._exec_cli_command(b'terminal width 0')
            except AnsibleConnectionFailure:
                pass
        except AnsibleConnectionFailure:
            display.display('WARNING: Unable to set terminal width, command responses may be truncated')

    def on_become(self, passwd=None):
        if False:
            i = 10
            return i + 15
        if self._get_prompt().endswith(b'#'):
            return
        cmd = {u'command': u'enable'}
        if passwd:
            cmd[u'prompt'] = to_text('[\\r\\n]?(?:.*)?[Pp]assword: ?$', errors='surrogate_or_strict')
            cmd[u'answer'] = passwd
            cmd[u'prompt_retry_check'] = True
        try:
            self._exec_cli_command(to_bytes(json.dumps(cmd), errors='surrogate_or_strict'))
            prompt = self._get_prompt()
            if prompt is None or not prompt.endswith(b'#'):
                raise AnsibleConnectionFailure('failed to elevate privilege to enable mode still at prompt [%s]' % prompt)
        except AnsibleConnectionFailure as e:
            prompt = self._get_prompt()
            raise AnsibleConnectionFailure('unable to elevate privilege to enable mode, at prompt [%s] with error: %s' % (prompt, e.message))

    def on_unbecome(self):
        if False:
            while True:
                i = 10
        prompt = self._get_prompt()
        if prompt is None:
            return
        if b'(config' in prompt:
            self._exec_cli_command(b'end')
            self._exec_cli_command(b'disable')
        elif prompt.endswith(b'#'):
            self._exec_cli_command(b'disable')