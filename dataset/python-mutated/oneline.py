from __future__ import annotations
DOCUMENTATION = '\n    name: oneline\n    type: stdout\n    short_description: oneline Ansible screen output\n    version_added: historical\n    description:\n        - This is the output callback used by the C(-o)/C(--one-line) command line option.\n'
from ansible.plugins.callback import CallbackBase
from ansible import constants as C

class CallbackModule(CallbackBase):
    """
    This is the default callback interface, which simply prints messages
    to stdout when new callback events are received.
    """
    CALLBACK_VERSION = 2.0
    CALLBACK_TYPE = 'stdout'
    CALLBACK_NAME = 'oneline'

    def _command_generic_msg(self, hostname, result, caption):
        if False:
            print('Hello World!')
        stdout = result.get('stdout', '').replace('\n', '\\n').replace('\r', '\\r')
        if 'stderr' in result and result['stderr']:
            stderr = result.get('stderr', '').replace('\n', '\\n').replace('\r', '\\r')
            return '%s | %s | rc=%s | (stdout) %s (stderr) %s' % (hostname, caption, result.get('rc', -1), stdout, stderr)
        else:
            return '%s | %s | rc=%s | (stdout) %s' % (hostname, caption, result.get('rc', -1), stdout)

    def v2_runner_on_failed(self, result, ignore_errors=False):
        if False:
            print('Hello World!')
        if 'exception' in result._result:
            if self._display.verbosity < 3:
                error = result._result['exception'].strip().split('\n')[-1]
                msg = 'An exception occurred during task execution. To see the full traceback, use -vvv. The error was: %s' % error
            else:
                msg = 'An exception occurred during task execution. The full traceback is:\n' + result._result['exception'].replace('\n', '')
            if result._task.action in C.MODULE_NO_JSON and 'module_stderr' not in result._result:
                self._display.display(self._command_generic_msg(result._host.get_name(), result._result, 'FAILED'), color=C.COLOR_ERROR)
            else:
                self._display.display(msg, color=C.COLOR_ERROR)
        self._display.display('%s | FAILED! => %s' % (result._host.get_name(), self._dump_results(result._result, indent=0).replace('\n', '')), color=C.COLOR_ERROR)

    def v2_runner_on_ok(self, result):
        if False:
            while True:
                i = 10
        if result._result.get('changed', False):
            color = C.COLOR_CHANGED
            state = 'CHANGED'
        else:
            color = C.COLOR_OK
            state = 'SUCCESS'
        if result._task.action in C.MODULE_NO_JSON and 'ansible_job_id' not in result._result:
            self._display.display(self._command_generic_msg(result._host.get_name(), result._result, state), color=color)
        else:
            self._display.display('%s | %s => %s' % (result._host.get_name(), state, self._dump_results(result._result, indent=0).replace('\n', '')), color=color)

    def v2_runner_on_unreachable(self, result):
        if False:
            i = 10
            return i + 15
        self._display.display('%s | UNREACHABLE!: %s' % (result._host.get_name(), result._result.get('msg', '')), color=C.COLOR_UNREACHABLE)

    def v2_runner_on_skipped(self, result):
        if False:
            i = 10
            return i + 15
        self._display.display('%s | SKIPPED' % result._host.get_name(), color=C.COLOR_SKIP)