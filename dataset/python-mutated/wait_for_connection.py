from __future__ import annotations
import time
from datetime import datetime, timedelta, timezone
from ansible.module_utils.common.text.converters import to_text
from ansible.plugins.action import ActionBase
from ansible.utils.display import Display
display = Display()

class TimedOutException(Exception):
    pass

class ActionModule(ActionBase):
    TRANSFERS_FILES = False
    _VALID_ARGS = frozenset(('connect_timeout', 'delay', 'sleep', 'timeout'))
    DEFAULT_CONNECT_TIMEOUT = 5
    DEFAULT_DELAY = 0
    DEFAULT_SLEEP = 1
    DEFAULT_TIMEOUT = 600

    def do_until_success_or_timeout(self, what, timeout, connect_timeout, what_desc, sleep=1):
        if False:
            print('Hello World!')
        max_end_time = datetime.now(timezone.utc) + timedelta(seconds=timeout)
        e = None
        while datetime.now(timezone.utc) < max_end_time:
            try:
                what(connect_timeout)
                if what_desc:
                    display.debug('wait_for_connection: %s success' % what_desc)
                return
            except Exception as e:
                error = e
                if what_desc:
                    display.debug('wait_for_connection: %s fail (expected), retrying in %d seconds...' % (what_desc, sleep))
                time.sleep(sleep)
        raise TimedOutException('timed out waiting for %s: %s' % (what_desc, error))

    def run(self, tmp=None, task_vars=None):
        if False:
            for i in range(10):
                print('nop')
        if task_vars is None:
            task_vars = dict()
        connect_timeout = int(self._task.args.get('connect_timeout', self.DEFAULT_CONNECT_TIMEOUT))
        delay = int(self._task.args.get('delay', self.DEFAULT_DELAY))
        sleep = int(self._task.args.get('sleep', self.DEFAULT_SLEEP))
        timeout = int(self._task.args.get('timeout', self.DEFAULT_TIMEOUT))
        if self._play_context.check_mode:
            display.vvv('wait_for_connection: skipping for check_mode')
            return dict(skipped=True)
        result = super(ActionModule, self).run(tmp, task_vars)
        del tmp

        def ping_module_test(connect_timeout):
            if False:
                return 10
            ' Test ping module, if available '
            display.vvv('wait_for_connection: attempting ping module test')
            if self._discovered_interpreter_key:
                task_vars['ansible_facts'].pop(self._discovered_interpreter_key, None)
            try:
                self._connection.reset()
            except AttributeError:
                pass
            ping_result = self._execute_module(module_name='ansible.legacy.ping', module_args=dict(), task_vars=task_vars)
            if ping_result['ping'] != 'pong':
                raise Exception('ping test failed')
        start = datetime.now()
        if delay:
            time.sleep(delay)
        try:
            if hasattr(self._connection, 'transport_test'):
                self.do_until_success_or_timeout(self._connection.transport_test, timeout, connect_timeout, what_desc='connection port up', sleep=sleep)
            self.do_until_success_or_timeout(ping_module_test, timeout, connect_timeout, what_desc='ping module test', sleep=sleep)
        except TimedOutException as e:
            result['failed'] = True
            result['msg'] = to_text(e)
        elapsed = datetime.now() - start
        result['elapsed'] = elapsed.seconds
        self._remove_tmp_path(self._connection._shell.tmpdir)
        return result