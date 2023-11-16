from __future__ import absolute_import
import os
import sys
import signal
import tempfile
from st2common.util import concurrency
from st2common.constants.timer import TIMER_ENABLED_LOG_LINE
from st2common.constants.timer import TIMER_DISABLED_LOG_LINE
from st2tests.base import IntegrationTestCase
from st2tests.base import CleanDbTestCase
__all__ = ['TimersEngineServiceEnableDisableTestCase']
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ST2_CONFIG_PATH = os.path.join(BASE_DIR, '../../../conf/st2.tests.conf')
ST2_CONFIG_PATH = os.path.abspath(ST2_CONFIG_PATH)
PYTHON_BINARY = sys.executable
BINARY = os.path.join(BASE_DIR, '../../../st2reactor/bin/st2timersengine')
BINARY = os.path.abspath(BINARY)
CMD = [PYTHON_BINARY, BINARY, '--config-file']

class TimersEngineServiceEnableDisableTestCase(IntegrationTestCase, CleanDbTestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super(TimersEngineServiceEnableDisableTestCase, self).setUp()
        config_text = open(ST2_CONFIG_PATH).read()
        (self.cfg_fd, self.cfg_path) = tempfile.mkstemp()
        with open(self.cfg_path, 'w') as f:
            f.write(config_text)
        self.cmd = []
        self.cmd.extend(CMD)
        self.cmd.append(self.cfg_path)

    def tearDown(self):
        if False:
            print('Hello World!')
        self.cmd = None
        self._remove_tempfile(self.cfg_fd, self.cfg_path)
        super(TimersEngineServiceEnableDisableTestCase, self).tearDown()

    def test_timer_enable_implicit(self):
        if False:
            while True:
                i = 10
        process = None
        seen_line = False
        try:
            process = self._start_times_engine(cmd=self.cmd)
            lines = 0
            while lines < 100:
                line = process.stdout.readline().decode('utf-8')
                lines += 1
                sys.stdout.write(line)
                if TIMER_ENABLED_LOG_LINE in line:
                    seen_line = True
                    break
        finally:
            if process:
                process.send_signal(signal.SIGKILL)
                self.remove_process(process=process)
        if not seen_line:
            raise AssertionError('Didn\'t see "%s" log line in timer output' % TIMER_ENABLED_LOG_LINE)

    def test_timer_enable_explicit(self):
        if False:
            for i in range(10):
                print('nop')
        self._append_to_cfg_file(cfg_path=self.cfg_path, content='\n[timersengine]\nenable = True\n[timer]\nenable = True')
        process = None
        seen_line = False
        try:
            process = self._start_times_engine(cmd=self.cmd)
            lines = 0
            while lines < 100:
                line = process.stdout.readline().decode('utf-8')
                lines += 1
                sys.stdout.write(line)
                if TIMER_ENABLED_LOG_LINE in line:
                    seen_line = True
                    break
        finally:
            if process:
                process.send_signal(signal.SIGKILL)
                self.remove_process(process=process)
        if not seen_line:
            raise AssertionError('Didn\'t see "%s" log line in timer output' % TIMER_ENABLED_LOG_LINE)

    def test_timer_disable_explicit(self):
        if False:
            for i in range(10):
                print('nop')
        self._append_to_cfg_file(cfg_path=self.cfg_path, content='\n[timersengine]\nenable = False\n[timer]\nenable = False')
        process = None
        seen_line = False
        try:
            process = self._start_times_engine(cmd=self.cmd)
            lines = 0
            while lines < 100:
                line = process.stdout.readline().decode('utf-8')
                lines += 1
                sys.stdout.write(line)
                if TIMER_DISABLED_LOG_LINE in line:
                    seen_line = True
                    break
        finally:
            if process:
                process.send_signal(signal.SIGKILL)
                self.remove_process(process=process)
        if not seen_line:
            raise AssertionError('Didn\'t see "%s" log line in timer output' % TIMER_DISABLED_LOG_LINE)

    def _start_times_engine(self, cmd):
        if False:
            print('Hello World!')
        subprocess = concurrency.get_subprocess_module()
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, preexec_fn=os.setsid)
        self.add_process(process=process)
        return process

    def _append_to_cfg_file(self, cfg_path, content):
        if False:
            i = 10
            return i + 15
        with open(cfg_path, 'a') as f:
            f.write(content)

    def _remove_tempfile(self, fd, path):
        if False:
            return 10
        os.close(fd)
        os.unlink(path)