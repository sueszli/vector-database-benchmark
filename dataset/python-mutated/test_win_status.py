import sys
import pytest
import salt.modules.win_status as status
from tests.support.mock import ANY, Mock, patch
from tests.support.unit import TestCase
try:
    import wmi
except ImportError:
    pass

@pytest.mark.skipif(status.HAS_WMI is False, reason='This test requires Windows')
class TestProcsBase(TestCase):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        TestCase.__init__(self, *args, **kwargs)
        self.__processes = []

    def add_process(self, pid=100, cmd='cmd', name='name', user='user', user_domain='domain', get_owner_result=0):
        if False:
            print('Hello World!')
        process = Mock()
        process.GetOwner = Mock(return_value=(user_domain, get_owner_result, user))
        process.ProcessId = pid
        process.CommandLine = cmd
        process.Name = name
        self.__processes.append(process)

    def call_procs(self):
        if False:
            print('Hello World!')
        WMI = Mock()
        WMI.win32_process = Mock(return_value=self.__processes)
        with patch.object(wmi, 'WMI', Mock(return_value=WMI)):
            self.result = status.procs()

class TestProcsCount(TestProcsBase):

    def setUp(self):
        if False:
            return 10
        self.add_process(pid=100)
        self.add_process(pid=101)
        self.call_procs()

    def test_process_count(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(len(self.result), 2)

    def test_process_key_is_pid(self):
        if False:
            i = 10
            return i + 15
        self.assertSetEqual(set(self.result.keys()), {100, 101})

class TestProcsAttributes(TestProcsBase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self._expected_name = 'name'
        self._expected_cmd = 'cmd'
        self._expected_user = 'user'
        self._expected_domain = 'domain'
        pid = 100
        self.add_process(pid=pid, cmd=self._expected_cmd, user=self._expected_user, user_domain=self._expected_domain, get_owner_result=0)
        self.call_procs()
        self.proc = self.result[pid]

    def test_process_cmd_is_set(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.proc['cmd'], self._expected_cmd)

    def test_process_name_is_set(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.proc['name'], self._expected_name)

    def test_process_user_is_set(self):
        if False:
            return 10
        self.assertEqual(self.proc['user'], self._expected_user)

    def test_process_user_domain_is_set(self):
        if False:
            return 10
        self.assertEqual(self.proc['user_domain'], self._expected_domain)

@pytest.mark.skipif(sys.stdin.encoding != 'UTF-8', reason='UTF-8 encoding required for this test is not supported')
class TestProcsUnicodeAttributes(TestProcsBase):

    def setUp(self):
        if False:
            return 10
        unicode_str = 'Á'
        self.ustr = unicode_str
        pid = 100
        self.add_process(pid=pid, user=unicode_str, user_domain=unicode_str, cmd=unicode_str, name=unicode_str)
        self.call_procs()
        self.proc = self.result[pid]

    def test_process_cmd_is_utf8(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.proc['cmd'], self.ustr)

    def test_process_name_is_utf8(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.proc['name'], self.ustr)

    def test_process_user_is_utf8(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.proc['user'], self.ustr)

    def test_process_user_domain_is_utf8(self):
        if False:
            return 10
        self.assertEqual(self.proc['user_domain'], self.ustr)

class TestProcsWMIGetOwnerAccessDeniedWorkaround(TestProcsBase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.expected_user = 'SYSTEM'
        self.expected_domain = 'NT AUTHORITY'
        self.add_process(pid=0, get_owner_result=2)
        self.add_process(pid=4, get_owner_result=2)
        self.call_procs()

    def test_user_is_set(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.result[0]['user'], self.expected_user)
        self.assertEqual(self.result[4]['user'], self.expected_user)

    def test_process_user_domain_is_set(self):
        if False:
            return 10
        self.assertEqual(self.result[0]['user_domain'], self.expected_domain)
        self.assertEqual(self.result[4]['user_domain'], self.expected_domain)

class TestProcsWMIGetOwnerErrorsAreLogged(TestProcsBase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.expected_error_code = 8
        self.add_process(get_owner_result=self.expected_error_code)

    def test_error_logged_if_process_get_owner_fails(self):
        if False:
            i = 10
            return i + 15
        with patch('salt.modules.win_status.log') as log:
            self.call_procs()
        log.warning.assert_called_once_with(ANY, ANY, self.expected_error_code)

class TestEmptyCommandLine(TestProcsBase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.expected_error_code = 8
        pid = 100
        self.add_process(pid=pid, cmd=None)
        self.call_procs()
        self.proc = self.result[pid]

    def test_cmd_is_empty_string(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.proc['cmd'], '')