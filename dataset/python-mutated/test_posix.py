"""POSIX specific tests."""
import datetime
import errno
import os
import re
import subprocess
import time
import unittest
import psutil
from psutil import AIX
from psutil import BSD
from psutil import LINUX
from psutil import MACOS
from psutil import OPENBSD
from psutil import POSIX
from psutil import SUNOS
from psutil.tests import HAS_NET_IO_COUNTERS
from psutil.tests import PYTHON_EXE
from psutil.tests import PsutilTestCase
from psutil.tests import mock
from psutil.tests import retry_on_failure
from psutil.tests import sh
from psutil.tests import skip_on_access_denied
from psutil.tests import spawn_testproc
from psutil.tests import terminate
from psutil.tests import which
if POSIX:
    import mmap
    import resource
    from psutil._psutil_posix import getpagesize

def ps(fmt, pid=None):
    if False:
        while True:
            i = 10
    'Wrapper for calling the ps command with a little bit of cross-platform\n    support for a narrow range of features.\n    '
    cmd = ['ps']
    if LINUX:
        cmd.append('--no-headers')
    if pid is not None:
        cmd.extend(['-p', str(pid)])
    elif SUNOS or AIX:
        cmd.append('-A')
    else:
        cmd.append('ax')
    if SUNOS:
        fmt = fmt.replace('start', 'stime')
    cmd.extend(['-o', fmt])
    output = sh(cmd)
    output = output.splitlines() if LINUX else output.splitlines()[1:]
    all_output = []
    for line in output:
        line = line.strip()
        try:
            line = int(line)
        except ValueError:
            pass
        all_output.append(line)
    if pid is None:
        return all_output
    else:
        return all_output[0]

def ps_name(pid):
    if False:
        print('Hello World!')
    field = 'command'
    if SUNOS:
        field = 'comm'
    return ps(field, pid).split()[0]

def ps_args(pid):
    if False:
        return 10
    field = 'command'
    if AIX or SUNOS:
        field = 'args'
    out = ps(field, pid)
    out = re.sub('\\(python.*?\\)$', '', out)
    return out.strip()

def ps_rss(pid):
    if False:
        for i in range(10):
            print('nop')
    field = 'rss'
    if AIX:
        field = 'rssize'
    return ps(field, pid)

def ps_vsz(pid):
    if False:
        print('Hello World!')
    field = 'vsz'
    if AIX:
        field = 'vsize'
    return ps(field, pid)

@unittest.skipIf(not POSIX, 'POSIX only')
class TestProcess(PsutilTestCase):
    """Compare psutil results against 'ps' command line utility (mainly)."""

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        cls.pid = spawn_testproc([PYTHON_EXE, '-E', '-O'], stdin=subprocess.PIPE).pid

    @classmethod
    def tearDownClass(cls):
        if False:
            print('Hello World!')
        terminate(cls.pid)

    def test_ppid(self):
        if False:
            while True:
                i = 10
        ppid_ps = ps('ppid', self.pid)
        ppid_psutil = psutil.Process(self.pid).ppid()
        self.assertEqual(ppid_ps, ppid_psutil)

    def test_uid(self):
        if False:
            print('Hello World!')
        uid_ps = ps('uid', self.pid)
        uid_psutil = psutil.Process(self.pid).uids().real
        self.assertEqual(uid_ps, uid_psutil)

    def test_gid(self):
        if False:
            while True:
                i = 10
        gid_ps = ps('rgid', self.pid)
        gid_psutil = psutil.Process(self.pid).gids().real
        self.assertEqual(gid_ps, gid_psutil)

    def test_username(self):
        if False:
            while True:
                i = 10
        username_ps = ps('user', self.pid)
        username_psutil = psutil.Process(self.pid).username()
        self.assertEqual(username_ps, username_psutil)

    def test_username_no_resolution(self):
        if False:
            print('Hello World!')
        p = psutil.Process()
        with mock.patch('psutil.pwd.getpwuid', side_effect=KeyError) as fun:
            self.assertEqual(p.username(), str(p.uids().real))
            assert fun.called

    @skip_on_access_denied()
    @retry_on_failure()
    def test_rss_memory(self):
        if False:
            while True:
                i = 10
        time.sleep(0.1)
        rss_ps = ps_rss(self.pid)
        rss_psutil = psutil.Process(self.pid).memory_info()[0] / 1024
        self.assertEqual(rss_ps, rss_psutil)

    @skip_on_access_denied()
    @retry_on_failure()
    def test_vsz_memory(self):
        if False:
            while True:
                i = 10
        time.sleep(0.1)
        vsz_ps = ps_vsz(self.pid)
        vsz_psutil = psutil.Process(self.pid).memory_info()[1] / 1024
        self.assertEqual(vsz_ps, vsz_psutil)

    def test_name(self):
        if False:
            return 10
        name_ps = ps_name(self.pid)
        name_ps = os.path.basename(name_ps).lower()
        name_psutil = psutil.Process(self.pid).name().lower()
        name_ps = re.sub('\\d.\\d', '', name_ps)
        name_psutil = re.sub('\\d.\\d', '', name_psutil)
        name_ps = re.sub('\\d', '', name_ps)
        name_psutil = re.sub('\\d', '', name_psutil)
        self.assertEqual(name_ps, name_psutil)

    def test_name_long(self):
        if False:
            return 10
        name = 'long-program-name'
        cmdline = ['long-program-name-extended', 'foo', 'bar']
        with mock.patch('psutil._psplatform.Process.name', return_value=name):
            with mock.patch('psutil._psplatform.Process.cmdline', return_value=cmdline):
                p = psutil.Process()
                self.assertEqual(p.name(), 'long-program-name-extended')

    def test_name_long_cmdline_ad_exc(self):
        if False:
            return 10
        name = 'long-program-name'
        with mock.patch('psutil._psplatform.Process.name', return_value=name):
            with mock.patch('psutil._psplatform.Process.cmdline', side_effect=psutil.AccessDenied(0, '')):
                p = psutil.Process()
                self.assertEqual(p.name(), 'long-program-name')

    def test_name_long_cmdline_nsp_exc(self):
        if False:
            for i in range(10):
                print('nop')
        name = 'long-program-name'
        with mock.patch('psutil._psplatform.Process.name', return_value=name):
            with mock.patch('psutil._psplatform.Process.cmdline', side_effect=psutil.NoSuchProcess(0, '')):
                p = psutil.Process()
                self.assertRaises(psutil.NoSuchProcess, p.name)

    @unittest.skipIf(MACOS or BSD, 'ps -o start not available')
    def test_create_time(self):
        if False:
            print('Hello World!')
        time_ps = ps('start', self.pid)
        time_psutil = psutil.Process(self.pid).create_time()
        time_psutil_tstamp = datetime.datetime.fromtimestamp(time_psutil).strftime('%H:%M:%S')
        round_time_psutil = round(time_psutil)
        round_time_psutil_tstamp = datetime.datetime.fromtimestamp(round_time_psutil).strftime('%H:%M:%S')
        self.assertIn(time_ps, [time_psutil_tstamp, round_time_psutil_tstamp])

    def test_exe(self):
        if False:
            return 10
        ps_pathname = ps_name(self.pid)
        psutil_pathname = psutil.Process(self.pid).exe()
        try:
            self.assertEqual(ps_pathname, psutil_pathname)
        except AssertionError:
            adjusted_ps_pathname = ps_pathname[:len(ps_pathname)]
            self.assertEqual(ps_pathname, adjusted_ps_pathname)

    @retry_on_failure()
    def test_cmdline(self):
        if False:
            return 10
        ps_cmdline = ps_args(self.pid)
        psutil_cmdline = ' '.join(psutil.Process(self.pid).cmdline())
        self.assertEqual(ps_cmdline, psutil_cmdline)

    @unittest.skipIf(SUNOS, 'not reliable on SUNOS')
    @unittest.skipIf(AIX, 'not reliable on AIX')
    def test_nice(self):
        if False:
            print('Hello World!')
        ps_nice = ps('nice', self.pid)
        psutil_nice = psutil.Process().nice()
        self.assertEqual(ps_nice, psutil_nice)

@unittest.skipIf(not POSIX, 'POSIX only')
class TestSystemAPIs(PsutilTestCase):
    """Test some system APIs."""

    @retry_on_failure()
    def test_pids(self):
        if False:
            for i in range(10):
                print('nop')
        pids_ps = sorted(ps('pid'))
        pids_psutil = psutil.pids()
        if MACOS or (OPENBSD and 0 not in pids_ps):
            pids_ps.insert(0, 0)
        if len(pids_ps) - len(pids_psutil) > 1:
            difference = [x for x in pids_psutil if x not in pids_ps] + [x for x in pids_ps if x not in pids_psutil]
            raise self.fail('difference: ' + str(difference))

    @unittest.skipIf(SUNOS, 'unreliable on SUNOS')
    @unittest.skipIf(not which('ifconfig'), 'no ifconfig cmd')
    @unittest.skipIf(not HAS_NET_IO_COUNTERS, 'not supported')
    def test_nic_names(self):
        if False:
            while True:
                i = 10
        output = sh('ifconfig -a')
        for nic in psutil.net_io_counters(pernic=True):
            for line in output.split():
                if line.startswith(nic):
                    break
            else:
                raise self.fail("couldn't find %s nic in 'ifconfig -a' output\n%s" % (nic, output))

    @retry_on_failure()
    def test_users(self):
        if False:
            for i in range(10):
                print('nop')
        out = sh('who -u')
        if not out.strip():
            raise self.skipTest('no users on this system')
        lines = out.split('\n')
        users = [x.split()[0] for x in lines]
        terminals = [x.split()[1] for x in lines]
        self.assertEqual(len(users), len(psutil.users()))
        with self.subTest(psutil=psutil.users(), who=out):
            for (idx, u) in enumerate(psutil.users()):
                self.assertEqual(u.name, users[idx])
                self.assertEqual(u.terminal, terminals[idx])
                if u.pid is not None:
                    psutil.Process(u.pid)

    @retry_on_failure()
    def test_users_started(self):
        if False:
            print('Hello World!')
        out = sh('who -u')
        if not out.strip():
            raise self.skipTest('no users on this system')
        tstamp = None
        started = re.findall('\\d\\d\\d\\d-\\d\\d-\\d\\d \\d\\d:\\d\\d', out)
        if started:
            tstamp = '%Y-%m-%d %H:%M'
        else:
            started = re.findall('[A-Z][a-z][a-z] \\d\\d \\d\\d:\\d\\d', out)
            if started:
                tstamp = '%b %d %H:%M'
            else:
                started = re.findall('[A-Z][a-z][a-z] \\d\\d', out)
                if started:
                    tstamp = '%b %d'
                else:
                    started = re.findall('[a-z][a-z][a-z] \\d\\d', out)
                    if started:
                        tstamp = '%b %d'
                        started = [x.capitalize() for x in started]
        if not tstamp:
            raise unittest.SkipTest('cannot interpret tstamp in who output\n%s' % out)
        with self.subTest(psutil=psutil.users(), who=out):
            for (idx, u) in enumerate(psutil.users()):
                psutil_value = datetime.datetime.fromtimestamp(u.started).strftime(tstamp)
                self.assertEqual(psutil_value, started[idx])

    def test_pid_exists_let_raise(self):
        if False:
            for i in range(10):
                print('nop')
        with mock.patch('psutil._psposix.os.kill', side_effect=OSError(errno.EBADF, '')) as m:
            self.assertRaises(OSError, psutil._psposix.pid_exists, os.getpid())
            assert m.called

    def test_os_waitpid_let_raise(self):
        if False:
            return 10
        with mock.patch('psutil._psposix.os.waitpid', side_effect=OSError(errno.EBADF, '')) as m:
            self.assertRaises(OSError, psutil._psposix.wait_pid, os.getpid())
            assert m.called

    def test_os_waitpid_eintr(self):
        if False:
            while True:
                i = 10
        with mock.patch('psutil._psposix.os.waitpid', side_effect=OSError(errno.EINTR, '')) as m:
            self.assertRaises(psutil._psposix.TimeoutExpired, psutil._psposix.wait_pid, os.getpid(), timeout=0.01)
            assert m.called

    def test_os_waitpid_bad_ret_status(self):
        if False:
            for i in range(10):
                print('nop')
        with mock.patch('psutil._psposix.os.waitpid', return_value=(1, -1)) as m:
            self.assertRaises(ValueError, psutil._psposix.wait_pid, os.getpid())
            assert m.called

    @unittest.skipIf(AIX, 'unreliable on AIX')
    @retry_on_failure()
    def test_disk_usage(self):
        if False:
            while True:
                i = 10

        def df(device):
            if False:
                i = 10
                return i + 15
            try:
                out = sh('df -k %s' % device).strip()
            except RuntimeError as err:
                if 'device busy' in str(err).lower():
                    raise self.skipTest('df returned EBUSY')
                raise
            line = out.split('\n')[1]
            fields = line.split()
            total = int(fields[1]) * 1024
            used = int(fields[2]) * 1024
            free = int(fields[3]) * 1024
            percent = float(fields[4].replace('%', ''))
            return (total, used, free, percent)
        tolerance = 4 * 1024 * 1024
        for part in psutil.disk_partitions(all=False):
            usage = psutil.disk_usage(part.mountpoint)
            try:
                (total, used, free, percent) = df(part.device)
            except RuntimeError as err:
                err = str(err).lower()
                if 'no such file or directory' in err or 'raw devices not supported' in err or 'permission denied' in err:
                    continue
                raise
            else:
                self.assertAlmostEqual(usage.total, total, delta=tolerance)
                self.assertAlmostEqual(usage.used, used, delta=tolerance)
                self.assertAlmostEqual(usage.free, free, delta=tolerance)
                self.assertAlmostEqual(usage.percent, percent, delta=1)

@unittest.skipIf(not POSIX, 'POSIX only')
class TestMisc(PsutilTestCase):

    def test_getpagesize(self):
        if False:
            for i in range(10):
                print('nop')
        pagesize = getpagesize()
        self.assertGreater(pagesize, 0)
        self.assertEqual(pagesize, resource.getpagesize())
        self.assertEqual(pagesize, mmap.PAGESIZE)
if __name__ == '__main__':
    from psutil.tests.runner import run_from_name
    run_from_name(__file__)