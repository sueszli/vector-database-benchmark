__author__ = 'Serg Brester'
__copyright__ = 'Copyright (c) 2014- Serg G. Brester (sebres), 2008- Fail2Ban Contributors'
__license__ = 'GPL'
import fileinput
import os
import re
import sys
import time
import signal
import unittest
from os.path import join as pjoin, isdir, isfile, exists, dirname
from functools import wraps
from threading import Thread
from ..client import fail2banclient, fail2banserver, fail2bancmdline
from ..client.fail2bancmdline import Fail2banCmdLine
from ..client.fail2banclient import exec_command_line as _exec_client, CSocket, VisualWait
from ..client.fail2banserver import Fail2banServer, exec_command_line as _exec_server
from .. import protocol
from ..server import server
from ..server.mytime import MyTime
from ..server.utils import Utils
from .utils import LogCaptureTestCase, logSys as DefLogSys, with_tmpdir, shutil, logging, STOCK, CONFIG_DIR as STOCK_CONF_DIR, TEST_NOW, tearDownMyTime
from ..helpers import getLogger
logSys = getLogger(__name__)
CLIENT = 'fail2ban-client'
SERVER = 'fail2ban-server'
BIN = dirname(Fail2banServer.getServerPath())
MAX_WAITTIME = unittest.F2B.maxWaitTime(unittest.F2B.MAX_WAITTIME)
MID_WAITTIME = unittest.F2B.maxWaitTime(unittest.F2B.MID_WAITTIME)
fail2bancmdline.MAX_WAITTIME = MAX_WAITTIME - 1
fail2bancmdline.logSys = fail2banclient.logSys = fail2banserver.logSys = logSys
SRV_DEF_LOGTARGET = server.DEF_LOGTARGET
SRV_DEF_LOGLEVEL = server.DEF_LOGLEVEL

def _test_output(*args):
    if False:
        print('Hello World!')
    logSys.info(args[0])
fail2bancmdline.output = fail2banclient.output = fail2banserver.output = protocol.output = _test_output

def _time_shift(shift):
    if False:
        print('Hello World!')
    logSys.debug('===>>> time shift + %s min', shift)
    MyTime.setTime(MyTime.time() + shift * 60)
Observers = server.Observers

def _observer_wait_idle():
    if False:
        i = 10
        return i + 15
    'Helper to wait observer becomes idle'
    if Observers.Main is not None:
        Observers.Main.wait_empty(MID_WAITTIME)
        Observers.Main.wait_idle(MID_WAITTIME / 5)

def _observer_wait_before_incrban(cond, timeout=MID_WAITTIME):
    if False:
        return 10
    'Helper to block observer before increase bantime until some condition gets true'
    if Observers.Main is not None:
        _obs_banFound = Observers.Main.banFound

        def _banFound(*args, **kwargs):
            if False:
                print('Hello World!')
            Observers.Main.banFound = _obs_banFound
            logSys.debug('  [Observer::banFound] *** observer blocked for test')
            Utils.wait_for(cond, timeout)
            logSys.debug('  [Observer::banFound] +++ observer runs again')
            _obs_banFound(*args, **kwargs)
        Observers.Main.banFound = _banFound

class ExitException(fail2bancmdline.ExitException):
    """Exception upon a normal exit"""
    pass

class FailExitException(fail2bancmdline.ExitException):
    """Exception upon abnormal exit"""
    pass
SUCCESS = ExitException
FAILED = FailExitException
INTERACT = []

def _test_input_command(*args):
    if False:
        while True:
            i = 10
    if len(INTERACT):
        return INTERACT.pop(0)
    else:
        return 'exit'
fail2banclient.input_command = _test_input_command
fail2bancmdline.PRODUCTION = fail2banserver.PRODUCTION = False
_out_file = LogCaptureTestCase.dumpFile

def _write_file(fn, mode, *lines):
    if False:
        return 10
    f = open(fn, mode)
    f.write('\n'.join(lines) + ('\n' if lines else ''))
    f.close()

def _read_file(fn):
    if False:
        while True:
            i = 10
    f = None
    try:
        f = open(fn)
        return f.read()
    finally:
        if f is not None:
            f.close()

def _start_params(tmp, use_stock=False, use_stock_cfg=None, logtarget='/dev/null', db=':memory:', f2b_local=(), jails=('',), create_before_start=None):
    if False:
        for i in range(10):
            print('nop')
    cfg = pjoin(tmp, 'config')
    if db == 'auto':
        db = pjoin(tmp, 'f2b-db.sqlite3')
    j_conf = 'jail.conf'
    if use_stock and STOCK:

        def ig_dirs(dir, files):
            if False:
                print('Hello World!')
            "Filters list of 'files' to contain only directories (under dir)"
            return [f for f in files if isdir(pjoin(dir, f))]
        shutil.copytree(STOCK_CONF_DIR, cfg, ignore=ig_dirs)
        if use_stock_cfg is None:
            use_stock_cfg = ('action.d', 'filter.d')
        r = re.compile('^dbfile\\s*=')
        for line in fileinput.input(pjoin(cfg, 'fail2ban.conf'), inplace=True):
            line = line.rstrip('\n')
            if r.match(line):
                line = 'dbfile = :memory:'
            print(line)
        r = re.compile('^backend\\s*=')
        for line in fileinput.input(pjoin(cfg, 'jail.conf'), inplace=True):
            line = line.rstrip('\n')
            if r.match(line):
                line = 'backend = polling'
            print(line)
        j_conf = 'jail.local' if jails else ''
    else:
        os.mkdir(cfg)
        _write_file(pjoin(cfg, 'fail2ban.conf'), 'w', '[Definition]', 'loglevel = INFO', 'logtarget = ' + logtarget.replace('%', '%%'), 'syslogsocket = auto', 'socket = ' + pjoin(tmp, 'f2b.sock'), 'pidfile = ' + pjoin(tmp, 'f2b.pid'), 'backend = polling', 'dbfile = ' + db, 'dbmaxmatches = 100', 'dbpurgeage = 1d', '')
    if j_conf:
        _write_file(pjoin(cfg, j_conf), 'w', *('[INCLUDES]', '', '[DEFAULT]', 'tmp = ' + tmp, '') + jails)
    if f2b_local:
        _write_file(pjoin(cfg, 'fail2ban.local'), 'w', *f2b_local)
    if unittest.F2B.log_level < logging.DEBUG:
        _out_file(pjoin(cfg, 'fail2ban.conf'))
        _out_file(pjoin(cfg, 'jail.conf'))
        if f2b_local:
            _out_file(pjoin(cfg, 'fail2ban.local'))
        if j_conf and j_conf != 'jail.conf':
            _out_file(pjoin(cfg, j_conf))
    if use_stock_cfg and STOCK:
        for n in use_stock_cfg:
            os.symlink(os.path.abspath(pjoin(STOCK_CONF_DIR, n)), pjoin(cfg, n))
    if create_before_start:
        for n in create_before_start:
            _write_file(n % {'tmp': tmp}, 'w')
    (vvv, llev) = ((), 'INFO')
    if unittest.F2B.log_level < logging.INFO:
        llev = str(unittest.F2B.log_level)
        if unittest.F2B.verbosity > 1:
            vvv = ('-' + 'v' * unittest.F2B.verbosity,)
    llev = vvv + ('--loglevel', llev)
    return ('-c', cfg, '-s', pjoin(tmp, 'f2b.sock'), '-p', pjoin(tmp, 'f2b.pid'), '--logtarget', logtarget) + llev + ('--syslogsocket', 'auto', '--timeout', str(fail2bancmdline.MAX_WAITTIME))

def _inherited_log(startparams):
    if False:
        return 10
    try:
        return startparams[startparams.index('--logtarget') + 1] == 'INHERITED'
    except ValueError:
        return False

def _get_pid_from_file(pidfile):
    if False:
        for i in range(10):
            print('nop')
    pid = None
    try:
        pid = _read_file(pidfile)
        pid = re.match('\\S+', pid).group()
        return int(pid)
    except Exception as e:
        logSys.debug(e)
    return pid

def _kill_srv(pidfile):
    if False:
        for i in range(10):
            print('nop')
    logSys.debug('cleanup: %r', (pidfile, isdir(pidfile)))
    if isdir(pidfile):
        piddir = pidfile
        pidfile = pjoin(piddir, 'f2b.pid')
        if not isfile(pidfile):
            pidfile = pjoin(piddir, 'fail2ban.pid')
    if unittest.F2B.log_level < logging.DEBUG:
        logfile = pjoin(piddir, 'f2b.log')
        if isfile(logfile):
            _out_file(logfile)
        else:
            logSys.log(5, 'no logfile %r', logfile)
    if not isfile(pidfile):
        logSys.debug('cleanup: no pidfile for %r', piddir)
        return True
    logSys.debug('cleanup pidfile: %r', pidfile)
    pid = _get_pid_from_file(pidfile)
    if pid is None:
        return False
    try:
        logSys.debug('cleanup pid: %r', pid)
        if pid <= 0 or pid == os.getpid():
            raise ValueError('pid %s of %s is invalid' % (pid, pidfile))
        if not Utils.pid_exists(pid):
            return True
        os.kill(pid, signal.SIGTERM)
        if not Utils.wait_for(lambda : not Utils.pid_exists(pid), 1):
            os.kill(pid, signal.SIGKILL)
        logSys.debug('cleanup: kill ready')
        return not Utils.pid_exists(pid)
    except Exception as e:
        logSys.exception(e)
    return True

def with_kill_srv(f):
    if False:
        return 10
    'Helper to decorate tests which receive in the last argument tmpdir to pass to kill_srv\n\n\tTo be used in tandem with @with_tmpdir\n\t'

    @wraps(f)
    def wrapper(self, *args):
        if False:
            for i in range(10):
                print('nop')
        pidfile = args[-1]
        try:
            return f(self, *args)
        finally:
            _kill_srv(pidfile)
    return wrapper

def with_foreground_server_thread(startextra={}):
    if False:
        while True:
            i = 10
    'Helper to decorate tests uses foreground server (as thread), started directly in test-cases\n\n\tTo be used only in subclasses\n\t'

    def _deco_wrapper(f):
        if False:
            return 10

        @with_tmpdir
        @wraps(f)
        def wrapper(self, tmp, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            th = None
            phase = dict()
            try:
                startparams = _start_params(tmp, logtarget='INHERITED', **startextra)
                th = Thread(name='_TestCaseWorker', target=self._testStartForeground, args=(tmp, startparams, phase))
                th.daemon = True
                th.start()

                def _stopAndWaitForServerEnd(code=(SUCCESS, FAILED)):
                    if False:
                        for i in range(10):
                            print('nop')
                    tearDownMyTime()
                    if not phase.get('end', None) and (not os.path.exists(pjoin(tmp, 'f2b.pid'))):
                        Utils.wait_for(lambda : phase.get('end', None) is not None, MID_WAITTIME)
                    if not phase.get('end', None):
                        self.execCmd(code, startparams, 'stop')
                        Utils.wait_for(lambda : phase.get('end', None) is not None, MAX_WAITTIME)
                        self.assertTrue(phase.get('end', None))
                        self.assertLogged('Shutdown successful', 'Exiting Fail2ban', all=True, wait=MAX_WAITTIME)
                    self.stopAndWaitForServerEnd = lambda *args, **kwargs: None
                self.stopAndWaitForServerEnd = _stopAndWaitForServerEnd
                Utils.wait_for(lambda : phase.get('start', None) is not None, MAX_WAITTIME)
                self.assertTrue(phase.get('start', None))
                self._wait_for_srv(tmp, True, startparams=startparams, phase=phase)
                DefLogSys.info('=== within server: begin ===')
                self.pruneLog()
                return f(self, tmp, startparams, *args, **kwargs)
            except Exception as e:
                print('=== Catch an exception: %s' % e)
                log = self.getLog()
                if log:
                    print('=== Error of server, log: ===\n%s===' % log)
                    self.pruneLog()
                raise
            finally:
                if th:
                    DefLogSys.info('=== within server: end.  ===')
                    self.pruneLog()
                    self.stopAndWaitForServerEnd()
                    if phase.get('end', None):
                        th.join()
                tearDownMyTime()
        return wrapper
    return _deco_wrapper

class Fail2banClientServerBase(LogCaptureTestCase):
    _orig_exit = Fail2banCmdLine._exit

    def _setLogLevel(self, *args, **kwargs):
        if False:
            print('Hello World!')
        pass

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        'Call before every test case.'
        LogCaptureTestCase.setUp(self)
        server.DEF_LOGTARGET = 'INHERITED'
        server.DEF_LOGLEVEL = DefLogSys.level
        Fail2banCmdLine._exit = staticmethod(self._test_exit)

    def tearDown(self):
        if False:
            return 10
        'Call after every test case.'
        Fail2banCmdLine._exit = self._orig_exit
        server.DEF_LOGTARGET = SRV_DEF_LOGTARGET
        server.DEF_LOGLEVEL = SRV_DEF_LOGLEVEL
        LogCaptureTestCase.tearDown(self)
        tearDownMyTime()

    @staticmethod
    def _test_exit(code=0):
        if False:
            return 10
        if code == 0:
            raise ExitException()
        else:
            raise FailExitException()

    def _wait_for_srv(self, tmp, ready=True, startparams=None, phase=None):
        if False:
            for i in range(10):
                print('nop')
        if not phase:
            phase = {}
        try:
            sock = pjoin(tmp, 'f2b.sock')
            ret = Utils.wait_for(lambda : phase.get('end') or exists(sock), MAX_WAITTIME)
            if not ret or phase.get('end'):
                raise Exception('Unexpected: Socket file does not exists.\nStart failed: %r' % (startparams,))
            if ready:
                ret = Utils.wait_for(lambda : 'Server ready' in self.getLog(), MAX_WAITTIME)
                if not ret:
                    raise Exception('Unexpected: Server ready was not found, phase %r.\nStart failed: %r' % (phase, startparams))
        except:
            if _inherited_log(startparams):
                print('=== Error by wait fot server, log: ===\n%s===' % self.getLog())
                self.pruneLog()
            log = pjoin(tmp, 'f2b.log')
            if isfile(log):
                _out_file(log)
            elif not _inherited_log(startparams):
                logSys.debug('No log file %s to examine details of error', log)
            raise

    def execCmd(self, exitType, startparams, *args):
        if False:
            print('Hello World!')
        self.assertRaises(exitType, self.exec_command_line[0], self.exec_command_line[1:] + startparams + args)

    def execCmdDirect(self, startparams, *args):
        if False:
            while True:
                i = 10
        sock = startparams[startparams.index('-s') + 1]
        s = CSocket(sock)
        try:
            return s.send(args)
        finally:
            s.close()

    def _testStartForeground(self, tmp, startparams, phase):
        if False:
            return 10
        logSys.debug('start of test worker')
        phase['start'] = True
        try:
            self.execCmd(SUCCESS, ('-f',) + startparams, 'start')
        finally:
            phase['start'] = False
            phase['end'] = True
            logSys.debug('end of test worker')

    @with_foreground_server_thread(startextra={'f2b_local': ('[Thread]', 'stacksize = 128')})
    def testStartForeground(self, tmp, startparams):
        if False:
            while True:
                i = 10
        self.pruneLog()
        self.execCmd(SUCCESS, startparams, 'get', 'thread')
        self.assertLogged("{'stacksize': 128}")
        self.execCmd(SUCCESS, startparams, 'ping')
        self.execCmd(FAILED, startparams, '~~unknown~cmd~failed~~')
        self.execCmd(SUCCESS, startparams, 'echo', 'TEST-ECHO')

    @with_tmpdir
    @with_kill_srv
    def testStartFailsInForeground(self, tmp):
        if False:
            print('Hello World!')
        if not server.Fail2BanDb:
            raise unittest.SkipTest('Skip test because no database')
        dbname = pjoin(tmp, 'tmp.db')
        db = server.Fail2BanDb(dbname)
        cur = db._db.cursor()
        cur.executescript('UPDATE fail2banDb SET version = 555')
        cur.close()
        startparams = _start_params(tmp, db=dbname, logtarget='INHERITED')
        phase = {'stop': True}

        def _stopTimeout(startparams, phase):
            if False:
                for i in range(10):
                    print('nop')
            if not Utils.wait_for(lambda : not phase['stop'], MAX_WAITTIME):
                self.execCmdDirect(startparams, 'stop')
        th = Thread(name='_TestCaseWorker', target=_stopTimeout, args=(startparams, phase))
        th.start()
        try:
            self.execCmd(FAILED, ('-f',) + startparams, 'start')
        finally:
            phase['stop'] = False
            th.join()
        self.assertLogged('Attempt to travel to future version of database', 'Exit with code 255', all=True)

class Fail2banClientTest(Fail2banClientServerBase):
    exec_command_line = (_exec_client, CLIENT)

    def testConsistency(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(isfile(pjoin(BIN, CLIENT)))
        self.assertTrue(isfile(pjoin(BIN, SERVER)))

    def testClientUsage(self):
        if False:
            i = 10
            return i + 15
        self.execCmd(SUCCESS, (), '-h')
        self.assertLogged('Usage: ' + CLIENT)
        self.assertLogged('Report bugs to ')
        self.pruneLog()
        self.execCmd(SUCCESS, (), '-V')
        self.assertLogged(fail2bancmdline.normVersion())
        self.pruneLog()
        self.execCmd(SUCCESS, (), '-vq', '--version')
        self.assertLogged('Fail2Ban v' + fail2bancmdline.version)
        self.pruneLog()
        self.execCmd(SUCCESS, (), '--str2sec', '1d12h30m')
        self.assertLogged('131400')

    @with_tmpdir
    def testClientDump(self, tmp):
        if False:
            while True:
                i = 10
        startparams = _start_params(tmp, True)
        self.execCmd(SUCCESS, startparams, '-vvd')
        self.assertLogged('Loading files')
        self.assertLogged("['set', 'logtarget',")
        self.pruneLog()
        self.execCmd(SUCCESS, startparams, '--dp')
        self.assertLogged("['set', 'logtarget',")

    @with_tmpdir
    @with_kill_srv
    def testClientStartBackgroundInside(self, tmp):
        if False:
            return 10
        startparams = _start_params(tmp, True)
        self.execCmd(SUCCESS, ('-b',) + startparams, 'start')
        self._wait_for_srv(tmp, True, startparams=startparams)
        self.assertLogged('Server ready')
        self.assertLogged('Exit with code 0')
        try:
            self.execCmd(SUCCESS, startparams, 'echo', 'TEST-ECHO')
            self.execCmd(FAILED, startparams, '~~unknown~cmd~failed~~')
            self.pruneLog()
            self.execCmd(FAILED, ('-b',) + startparams, 'start')
            self.assertLogged('Server already running')
        finally:
            self.pruneLog()
            self.execCmd(SUCCESS, startparams, 'stop')
            self.assertLogged('Shutdown successful')
            self.assertLogged('Exit with code 0')
        self.pruneLog()
        self.execCmd(FAILED, startparams, 'stop')
        self.assertLogged('Failed to access socket path')
        self.assertLogged('Is fail2ban running?')

    @with_tmpdir
    @with_kill_srv
    def testClientStartBackgroundCall(self, tmp):
        if False:
            i = 10
            return i + 15
        global INTERACT
        startparams = _start_params(tmp, logtarget=pjoin(tmp, 'f2b.log'))
        if unittest.F2B.fast:
            self.execCmd(SUCCESS, startparams + ('start',))
        else:
            cmd = (sys.executable, pjoin(BIN, CLIENT))
            logSys.debug('Start %s ...', cmd)
            cmd = cmd + startparams + ('--async', 'start')
            ret = Utils.executeCmd(cmd, timeout=MAX_WAITTIME, shell=False, output=True)
            self.assertTrue(len(ret) and ret[0])
            self._wait_for_srv(tmp, True, startparams=cmd)
        self.assertLogged('Server ready')
        self.pruneLog()
        try:
            self.execCmd(SUCCESS, startparams, 'echo', 'TEST-ECHO')
            self.assertLogged('TEST-ECHO')
            self.assertLogged('Exit with code 0')
            self.pruneLog()
            self.execCmd(SUCCESS, startparams, 'ping', '0.1')
            self.assertLogged('Server replied: pong')
            self.pruneLog()
            pid = _get_pid_from_file(pjoin(tmp, 'f2b.pid'))
            try:
                os.kill(pid, signal.SIGSTOP)
                time.sleep(Utils.DEFAULT_SHORT_INTERVAL)
                self.execCmd(FAILED, startparams, 'ping', '1e-10')
            finally:
                os.kill(pid, signal.SIGCONT)
            self.assertLogged('timed out')
            self.pruneLog()
            try:
                import readline
            except ImportError as e:
                raise unittest.SkipTest('Skip test because of import error: %s' % e)
            INTERACT += ['echo INTERACT-ECHO', 'status', 'exit']
            self.execCmd(SUCCESS, startparams, '-i')
            self.assertLogged('INTERACT-ECHO')
            self.assertLogged('Status', 'Number of jail:')
            self.assertLogged('Exit with code 0')
            self.pruneLog()
            INTERACT += ['reload', 'restart', 'exit']
            self.execCmd(SUCCESS, startparams, '-i')
            self.assertLogged('Reading config files:')
            self.assertLogged('Shutdown successful')
            self.assertLogged('Server ready')
            self.assertLogged('Exit with code 0')
            self.pruneLog()
            INTERACT += ['reload ~~unknown~jail~fail~~', 'exit']
            self.execCmd(SUCCESS, startparams, '-i')
            self.assertLogged("Failed during configuration: No section: '~~unknown~jail~fail~~'")
            self.pruneLog()
            self.execCmd(FAILED, startparams, 'reload', '~~unknown~jail~fail~~')
            self.assertLogged("Failed during configuration: No section: '~~unknown~jail~fail~~'")
            self.assertLogged('Exit with code 255')
            self.pruneLog()
        finally:
            self.pruneLog()
            self.execCmd(SUCCESS, startparams, 'stop')
            self.assertLogged('Shutdown successful')
            self.assertLogged('Exit with code 0')

    @with_tmpdir
    @with_kill_srv
    def testClientFailStart(self, tmp):
        if False:
            i = 10
            return i + 15
        startparams = _start_params(tmp, logtarget='INHERITED')
        self.execCmd(FAILED, (), '--async', '-c', pjoin(tmp, 'miss'), 'start')
        self.assertLogged('Base configuration directory ' + pjoin(tmp, 'miss') + ' does not exist')
        self.pruneLog()
        self.execCmd(FAILED, (), '-c', pjoin(tmp, 'config'), '-s', pjoin(tmp, 'f2b.sock'), 'reload')
        self.assertLogged('Could not find server')
        self.pruneLog()
        open(pjoin(tmp, 'f2b.sock'), 'a').close()
        self.execCmd(FAILED, (), '--async', '-c', pjoin(tmp, 'config'), '-s', pjoin(tmp, 'f2b.sock'), 'start')
        self.assertLogged('Fail2ban seems to be in unexpected state (not running but the socket exists)')
        self.pruneLog()
        os.remove(pjoin(tmp, 'f2b.sock'))
        self.execCmd(FAILED, (), '-s')
        self.assertLogged('Usage: ')
        self.pruneLog()

    @with_tmpdir
    def testClientFailCommands(self, tmp):
        if False:
            while True:
                i = 10
        startparams = _start_params(tmp, logtarget='INHERITED')
        self.execCmd(FAILED, startparams, 'reload', 'jail')
        self.assertLogged('Could not find server')
        self.pruneLog()
        self.execCmd(FAILED, startparams, '--async', 'reload', '--xxx', 'jail')
        self.assertLogged('Unexpected argument(s) for reload:')
        self.pruneLog()

    def testVisualWait(self):
        if False:
            while True:
                i = 10
        sleeptime = 0.035
        for verbose in (2, 0):
            cntr = 15
            with VisualWait(verbose, 5) as vis:
                while cntr:
                    vis.heartbeat()
                    if verbose and (not unittest.F2B.fast):
                        time.sleep(sleeptime)
                    cntr -= 1

class Fail2banServerTest(Fail2banClientServerBase):
    exec_command_line = (_exec_server, SERVER)

    def testServerUsage(self):
        if False:
            for i in range(10):
                print('nop')
        self.execCmd(SUCCESS, (), '-h')
        self.assertLogged('Usage: ' + SERVER)
        self.assertLogged('Report bugs to ')

    @with_tmpdir
    @with_kill_srv
    def testServerStartBackground(self, tmp):
        if False:
            return 10
        startparams = _start_params(tmp, logtarget=pjoin(tmp, 'f2b.log'))
        cmd = (sys.executable, pjoin(BIN, SERVER))
        logSys.debug('Start %s ...', cmd)
        cmd = cmd + startparams + ('-b',)
        ret = Utils.executeCmd(cmd, timeout=MAX_WAITTIME, shell=False, output=True)
        self.assertTrue(len(ret) and ret[0])
        self._wait_for_srv(tmp, True, startparams=cmd)
        self.assertLogged('Server ready')
        self.pruneLog()
        try:
            self.execCmd(SUCCESS, startparams, 'echo', 'TEST-ECHO')
            self.execCmd(FAILED, startparams, '~~unknown~cmd~failed~~')
        finally:
            self.pruneLog()
            self.execCmd(SUCCESS, startparams, 'stop')
            self.assertLogged('Shutdown successful')
            self.assertLogged('Exit with code 0')

    @with_tmpdir
    @with_kill_srv
    def testServerFailStart(self, tmp):
        if False:
            while True:
                i = 10
        startparams = _start_params(tmp, logtarget='INHERITED')
        self.execCmd(FAILED, (), '-c', pjoin(tmp, 'miss'))
        self.assertLogged('Base configuration directory ' + pjoin(tmp, 'miss') + ' does not exist')
        self.pruneLog()
        open(pjoin(tmp, 'f2b.sock'), 'a').close()
        self.execCmd(FAILED, (), '-c', pjoin(tmp, 'config'), '-s', pjoin(tmp, 'f2b.sock'))
        self.assertLogged('Fail2ban seems to be in unexpected state (not running but the socket exists)')
        self.pruneLog()
        os.remove(pjoin(tmp, 'f2b.sock'))

    @with_tmpdir
    @with_kill_srv
    def testServerTestFailStart(self, tmp):
        if False:
            return 10
        startparams = _start_params(tmp, logtarget='INHERITED')
        cfg = pjoin(tmp, 'config')
        self.pruneLog('[test-phase 0]')
        self.execCmd(SUCCESS, startparams, '--test')
        self.assertLogged('OK: configuration test is successful')
        _write_file(pjoin(cfg, 'jail.conf'), 'a', '', '[broken-jail]', '', 'filter = broken-jail-filter', 'enabled = true')
        self.pruneLog('[test-phase 0a]')
        self.execCmd(FAILED, startparams, '--test')
        self.assertLogged("Unable to read the filter 'broken-jail-filter'", "Errors in jail 'broken-jail'.", 'ERROR: test configuration failed', all=True)
        self.pruneLog('[test-phase 0b]')
        self.execCmd(FAILED, startparams, '-t', 'start')
        self.assertLogged("Unable to read the filter 'broken-jail-filter'", "Errors in jail 'broken-jail'.", 'ERROR: test configuration failed', all=True)

    @with_tmpdir
    def testKillAfterStart(self, tmp):
        if False:
            return 10
        try:
            startparams = _start_params(tmp, logtarget=pjoin(tmp, 'f2b.log[format="SRV: %(relativeCreated)3d | %(message)s", datetime=off]'))
            cmd = (sys.executable, pjoin(BIN, SERVER))
            logSys.debug('Start %s ...', cmd)
            cmd = cmd + startparams + ('-b',)
            ret = Utils.executeCmd(cmd, timeout=MAX_WAITTIME, shell=False, output=True)
            self.assertTrue(len(ret) and ret[0])
            self._wait_for_srv(tmp, True, startparams=cmd)
            self.assertLogged('Server ready')
            self.pruneLog()
            logSys.debug('Kill server ... %s', tmp)
        finally:
            self.assertTrue(_kill_srv(tmp))
        Utils.wait_for(lambda : not isfile(pjoin(tmp, 'f2b.pid')), MAX_WAITTIME)
        self.assertFalse(isfile(pjoin(tmp, 'f2b.pid')))
        self.assertLogged('cleanup: kill ready')
        self.pruneLog()
        self.assertTrue(_kill_srv(tmp))
        self.assertLogged('cleanup: no pidfile for')

    @with_foreground_server_thread(startextra={'db': 'auto'})
    def testServerReloadTest(self, tmp, startparams):
        if False:
            return 10
        cfg = pjoin(tmp, 'config')
        test1log = pjoin(tmp, 'test1.log')
        test2log = pjoin(tmp, 'test2.log')
        test3log = pjoin(tmp, 'test3.log')
        os.mkdir(pjoin(cfg, 'action.d'))

        def _write_action_cfg(actname='test-action1', allow=True, start='', reload='', ban='', unban='', stop=''):
            if False:
                for i in range(10):
                    print('nop')
            fn = pjoin(cfg, 'action.d', '%s.conf' % actname)
            if not allow:
                os.remove(fn)
                return
            _write_file(fn, 'w', '[DEFAULT]', '_exec_once = 0', '', '[Definition]', 'norestored = %(_exec_once)s', 'restore = ', 'info = ', "_use_flush_ = echo '[%(name)s] %(actname)s: -- flushing IPs'", "actionstart =  echo '[%(name)s] %(actname)s: ** start'", start, "actionreload = echo '[%(name)s] %(actname)s: .. reload'", reload, "actionban =    echo '[%(name)s] %(actname)s: ++ ban <ip> %(restore)s%(info)s'", ban, "actionunban =  echo '[%(name)s] %(actname)s: -- unban <ip>'", unban, "actionstop =   echo '[%(name)s] %(actname)s: __ stop'", stop)
            if unittest.F2B.log_level <= logging.DEBUG:
                _out_file(fn)

        def _write_jail_cfg(enabled=(1, 2), actions=(), backend='polling'):
            if False:
                return 10
            _write_file(pjoin(cfg, 'jail.conf'), 'w', '[INCLUDES]', '', '[DEFAULT]', '', 'usedns = no', 'maxretry = 3', 'findtime = 10m', 'failregex = ^\\s*failure <F-ERRCODE>401|403</F-ERRCODE> from <HOST>', 'datepattern = {^LN-BEG}EPOCH', 'ignoreip = 127.0.0.1/8 ::1', '', '[test-jail1]', 'backend = ' + backend, 'filter =', 'action = ', "         test-action1[name='%(__name__)s']" if 1 in actions else '', "         test-action2[name='%(__name__)s', restore='restored: <restored>', info=', err-code: <F-ERRCODE>']" if 2 in actions else '', "         test-action2[name='%(__name__)s', actname=test-action3, _exec_once=1, restore='restored: <restored>', actionflush=<_use_flush_>]" if 3 in actions else '', 'logpath = ' + test1log, '          ' + test2log if 2 in enabled else '', '          ' + test3log if 2 in enabled else '', 'failregex = ^\\s*failure <F-ERRCODE>401|403</F-ERRCODE> from <HOST>', '            ^\\s*error <F-ERRCODE>401|403</F-ERRCODE> from <HOST>' if 2 in enabled else '', 'enabled = true' if 1 in enabled else '', '', '[test-jail2]', 'backend = ' + backend, 'filter =', 'action = ', "         test-action2[name='%(__name__)s', restore='restored: <restored>', info=', err-code: <F-ERRCODE>']" if 2 in actions else '', "         test-action2[name='%(__name__)s', actname=test-action3, _exec_once=1, restore='restored: <restored>', actionflush=<_use_flush_>]" if 3 in actions else '', 'logpath = ' + test2log, 'enabled = true' if 2 in enabled else '')
            if unittest.F2B.log_level <= logging.DEBUG:
                _out_file(pjoin(cfg, 'jail.conf'))
        _write_action_cfg(actname='test-action1')
        _write_action_cfg(actname='test-action2')
        _write_jail_cfg(enabled=[1], actions=[1, 2, 3])
        _write_file(pjoin(cfg, 'jail.conf'), 'a', '', '[broken-jail]', '', 'filter = broken-jail-filter', 'enabled = true')
        _write_file(test1log, 'w', *(str(int(MyTime.time())) + ' failure 401 from 192.0.2.1: test 1',) * 3)
        _write_file(test2log, 'w')
        _write_file(test3log, 'w')
        self.pruneLog('[test-phase 1a]')
        if unittest.F2B.log_level < logging.DEBUG:
            _out_file(test1log)
        self.execCmd(SUCCESS, startparams, 'reload')
        self.assertLogged('Reload finished.', "1 ticket(s) in 'test-jail1", all=True, wait=MID_WAITTIME)
        self.assertLogged('Added logfile: %r' % test1log)
        self.assertLogged('[test-jail1] Ban 192.0.2.1')
        self.assertLogged("stdout: '[test-jail1] test-action1: ** start'", "stdout: '[test-jail1] test-action2: ** start'", all=True)
        self.assertLogged("stdout: '[test-jail1] test-action2: ++ ban 192.0.2.1 restored: 0, err-code: 401'", "stdout: '[test-jail1] test-action3: ++ ban 192.0.2.1 restored: 0'", all=True, wait=MID_WAITTIME)
        self.assertLogged("Unable to read the filter 'broken-jail-filter'", "Errors in jail 'broken-jail'. Skipping...", "Jail 'broken-jail' skipped, because of wrong configuration", all=True)
        self.pruneLog('[test-phase 1b]')
        _write_jail_cfg(actions=[1, 2])
        if unittest.F2B.log_level < logging.DEBUG:
            _out_file(test1log)
        self.execCmd(SUCCESS, startparams, 'reload')
        self.assertLogged('Reload finished.', wait=MID_WAITTIME)
        self.assertNotLogged('[test-jail1] Unban 192.0.2.1', '[test-jail1] Ban 192.0.2.1', all=True)
        self.assertLogged('Added logfile: %r' % test2log, 'Added logfile: %r' % test3log, all=True)
        self.assertLogged("stdout: '[test-jail1] test-action1: .. reload'", "stdout: '[test-jail1] test-action2: .. reload'", all=True)
        self.assertLogged("Creating new jail 'test-jail2'", "Jail 'test-jail2' started", all=True)
        self.assertLogged("stdout: '[test-jail1] test-action3: -- flushing IPs'", "stdout: '[test-jail1] test-action3: __ stop'", all=True)
        self.assertNotLogged("stdout: '[test-jail1] test-action3: -- unban 192.0.2.1'")
        self.pruneLog('[test-phase 2a]')
        _write_jail_cfg(actions=[1])
        _write_action_cfg(actname='test-action1', start="               echo '[<name>] %s: started.'" % 'test-action1', reload="               echo '[<name>] %s: reloaded.'" % 'test-action1', stop="               echo '[<name>] %s: stopped.'" % 'test-action1')
        self.execCmd(SUCCESS, startparams, 'reload')
        self.assertLogged('Reload finished.', wait=MID_WAITTIME)
        self.assertNotLogged('[test-jail1] Unban 192.0.2.1', '[test-jail1] Ban 192.0.2.1', all=True)
        self.assertNotLogged('Added logfile:')
        self.assertLogged("stdout: '[test-jail1] test-action1: .. reload'", "stdout: '[test-jail1] test-action1: reloaded.'", all=True)
        self.assertLogged("stdout: '[test-jail1] test-action2: -- unban 192.0.2.1'")
        self.assertLogged("stdout: '[test-jail1] test-action2: __ stop'")
        self.assertNotLogged("stdout: '[test-jail1] test-action1: -- unban 192.0.2.1'")
        _write_action_cfg(actname='test-action1', allow=False)
        _write_jail_cfg(actions=[2, 3])
        self.pruneLog('[test-phase 2b]')
        _write_file(test2log, 'a+', *(str(int(MyTime.time())) + '   error 403 from 192.0.2.2: test 2',) * 3 + (str(int(MyTime.time())) + '   error 403 from 192.0.2.3: test 2',) * 3 + (str(int(MyTime.time())) + ' failure 401 from 192.0.2.4: test 2',) * 3 + (str(int(MyTime.time())) + ' failure 401 from 192.0.2.8: test 2',) * 3)
        if unittest.F2B.log_level < logging.DEBUG:
            _out_file(test2log)
        self.assertLogged("2 ticket(s) in 'test-jail2", "5 ticket(s) in 'test-jail1", all=True, wait=MID_WAITTIME)
        self.execCmd(SUCCESS, startparams, 'set', 'test-jail2', 'banip', '192.0.2.9')
        self.assertLogged("3 ticket(s) in 'test-jail2", wait=MID_WAITTIME)
        self.assertLogged('[test-jail1] Ban 192.0.2.2', '[test-jail1] Ban 192.0.2.3', '[test-jail1] Ban 192.0.2.4', '[test-jail1] Ban 192.0.2.8', '[test-jail2] Ban 192.0.2.4', '[test-jail2] Ban 192.0.2.8', '[test-jail2] Ban 192.0.2.9', all=True)
        self.assertNotLogged('[test-jail2] Found 192.0.2.2', '[test-jail2] Ban 192.0.2.2', '[test-jail2] Found 192.0.2.3', '[test-jail2] Ban 192.0.2.3', all=True)
        _observer_wait_idle()
        self.assertSortedEqual(self.execCmdDirect(startparams, 'banned'), (0, [{'test-jail1': ['192.0.2.4', '192.0.2.1', '192.0.2.8', '192.0.2.3', '192.0.2.2']}, {'test-jail2': ['192.0.2.4', '192.0.2.9', '192.0.2.8']}]))
        self.assertSortedEqual(self.execCmdDirect(startparams, 'banned', '192.0.2.1', '192.0.2.4', '192.0.2.222'), (0, [['test-jail1'], ['test-jail1', 'test-jail2'], []]))
        self.assertSortedEqual(self.execCmdDirect(startparams, 'get', 'test-jail1', 'banned')[1], ['192.0.2.4', '192.0.2.1', '192.0.2.8', '192.0.2.3', '192.0.2.2'])
        self.assertSortedEqual(self.execCmdDirect(startparams, 'get', 'test-jail2', 'banned')[1], ['192.0.2.4', '192.0.2.9', '192.0.2.8'])
        self.assertEqual(self.execCmdDirect(startparams, 'get', 'test-jail1', 'banned', '192.0.2.3')[1], 1)
        self.assertEqual(self.execCmdDirect(startparams, 'get', 'test-jail1', 'banned', '192.0.2.9')[1], 0)
        self.assertEqual(self.execCmdDirect(startparams, 'get', 'test-jail1', 'banned', '192.0.2.3', '192.0.2.9')[1], [1, 0])
        self.pruneLog('[test-phase 2c]')
        self.execCmd(SUCCESS, startparams, 'restart', 'test-jail2')
        self.assertLogged('Reload finished.', 'Restore Ban', "3 ticket(s) in 'test-jail2", all=True, wait=MID_WAITTIME)
        self.assertLogged('[test-jail2] Unban 192.0.2.4', '[test-jail2] Unban 192.0.2.8', '[test-jail2] Unban 192.0.2.9', "Jail 'test-jail2' stopped", "Jail 'test-jail2' started", '[test-jail2] Restore Ban 192.0.2.4', '[test-jail2] Restore Ban 192.0.2.8', '[test-jail2] Restore Ban 192.0.2.9', all=True)
        self.assertLogged("stdout: '[test-jail2] test-action2: ++ ban 192.0.2.4 restored: 1, err-code: 401'", "stdout: '[test-jail2] test-action2: ++ ban 192.0.2.8 restored: 1, err-code: 401'", all=True, wait=MID_WAITTIME)
        self.assertNotLogged("stdout: '[test-jail2] test-action3: ++ ban 192.0.2.4 restored: 1'", "stdout: '[test-jail2] test-action3: ++ ban 192.0.2.8 restored: 1'", all=True)
        self.pruneLog('[test-phase 2d]')
        self.execCmd(SUCCESS, startparams, 'set', 'test-jail2', 'banip', '192.0.2.21')
        self.execCmd(SUCCESS, startparams, 'set', 'test-jail2', 'banip', '192.0.2.22')
        self.assertLogged("stdout: '[test-jail2] test-action3: ++ ban 192.0.2.22", "stdout: '[test-jail2] test-action3: ++ ban 192.0.2.22 ", all=True, wait=MID_WAITTIME)
        _observer_wait_idle()
        self.pruneLog('[test-phase 2d.1]')
        self.execCmd(SUCCESS, startparams, 'get', 'test-jail2', 'banip', '\n')
        self.assertLogged('192.0.2.4', '192.0.2.8', '192.0.2.21', '192.0.2.22', all=True, wait=MID_WAITTIME)
        self.pruneLog('[test-phase 2d.2]')
        self.execCmd(SUCCESS, startparams, 'get', 'test-jail1', 'banip')
        self.assertLogged('192.0.2.1', '192.0.2.2', '192.0.2.3', '192.0.2.4', '192.0.2.8', all=True, wait=MID_WAITTIME)
        self.pruneLog('[test-phase 2e]')
        self.execCmd(SUCCESS, startparams, 'restart', '--unban', 'test-jail2')
        self.assertLogged('Reload finished.', "Jail 'test-jail2' started", all=True, wait=MID_WAITTIME)
        self.assertLogged("Jail 'test-jail2' stopped", "Jail 'test-jail2' started", '[test-jail2] Unban 192.0.2.4', '[test-jail2] Unban 192.0.2.8', '[test-jail2] Unban 192.0.2.9', all=True)
        self.assertLogged("stdout: '[test-jail2] test-action2: -- unban 192.0.2.21", "stdout: '[test-jail2] test-action2: -- unban 192.0.2.22'", all=True)
        self.assertLogged("stdout: '[test-jail2] test-action3: -- flushing IPs'")
        self.assertNotLogged("stdout: '[test-jail2] test-action3: -- unban 192.0.2.21'", "stdout: '[test-jail2] test-action3: -- unban 192.0.2.22'", all=True)
        self.assertNotLogged('[test-jail2] Ban 192.0.2.4', '[test-jail2] Ban 192.0.2.8', all=True)
        _write_action_cfg(actname='test-action2', allow=False)
        _write_jail_cfg(actions=[])
        self.pruneLog('[test-phase 3]')
        self.execCmd(SUCCESS, startparams, 'reload', 'test-jail1')
        self.assertLogged('Reload finished.', wait=MID_WAITTIME)
        self.assertLogged("Reload jail 'test-jail1'", "Jail 'test-jail1' reloaded", all=True)
        self.assertNotLogged("Reload jail 'test-jail2'", "Jail 'test-jail2' reloaded", "Jail 'test-jail1' started", all=True)
        self.pruneLog('[test-phase 4]')
        _write_jail_cfg(enabled=[1])
        self.execCmd(SUCCESS, startparams, 'reload')
        self.assertLogged('Reload finished.', wait=MID_WAITTIME)
        self.assertLogged("Reload jail 'test-jail1'")
        self.assertLogged("Stopping jail 'test-jail2'", "Jail 'test-jail2' stopped", all=True)
        self.assertLogged('Removed logfile: %r' % test2log, 'Removed logfile: %r' % test3log, all=True)
        self.pruneLog('[test-phase 5]')
        _write_file(test1log, 'a+', *(str(int(MyTime.time())) + ' failure 401 from 192.0.2.1: test 5',) * 3 + (str(int(MyTime.time())) + '   error 403 from 192.0.2.5: test 5',) * 3 + (str(int(MyTime.time())) + ' failure 401 from 192.0.2.6: test 5',) * 3)
        if unittest.F2B.log_level < logging.DEBUG:
            _out_file(test1log)
        self.assertLogged("6 ticket(s) in 'test-jail1", '[test-jail1] 192.0.2.1 already banned', all=True, wait=MID_WAITTIME)
        self.assertLogged('[test-jail1] Found 192.0.2.1', '[test-jail1] Found 192.0.2.6', '[test-jail1] 192.0.2.1 already banned', '[test-jail1] Ban 192.0.2.6', all=True)
        self.assertNotLogged('[test-jail1] Found 192.0.2.5')
        self.pruneLog('[test-phase 6a]')
        self.execCmd(SUCCESS, startparams, '--async', 'unban', '192.0.2.5', '192.0.2.6')
        self.assertLogged('192.0.2.5 is not banned', '[test-jail1] Unban 192.0.2.6', all=True, wait=MID_WAITTIME)
        self.pruneLog('[test-phase 6b]')
        self.execCmd(SUCCESS, startparams, '--async', 'unban', '192.0.2.2/31')
        self.assertLogged('[test-jail1] Unban 192.0.2.2', '[test-jail1] Unban 192.0.2.3', all=True, wait=MID_WAITTIME)
        self.execCmd(SUCCESS, startparams, '--async', 'unban', '192.0.2.8/31', '192.0.2.100/31')
        self.assertLogged('[test-jail1] Unban 192.0.2.8', '192.0.2.100/31 is not banned', all=True, wait=MID_WAITTIME)
        self.pruneLog('[test-phase 6c]')
        self.execCmd(SUCCESS, startparams, '--async', 'set', 'test-jail1', 'banip', '192.0.2.96/28', '192.0.2.112/28')
        self.assertLogged('[test-jail1] Ban 192.0.2.96/28', '[test-jail1] Ban 192.0.2.112/28', all=True, wait=MID_WAITTIME)
        self.execCmd(SUCCESS, startparams, '--async', 'set', 'test-jail1', 'unbanip', '192.0.2.64/26')
        self.assertLogged('[test-jail1] Unban 192.0.2.96/28', '[test-jail1] Unban 192.0.2.112/28', all=True, wait=MID_WAITTIME)
        self.pruneLog('[test-phase 7]')
        self.execCmd(SUCCESS, startparams, 'reload', '--unban')
        self.assertLogged('Reload finished.', wait=MID_WAITTIME)
        self.assertLogged("Jail 'test-jail1' reloaded", '[test-jail1] Unban 192.0.2.1', '[test-jail1] Unban 192.0.2.4', all=True)
        self.assertNotLogged("Jail 'test-jail1' stopped", "Jail 'test-jail1' started", '[test-jail1] Ban 192.0.2.1', '[test-jail1] Ban 192.0.2.4', all=True)
        self.pruneLog('[test-phase 7b]')
        self.execCmd(SUCCESS, startparams, '--async', 'unban', '--all')
        self.assertLogged('Flush ban list', "Unbanned 0, 0 ticket(s) in 'test-jail1'", all=True)
        self.pruneLog('[test-phase 8a]')
        _write_jail_cfg(enabled=[1], backend='xxx-unknown-backend-zzz')
        self.execCmd(FAILED, startparams, 'reload')
        self.assertLogged('Reload finished.', wait=MID_WAITTIME)
        self.assertLogged("Restart jail 'test-jail1' (reason: 'polling' != ", 'Unknown backend ', all=True)
        self.pruneLog('[test-phase 8b]')
        _write_jail_cfg(enabled=[1])
        self.execCmd(SUCCESS, startparams, 'reload')
        self.assertLogged('Reload finished.', wait=MID_WAITTIME)
        self.pruneLog('[test-phase end-1]')
        self.execCmd(FAILED, startparams, '--async', 'reload', 'test-jail2')
        self.assertLogged('Reload finished.', wait=MID_WAITTIME)
        self.assertLogged("the jail 'test-jail2' does not exist")
        self.pruneLog()
        self.execCmd(SUCCESS, startparams, '--async', 'reload', '--if-exists', 'test-jail2')
        self.assertLogged('Reload finished.', wait=MID_WAITTIME)
        self.assertNotLogged("Creating new jail 'test-jail2'", "Jail 'test-jail2' started", all=True)
        self.pruneLog('[test-phase end-2]')
        self.execCmd(SUCCESS, startparams, '--async', 'reload', '--restart', '--all')
        self.assertLogged('Reload finished.', wait=MID_WAITTIME)
        self.assertLogged("Jail 'test-jail1' stopped", "Jail 'test-jail1' started", all=True, wait=MID_WAITTIME)
        self.pruneLog('[test-phase end-3]')
        self.execCmd(SUCCESS, startparams, '--async', 'set', 'test-jail1', 'addignoreip', '192.0.2.1/32', '2001:DB8::1/96')
        self.execCmd(SUCCESS, startparams, '--async', 'get', 'test-jail1', 'ignoreip')
        self.assertLogged('192.0.2.1/32', '2001:DB8::1/96', all=True)

    @unittest.F2B.skip_if_cfg_missing(action='nginx-block-map')
    @with_foreground_server_thread(startextra={'create_before_start': ('%(tmp)s/blck-failures.log',), 'use_stock_cfg': ('action.d',), 'jails': ('[nginx-blck-lst]', 'backend = polling', 'usedns = no', 'logpath = %(tmp)s/blck-failures.log', 'action = nginx-block-map[srv_cmd="echo nginx", srv_pid="%(tmp)s/f2b.pid", blck_lst_file="%(tmp)s/blck-lst.map"]', '         blocklist_de[actionban=\'curl() { echo "*** curl" "$*";}; <Definition/actionban>\', email="Fail2Ban <fail2ban@localhost>", apikey="TEST-API-KEY", agent="fail2ban-test-agent", service=<name>]', 'filter =', 'datepattern = ^Epoch', 'failregex = ^ failure "<F-ID>[^"]+</F-ID>" - <ADDR>', 'maxretry = 1', 'enabled = true')})
    def testServerActions_NginxBlockMap(self, tmp, startparams):
        if False:
            while True:
                i = 10
        cfg = pjoin(tmp, 'config')
        lgfn = '%(tmp)s/blck-failures.log' % {'tmp': tmp}
        mpfn = '%(tmp)s/blck-lst.map' % {'tmp': tmp}
        _write_file(lgfn, 'w+', str(int(MyTime.time())) + ' failure "125-000-001" - 192.0.2.1', str(int(MyTime.time())) + ' failure "125-000-002" - 192.0.2.1', str(int(MyTime.time())) + ' failure "125-000-003" - 192.0.2.1 (òðåòèé)', str(int(MyTime.time())) + ' failure "125-000-004" - 192.0.2.1 (òðåòèé)', str(int(MyTime.time())) + ' failure "125-000-005" - 192.0.2.1')
        self.assertLogged('[nginx-blck-lst] Ban 125-000-001', '[nginx-blck-lst] Ban 125-000-002', '[nginx-blck-lst] Ban 125-000-003', '[nginx-blck-lst] Ban 125-000-004', '[nginx-blck-lst] Ban 125-000-005', '5 ticket(s)', all=True, wait=MID_WAITTIME)
        _out_file(mpfn)
        mp = _read_file(mpfn)
        self.assertIn('\\125-000-001 1;\n', mp)
        self.assertIn('\\125-000-002 1;\n', mp)
        self.assertIn('\\125-000-003 1;\n', mp)
        self.assertIn('\\125-000-004 1;\n', mp)
        self.assertIn('\\125-000-005 1;\n', mp)
        self.assertLogged("stdout: 'nginx -qt'", "stdout: 'nginx -s reload'", all=True)
        self.assertLogged("stdout: '*** curl --fail --data-urlencode server=Fail2Ban <fail2ban@localhost> --data apikey=TEST-API-KEY --data service=nginx-blck-lst ", "stdout: ' --data format=text --user-agent fail2ban-test-agent", all=True, wait=MID_WAITTIME)
        self.execCmd(SUCCESS, startparams, 'unban', '125-000-001', '125-000-002', '125-000-005')
        _out_file(mpfn)
        mp = _read_file(mpfn)
        self.assertNotIn('\\125-000-001 1;\n', mp)
        self.assertNotIn('\\125-000-002 1;\n', mp)
        self.assertNotIn('\\125-000-005 1;\n', mp)
        self.assertIn('\\125-000-003 1;\n', mp)
        self.assertIn('\\125-000-004 1;\n', mp)
        self.stopAndWaitForServerEnd(SUCCESS)
        self.assertLogged('[nginx-blck-lst] Flush ticket(s) with nginx-block-map')
        _out_file(mpfn)
        mp = _read_file(mpfn)
        self.assertEqual(mp, '')

    @unittest.F2B.skip_if_cfg_missing(filter='sendmail-auth')
    @with_foreground_server_thread(startextra={'create_before_start': ('%(tmp)s/test.log',), 'use_stock': True, 'f2b_local': ('[DEFAULT]', 'dbmaxmatches = 1'), 'jails': ('test_action = dummy[actionstart_on_demand=1, init="start: %(__name__)s", target="%(tmp)s/test.txt",\n      actionban=\'<known/actionban>; echo "found: <jail.found> / <jail.found_total>, banned: <jail.banned> / <jail.banned_total>"\n        echo "<matches>"; printf "=====\\n%%b\\n=====\\n\\n" "<matches>" >> <target>\',\n      actionstop=\'<known/actionstop>; echo "stats <name> - found: <jail.found_total>, banned: <jail.banned_total>"\']', '[sendmail-auth]', 'backend = polling', 'usedns = no', 'logpath = %(tmp)s/test.log', 'action = %(test_action)s', 'filter = sendmail-auth[logtype=short]', 'datepattern = ^Epoch', 'maxretry = 3', 'maxmatches = 2', 'enabled = true', '[sendmail-reject]', 'backend = polling', 'usedns = no', 'logpath = %(tmp)s/test.log', 'action = %(test_action)s', 'filter = sendmail-reject[logtype=short]', 'datepattern = ^Epoch', 'maxretry = 3', 'enabled = true')})
    def testServerJails_Sendmail(self, tmp, startparams):
        if False:
            return 10
        cfg = pjoin(tmp, 'config')
        lgfn = '%(tmp)s/test.log' % {'tmp': tmp}
        tofn = '%(tmp)s/test.txt' % {'tmp': tmp}
        smaut_msg = (str(int(MyTime.time())) + ' smtp1 sm-mta[5133]: s1000000000001: [192.0.2.1]: possible SMTP attack: command=AUTH, count=1', str(int(MyTime.time())) + ' smtp1 sm-mta[5133]: s1000000000002: [192.0.2.1]: possible SMTP attack: command=AUTH, count=2', str(int(MyTime.time())) + ' smtp1 sm-mta[5133]: s1000000000003: [192.0.2.1]: possible SMTP attack: command=AUTH, count=3')
        smrej_msg = (str(int(MyTime.time())) + ' smtp1 sm-mta[21134]: s2000000000001: ruleset=check_rcpt, arg1=<123@example.com>, relay=xxx.dynamic.example.com [192.0.2.2], reject=550 5.7.1 <123@example.com>... Relaying denied. Proper authentication required.', str(int(MyTime.time())) + ' smtp1 sm-mta[21134]: s2000000000002: ruleset=check_rcpt, arg1=<345@example.com>, relay=xxx.dynamic.example.com [192.0.2.2], reject=550 5.7.1 <345@example.com>... Relaying denied. Proper authentication required.', str(int(MyTime.time())) + ' smtp1 sm-mta[21134]: s3000000000003: ruleset=check_rcpt, arg1=<567@example.com>, relay=xxx.dynamic.example.com [192.0.2.2], reject=550 5.7.1 <567@example.com>... Relaying denied. Proper authentication required.')
        self.pruneLog('[test-phase sendmail-auth]')
        _write_file(lgfn, 'w+', *smaut_msg)
        self.assertLogged('[sendmail-auth] Ban 192.0.2.1', "stdout: 'found: 0 / 3, banned: 1 / 1'", "1 ticket(s) in 'sendmail-auth'", all=True, wait=MID_WAITTIME)
        _out_file(tofn)
        td = _read_file(tofn)
        m = smaut_msg[0]
        self.assertNotIn(m, td)
        for m in smaut_msg[1:]:
            self.assertIn(m, td)
        self.pruneLog('[test-phase sendmail-reject]')
        _write_file(lgfn, 'a+', *smrej_msg)
        self.assertLogged('[sendmail-reject] Ban 192.0.2.2', "stdout: 'found: 0 / 3, banned: 1 / 1'", "1 ticket(s) in 'sendmail-reject'", all=True, wait=MID_WAITTIME)
        _out_file(tofn)
        td = _read_file(tofn)
        for m in smrej_msg:
            self.assertIn(m, td)
        self.pruneLog('[test-phase restart sendmail-*]')
        self.execCmd(SUCCESS, startparams, 'reload', '--restart', '--all')
        self.assertLogged('Reload finished.', "stdout: 'stats sendmail-auth - found: 3, banned: 1'", "stdout: 'stats sendmail-reject - found: 3, banned: 1'", '[sendmail-auth] Restore Ban 192.0.2.1', "1 ticket(s) in 'sendmail-auth'", all=True, wait=MID_WAITTIME)
        td = _read_file(tofn)
        m = smaut_msg[-1]
        self.assertLogged(m)
        self.assertIn(m, td)
        for m in smaut_msg[0:-1]:
            self.assertNotLogged(m)
            self.assertNotIn(m, td)
        self.assertLogged('[sendmail-reject] Restore Ban 192.0.2.2', "1 ticket(s) in 'sendmail-reject'", all=True, wait=MID_WAITTIME)
        td = _read_file(tofn)
        m = smrej_msg[-1]
        self.assertLogged(m)
        self.assertIn(m, td)
        for m in smrej_msg[0:-1]:
            self.assertNotLogged(m)
            self.assertNotIn(m, td)
        self.pruneLog('[test-phase stop server]')
        self.stopAndWaitForServerEnd(SUCCESS)
        self.assertFalse(exists(tofn))

    @with_foreground_server_thread()
    def testServerObserver(self, tmp, startparams):
        if False:
            return 10
        cfg = pjoin(tmp, 'config')
        test1log = pjoin(tmp, 'test1.log')
        os.mkdir(pjoin(cfg, 'action.d'))

        def _write_action_cfg(actname='test-action1', prolong=True):
            if False:
                return 10
            fn = pjoin(cfg, 'action.d', '%s.conf' % actname)
            _write_file(fn, 'w', '[DEFAULT]', '', '[Definition]', 'actionban =     printf %%s "[%(name)s] %(actname)s: ++ ban <ip> -c <bancount> -t <bantime> : <F-MSG>"', 'actionprolong = printf %%s "[%(name)s] %(actname)s: ++ prolong <ip> -c <bancount> -t <bantime> : <F-MSG>"' if prolong else '', "actionunban =   printf %%b '[%(name)s] %(actname)s: -- unban <ip>'")
            if unittest.F2B.log_level <= logging.DEBUG:
                _out_file(fn)

        def _write_jail_cfg(backend='polling'):
            if False:
                print('Hello World!')
            _write_file(pjoin(cfg, 'jail.conf'), 'w', '[INCLUDES]', '', '[DEFAULT]', '', 'usedns = no', 'maxretry = 3', 'findtime = 1m', 'bantime = 5m', 'bantime.increment = true', 'datepattern = {^LN-BEG}EPOCH', '', '[test-jail1]', 'backend = ' + backend, 'filter =', "action = test-action1[name='%(__name__)s']", "         test-action2[name='%(__name__)s']", 'logpath = ' + test1log, 'failregex = ^\\s*failure <F-ERRCODE>401|403</F-ERRCODE> from <HOST>:\\s*<F-MSG>.*</F-MSG>$', 'enabled = true', '')
            if unittest.F2B.log_level <= logging.DEBUG:
                _out_file(pjoin(cfg, 'jail.conf'))
        _write_action_cfg(actname='test-action1', prolong=False)
        _write_action_cfg(actname='test-action2', prolong=True)
        _write_jail_cfg()
        _write_file(test1log, 'w')
        self.pruneLog('[test-phase 0) time-0]')
        self.execCmd(SUCCESS, startparams, 'reload')
        _write_file(test1log, 'w+', *(str(int(MyTime.time())) + ' failure 401 from 192.0.2.11: I\'m bad "hacker" `` $(echo test)',) * 3)
        _observer_wait_idle()
        self.assertLogged("stdout: '[test-jail1] test-action1: ++ ban 192.0.2.11 -c 1 -t 300 : ", "stdout: '[test-jail1] test-action2: ++ ban 192.0.2.11 -c 1 -t 300 : ", all=True, wait=MID_WAITTIME)
        _observer_wait_idle()
        self.pruneLog('[test-phase 1) time+10m]')
        _time_shift(10)
        _observer_wait_idle()
        self.assertLogged("stdout: '[test-jail1] test-action1: -- unban 192.0.2.11", "stdout: '[test-jail1] test-action2: -- unban 192.0.2.11", "0 ticket(s) in 'test-jail1'", all=True, wait=MID_WAITTIME)
        _observer_wait_idle()
        self.pruneLog('[test-phase 2) time+10m]')
        wakeObs = False
        _observer_wait_before_incrban(lambda : wakeObs)
        _write_file(test1log, 'a+', *(str(int(MyTime.time())) + ' failure 401 from 192.0.2.11: I\'m very bad "hacker" `` $(echo test)',) * 2)
        self.assertLogged("stdout: '[test-jail1] test-action1: ++ ban 192.0.2.11 -c 2 -t 300 : ", "stdout: '[test-jail1] test-action2: ++ ban 192.0.2.11 -c 2 -t 300 : ", all=True, wait=MID_WAITTIME)
        self.pruneLog('[test-phase 2) time+10m - get-ips]')
        self.execCmd(SUCCESS, startparams, 'get', 'test-jail1', 'banip', '--with-time')
        self.assertLogged('192.0.2.11', '+ 300 =', all=True, wait=MID_WAITTIME)
        wakeObs = True
        _observer_wait_idle()
        self.pruneLog('[test-phase 2) time+11m]')
        _time_shift(1)
        _observer_wait_idle()
        self.assertLogged("stdout: '[test-jail1] test-action2: ++ prolong 192.0.2.11 -c 2 -t 600 : ", all=True, wait=MID_WAITTIME)
        _observer_wait_idle()
        self.pruneLog('[test-phase 2) time+11m - get-ips]')
        self.execCmd(SUCCESS, startparams, 'get', 'test-jail1', 'banip', '--with-time')
        self.assertLogged('192.0.2.11', '+ 600 =', all=True, wait=MID_WAITTIME)
        self.pruneLog('[test-phase end) stop on busy observer]')
        tearDownMyTime()
        a = {'state': 0}
        obsMain = Observers.Main

        def _long_action():
            if False:
                print('Hello World!')
            logSys.info('++ observer enters busy state ...')
            a['state'] = 1
            Utils.wait_for(lambda : a['state'] == 2, MAX_WAITTIME)
            obsMain.db_purge()
            logSys.info('-- observer leaves busy state.')
        obsMain.add('call', _long_action)
        obsMain.add('call', lambda : None)
        Utils.wait_for(lambda : a['state'] == 1, MAX_WAITTIME)
        obsMain_stop = obsMain.stop

        def _stop(wtime=0.01 if unittest.F2B.fast else 0.1, forceQuit=True):
            if False:
                i = 10
                return i + 15
            return obsMain_stop(wtime, forceQuit)
        obsMain.stop = _stop
        self.stopAndWaitForServerEnd(SUCCESS)
        self.assertNotLogged('observer leaves busy state')
        self.assertFalse(obsMain.idle)
        self.assertEqual(obsMain._ObserverThread__db, None)
        a['state'] = 2
        self.assertLogged('observer leaves busy state', wait=True)
        obsMain.join()
    if False:

        @with_foreground_server_thread()
        def _testServerStartStop(self, tmp, startparams):
            if False:
                return 10
            self.stopAndWaitForServerEnd(SUCCESS)

        def testServerStartStop(self):
            if False:
                while True:
                    i = 10
            for i in range(2000):
                self._testServerStartStop()