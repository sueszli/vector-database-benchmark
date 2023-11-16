__copyright__ = 'Copyright (c) 2004 Cyril Jaquier; 2012 Yaroslav Halchenko'
__license__ = 'GPL'
from builtins import open as fopen
import unittest
import os
import re
import sys
import time, datetime
import tempfile
import uuid
try:
    from systemd import journal
except ImportError:
    journal = None
from ..helpers import uni_bytes
from ..server.jail import Jail
from ..server.filterpoll import FilterPoll
from ..server.filter import FailTicket, Filter, FileFilter, FileContainer
from ..server.failmanager import FailManagerEmpty
from ..server.ipdns import asip, getfqdn, DNSUtils, IPAddr, IPAddrSet
from ..server.mytime import MyTime
from ..server.utils import Utils, uni_decode
from .databasetestcase import getFail2BanDb
from .utils import setUpMyTime, tearDownMyTime, mtimesleep, with_alt_time, with_tmpdir, LogCaptureTestCase, logSys as DefLogSys, CONFIG_DIR as STOCK_CONF_DIR
from .dummyjail import DummyJail
TEST_FILES_DIR = os.path.join(os.path.dirname(__file__), 'files')

def open(*args):
    if False:
        print('Hello World!')
    'Overload built in open so we could assure sufficiently large buffer\n\n\tExplicit .flush would be needed to assure that changes leave the buffer\n\t'
    if len(args) == 2:
        args = args + (50000,)
    return fopen(*args)

def _killfile(f, name):
    if False:
        print('Hello World!')
    try:
        f.close()
    except:
        pass
    try:
        os.unlink(name)
    except:
        pass
    if os.path.exists(name + '.bak'):
        _killfile(None, name + '.bak')
_maxWaitTime = unittest.F2B.maxWaitTime

class _tmSerial:
    _last_s = -2147483647
    _last_m = -2147483647
    _str_s = ''
    _str_m = ''

    @staticmethod
    def _tm(time):
        if False:
            for i in range(10):
                print('nop')
        c = _tmSerial
        sec = time % 60
        if c._last_s == time - sec:
            return '%s%02u' % (c._str_s, sec)
        mt = time % 3600
        if c._last_m == time - mt:
            c._last_s = time - sec
            c._str_s = '%s%02u:' % (c._str_m, mt // 60)
            return '%s%02u' % (c._str_s, sec)
        c._last_m = time - mt
        c._str_m = datetime.datetime.fromtimestamp(time).strftime('%Y-%m-%d %H:')
        c._last_s = time - sec
        c._str_s = '%s%02u:' % (c._str_m, mt // 60)
        return '%s%02u' % (c._str_s, sec)
_tm = _tmSerial._tm
_tmb = lambda t: uni_bytes(_tm(t))

def _assert_equal_entries(utest, found, output, count=None):
    if False:
        while True:
            i = 10
    'Little helper to unify comparisons with the target entries\n\n\tand report helpful failure reports instead of millions of seconds ;)\n\t'
    utest.assertEqual(found[0], output[0])
    utest.assertEqual(found[1], count or output[1])
    (found_time, output_time) = (MyTime.localtime(found[2]), MyTime.localtime(output[2]))
    try:
        utest.assertEqual(found_time, output_time)
    except AssertionError as e:
        utest.assertEqual((float(found[2]), found_time), (float(output[2]), output_time))
    if len(output) > 3 and count is None:
        if os.linesep != '\n' or sys.platform.startswith('cygwin'):
            srepr = lambda x: repr(x).replace('\\r', '')
        else:
            srepr = repr
        utest.assertEqual(srepr(found[3]), srepr(output[3]))

def _ticket_tuple(ticket):
    if False:
        return 10
    'Create a tuple for easy comparison from fail ticket\n\t'
    attempts = ticket.getAttempt()
    date = ticket.getTime()
    ip = ticket.getID()
    matches = ticket.getMatches()
    return (ip, attempts, date, matches)

def _assert_correct_last_attempt(utest, filter_, output, count=None):
    if False:
        return 10
    'Additional helper to wrap most common test case\n\n\tTest filter to contain target ticket\n\t'
    if not isinstance(output[0], (tuple, list)):
        tickcount = 1
        failcount = count if count else output[1]
    else:
        tickcount = len(output)
        failcount = count if count else sum((o[1] for o in output))
    found = []
    if isinstance(filter_, DummyJail):
        found.append(_ticket_tuple(filter_.getFailTicket()))
    else:
        if filter_.jail:
            while True:
                t = filter_.jail.getFailTicket()
                if not t:
                    break
                found.append(_ticket_tuple(t))
        if found:
            tickcount -= len(found)
        if tickcount > 0:
            Utils.wait_for(lambda : filter_.failManager.getFailCount() >= (tickcount, failcount), _maxWaitTime(10))
            while tickcount:
                try:
                    found.append(_ticket_tuple(filter_.failManager.toBan()))
                except FailManagerEmpty:
                    break
                tickcount -= 1
    if not isinstance(output[0], (tuple, list)):
        utest.assertEqual(len(found), 1)
        _assert_equal_entries(utest, found[0], output, count)
    else:
        utest.assertEqual(len(found), len(output))
        found = sorted(found, key=lambda x: str(x))
        output = sorted(output, key=lambda x: str(x))
        for (f, o) in zip(found, output):
            _assert_equal_entries(utest, f, o)

def _copy_lines_between_files(in_, fout, n=None, skip=0, mode='a', terminal_line='', lines=None):
    if False:
        while True:
            i = 10
    'Copy lines from one file to another (which might be already open)\n\n\tReturns open fout\n\t'
    mtimesleep()
    if terminal_line is not None:
        terminal_line = uni_bytes(terminal_line)
    if isinstance(in_, str):
        fin = open(in_, 'rb')
    else:
        fin = in_
    for i in range(skip):
        fin.readline()
    i = 0
    if lines:
        lines = list(map(uni_bytes, lines))
    else:
        lines = []
    while n is None or i < n:
        l = fin.readline().rstrip(b'\r\n')
        if terminal_line is not None and l == terminal_line:
            break
        lines.append(l)
        i += 1
    if isinstance(fout, str):
        fout = open(fout, mode + 'b')
    DefLogSys.debug('  ++ write %d test lines', len(lines))
    fout.write(b'\n'.join(lines) + b'\n')
    fout.flush()
    if isinstance(in_, str):
        fin.close()
    time.sleep(Utils.DEFAULT_SHORT_INTERVAL)
    return fout
TEST_JOURNAL_FIELDS = {'SYSLOG_IDENTIFIER': 'fail2ban-testcases', 'PRIORITY': '7'}

def _copy_lines_to_journal(in_, fields={}, n=None, skip=0, terminal_line=''):
    if False:
        while True:
            i = 10
    'Copy lines from one file to systemd journal\n\n\tReturns None\n\t'
    if isinstance(in_, str):
        fin = open(in_, 'rb')
    else:
        fin = in_
    fields.update(TEST_JOURNAL_FIELDS)
    for i in range(skip):
        fin.readline()
    i = 0
    while n is None or i < n:
        l = fin.readline().decode('UTF-8', 'replace').rstrip('\r\n')
        if terminal_line is not None and l == terminal_line:
            break
        journal.send(MESSAGE=l.strip(), **fields)
        i += 1
    if isinstance(in_, str):
        fin.close()

class BasicFilter(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        super(BasicFilter, self).setUp()
        self.filter = Filter(None)

    def testGetSetUseDNS(self):
        if False:
            return 10
        self.assertEqual(self.filter.getUseDns(), 'warn')
        self.filter.setUseDns(True)
        self.assertEqual(self.filter.getUseDns(), 'yes')
        self.filter.setUseDns(False)
        self.assertEqual(self.filter.getUseDns(), 'no')

    def testGetSetDatePattern(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.filter.getDatePattern(), (None, 'Default Detectors'))
        self.filter.setDatePattern('^%Y-%m-%d-%H%M%S\\.%f %z **')
        self.assertEqual(self.filter.getDatePattern(), ('^%Y-%m-%d-%H%M%S\\.%f %z **', '^Year-Month-Day-24hourMinuteSecond\\.Microseconds Zone offset **'))

    def testGetSetLogTimeZone(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.filter.getLogTimeZone(), None)
        self.filter.setLogTimeZone('UTC')
        self.assertEqual(self.filter.getLogTimeZone(), 'UTC')
        self.filter.setLogTimeZone('UTC-0400')
        self.assertEqual(self.filter.getLogTimeZone(), 'UTC-0400')
        self.filter.setLogTimeZone('UTC+0200')
        self.assertEqual(self.filter.getLogTimeZone(), 'UTC+0200')
        self.assertRaises(ValueError, self.filter.setLogTimeZone, 'not-a-time-zone')

    def testAssertWrongTime(self):
        if False:
            return 10
        self.assertRaises(AssertionError, lambda : _assert_equal_entries(self, ('1.1.1.1', 1, 1421262060.0), ('1.1.1.1', 1, 1421262059.0), 1))

    def testTest_tm(self):
        if False:
            while True:
                i = 10
        unittest.F2B.SkipIfFast()
        for i in range(1417512352, (1417512352 // 3600 + 3) * 3600):
            tm = MyTime.time2str(i)
            if _tm(i) != tm:
                self.assertEqual((_tm(i), i), (tm, i))

    def testWrongCharInTupleLine(self):
        if False:
            print('Hello World!')
        for a1 in ('', '', b''):
            for a2 in ('2016-09-05T20:18:56', '2016-09-05T20:18:56', b'2016-09-05T20:18:56'):
                for a3 in ('Fail for "gÃ¶ran" from 192.0.2.1', 'Fail for "gÃ¶ran" from 192.0.2.1', b'Fail for "g\xc3\xb6ran" from 192.0.2.1'):
                    ''.join([uni_decode(v) for v in (a1, a2, a3)])

class IgnoreIP(LogCaptureTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        'Call before every test case.'
        LogCaptureTestCase.setUp(self)
        self.jail = DummyJail()
        self.filter = FileFilter(self.jail)
        self.filter.ignoreSelf = False

    def testIgnoreSelfIP(self):
        if False:
            for i in range(10):
                print('nop')
        ipList = ('127.0.0.1',)
        for ip in ipList:
            self.assertFalse(self.filter.inIgnoreIPList(ip))
            self.assertNotLogged('[%s] Ignore %s by %s' % (self.jail.name, ip, 'ignoreself rule'))
        self.filter.ignoreSelf = True
        self.pruneLog()
        for ip in ipList:
            self.assertTrue(self.filter.inIgnoreIPList(ip))
            self.assertLogged('[%s] Ignore %s by %s' % (self.jail.name, ip, 'ignoreself rule'))

    def testIgnoreIPOK(self):
        if False:
            for i in range(10):
                print('nop')
        ipList = ('127.0.0.1', '192.168.0.1', '255.255.255.255', '99.99.99.99')
        for ip in ipList:
            self.filter.addIgnoreIP(ip)
            self.assertTrue(self.filter.inIgnoreIPList(ip))
            self.assertLogged('[%s] Ignore %s by %s' % (self.jail.name, ip, 'ip'))

    def testIgnoreIPNOK(self):
        if False:
            return 10
        ipList = ('', '999.999.999.999', 'abcdef.abcdef', '192.168.0.')
        for ip in ipList:
            self.filter.addIgnoreIP(ip)
            self.assertFalse(self.filter.inIgnoreIPList(ip))
        if not unittest.F2B.no_network:
            self.assertLogged('Unable to find a corresponding IP address for 999.999.999.999', 'Unable to find a corresponding IP address for abcdef.abcdef', 'Unable to find a corresponding IP address for 192.168.0.', all=True)

    def testIgnoreIPCIDR(self):
        if False:
            for i in range(10):
                print('nop')
        self.filter.addIgnoreIP('192.168.1.0/25')
        self.assertTrue(self.filter.inIgnoreIPList('192.168.1.0'))
        self.assertTrue(self.filter.inIgnoreIPList('192.168.1.1'))
        self.assertTrue(self.filter.inIgnoreIPList('192.168.1.127'))
        self.assertFalse(self.filter.inIgnoreIPList('192.168.1.128'))
        self.assertFalse(self.filter.inIgnoreIPList('192.168.1.255'))
        self.assertFalse(self.filter.inIgnoreIPList('192.168.0.255'))

    def testIgnoreIPMask(self):
        if False:
            for i in range(10):
                print('nop')
        self.filter.addIgnoreIP('192.168.1.0/255.255.255.128')
        self.assertTrue(self.filter.inIgnoreIPList('192.168.1.0'))
        self.assertTrue(self.filter.inIgnoreIPList('192.168.1.1'))
        self.assertTrue(self.filter.inIgnoreIPList('192.168.1.127'))
        self.assertFalse(self.filter.inIgnoreIPList('192.168.1.128'))
        self.assertFalse(self.filter.inIgnoreIPList('192.168.1.255'))
        self.assertFalse(self.filter.inIgnoreIPList('192.168.0.255'))

    def testWrongIPMask(self):
        if False:
            i = 10
            return i + 15
        self.filter.addIgnoreIP('192.168.1.0/255.255.0.0')
        self.assertRaises(ValueError, self.filter.addIgnoreIP, '192.168.1.0/255.255.0.128')

    def testIgnoreInProcessLine(self):
        if False:
            while True:
                i = 10
        setUpMyTime()
        try:
            self.filter.addIgnoreIP('192.168.1.0/25')
            self.filter.addFailRegex('<HOST>')
            self.filter.setDatePattern('{^LN-BEG}EPOCH')
            self.filter.processLineAndAdd('1387203300.222 192.168.1.32')
            self.assertLogged('Ignore 192.168.1.32')
        finally:
            tearDownMyTime()

    def _testTimeJump(self, inOperation=False):
        if False:
            print('Hello World!')
        try:
            self.filter.addFailRegex('^<HOST>')
            self.filter.setDatePattern('{^LN-BEG}%Y-%m-%d %H:%M:%S(?:\\s*%Z)?\\s')
            self.filter.setFindTime(10)
            self.filter.setMaxRetry(5)
            self.filter.inOperation = inOperation
            self.pruneLog('[phase 1] DST time jump')
            MyTime.setTime(1572137999)
            self.filter.processLineAndAdd('2019-10-27 02:59:59 192.0.2.5')
            MyTime.setTime(1572138000)
            self.filter.processLineAndAdd('2019-10-27 02:00:00 192.0.2.5')
            MyTime.setTime(1572138001)
            self.filter.processLineAndAdd('2019-10-27 02:00:01 192.0.2.5')
            self.assertLogged('Current failures from 1 IPs (IP:count): 192.0.2.5:1', 'Current failures from 1 IPs (IP:count): 192.0.2.5:2', 'Current failures from 1 IPs (IP:count): 192.0.2.5:3', 'Total # of detected failures: 3.', all=True, wait=True)
            self.assertNotLogged('Ignore line')
            self.pruneLog('[phase 2] UTC time jump (NTP correction)')
            MyTime.setTime(1572210000)
            self.filter.processLineAndAdd('2019-10-27 22:00:00 CET 192.0.2.6')
            MyTime.setTime(1572200000)
            self.filter.processLineAndAdd('2019-10-27 22:00:01 CET 192.0.2.6')
            self.filter.processLineAndAdd('2019-10-27 19:13:20 CET 192.0.2.6')
            self.filter.processLineAndAdd('2019-10-27 19:13:21 CET 192.0.2.6')
            self.assertLogged('192.0.2.6:1', '192.0.2.6:2', '192.0.2.6:3', '192.0.2.6:4', 'Total # of detected failures: 7.', all=True, wait=True)
            self.assertNotLogged('Ignore line')
        finally:
            tearDownMyTime()

    def testTimeJump(self):
        if False:
            for i in range(10):
                print('nop')
        self._testTimeJump(inOperation=False)

    def testTimeJump_InOperation(self):
        if False:
            while True:
                i = 10
        self._testTimeJump(inOperation=True)

    def testWrongTimeOrTZ(self):
        if False:
            while True:
                i = 10
        try:
            self.filter.addFailRegex('fail from <ADDR>$')
            self.filter.setDatePattern('{^LN-BEG}%Y-%m-%d %H:%M:%S(?:\\s*%Z)?\\s')
            self.filter.setMaxRetry(50)
            self.filter.inOperation = True
            MyTime.setTime(1572138000 + 3600)
            self.pruneLog('[phase 1] simulate wrong TZ')
            for i in (1, 2, 3):
                self.filter.processLineAndAdd('2019-10-27 02:00:00 fail from 192.0.2.15')
            self.assertLogged('Detected a log entry 1h before the current time in operation mode. This looks like a timezone problem.', 'Please check a jail for a timing issue.', '192.0.2.15:1', '192.0.2.15:2', '192.0.2.15:3', 'Total # of detected failures: 3.', all=True, wait=True)
            setattr(self.filter, '_next_simByTimeWarn', -1)
            self.pruneLog('[phase 2] wrong TZ given in log')
            for i in (1, 2, 3):
                self.filter.processLineAndAdd('2019-10-27 04:00:00 GMT fail from 192.0.2.16')
            self.assertLogged('Detected a log entry 2h after the current time in operation mode. This looks like a timezone problem.', 'Please check a jail for a timing issue.', '192.0.2.16:1', '192.0.2.16:2', '192.0.2.16:3', 'Total # of detected failures: 6.', all=True, wait=True)
            self.assertNotLogged('Found a match but no valid date/time found')
            self.pruneLog("[phase 3] other timestamp (don't match datepattern), regex matches")
            for i in range(3):
                self.filter.processLineAndAdd('27.10.2019 04:00:00 fail from 192.0.2.17')
            self.assertLogged('Found a match but no valid date/time found', 'Match without a timestamp:', '192.0.2.17:1', '192.0.2.17:2', '192.0.2.17:3', 'Total # of detected failures: 9.', all=True, wait=True)
            phase = 3
            for (delta, expect) in ((-90 * 60, 'timezone'), (-60 * 60, 'timezone'), (-10 * 60, 'timezone'), (-59, None), (59, None), (61, 'latency'), (55 * 60, 'latency'), (90 * 60, 'timezone')):
                phase += 1
                MyTime.setTime(1572138000 + delta)
                setattr(self.filter, '_next_simByTimeWarn', -1)
                self.pruneLog('[phase {phase}] log entries offset by {delta}s'.format(phase=phase, delta=delta))
                self.filter.processLineAndAdd('2019-10-27 02:00:00 fail from 192.0.2.15')
                self.assertLogged('Found 192.0.2.15', wait=True)
                if expect:
                    self.assertLogged(('timezone problem', 'latency problem')[int(expect == 'latency')], all=True)
                    self.assertNotLogged(('timezone problem', 'latency problem')[int(expect != 'latency')], all=True)
                else:
                    self.assertNotLogged('timezone problem', 'latency problem', all=True)
        finally:
            tearDownMyTime()

    def testAddAttempt(self):
        if False:
            return 10
        self.filter.setMaxRetry(3)
        for i in range(1, 1 + 3):
            self.filter.addAttempt('192.0.2.1')
            self.assertLogged('Attempt 192.0.2.1', '192.0.2.1:%d' % i, all=True, wait=True)
        self.jail.actions._Actions__checkBan()
        self.assertLogged('Ban 192.0.2.1', wait=True)

    def testIgnoreCommand(self):
        if False:
            while True:
                i = 10
        self.filter.ignoreCommand = sys.executable + ' ' + os.path.join(TEST_FILES_DIR, 'ignorecommand.py <ip>')
        self.assertTrue(self.filter.inIgnoreIPList('10.0.0.1'))
        self.assertFalse(self.filter.inIgnoreIPList('10.0.0.0'))
        self.assertLogged('returned successfully 0', 'returned successfully 1', all=True)
        self.pruneLog()
        self.assertFalse(self.filter.inIgnoreIPList(''))
        self.assertLogged('usage: ignorecommand IP', 'returned 10', all=True)

    def testIgnoreCommandForTicket(self):
        if False:
            for i in range(10):
                print('nop')
        self.filter.ignoreCommand = 'if [ "<ip-host>" = "test-host" ]; then exit 0; fi; exit 1'
        self.pruneLog()
        self.assertTrue(self.filter.inIgnoreIPList(FailTicket('2001:db8::1')))
        self.assertLogged('returned successfully 0')
        self.pruneLog()
        self.assertFalse(self.filter.inIgnoreIPList(FailTicket('2001:db8::ffff')))
        self.assertLogged('returned successfully 1')
        self.filter.ignoreCommand = 'if [ "<F-USER>" = "tester" ]; then exit 0; fi; exit 1'
        self.pruneLog()
        self.assertTrue(self.filter.inIgnoreIPList(FailTicket('tester', data={'user': 'tester'})))
        self.assertLogged('returned successfully 0')
        self.pruneLog()
        self.assertFalse(self.filter.inIgnoreIPList(FailTicket('root', data={'user': 'root'})))
        self.assertLogged('returned successfully 1', all=True)

    def testIgnoreCache(self):
        if False:
            for i in range(10):
                print('nop')
        self.filter.ignoreCache = {'key': '<ip>'}
        self.filter.ignoreCommand = 'if [ "<ip>" = "10.0.0.1" ]; then exit 0; fi; exit 1'
        for i in range(5):
            self.pruneLog()
            self.assertTrue(self.filter.inIgnoreIPList('10.0.0.1'))
            self.assertFalse(self.filter.inIgnoreIPList('10.0.0.0'))
            if not i:
                self.assertLogged('returned successfully 0', 'returned successfully 1', all=True)
            else:
                self.assertNotLogged('returned successfully 0', 'returned successfully 1', all=True)
        self.filter.ignoreCache = {'key': '<ip-host>'}
        self.filter.ignoreCommand = 'if [ "<ip-host>" = "test-host" ]; then exit 0; fi; exit 1'
        for i in range(5):
            self.pruneLog()
            self.assertTrue(self.filter.inIgnoreIPList(FailTicket('2001:db8::1')))
            self.assertFalse(self.filter.inIgnoreIPList(FailTicket('2001:db8::ffff')))
            if not i:
                self.assertLogged('returned successfully')
            else:
                self.assertNotLogged('returned successfully')
        self.filter.ignoreCache = {'key': '<F-USER>', 'max-count': '10', 'max-time': '1h'}
        self.assertEqual(self.filter.ignoreCache, ['<F-USER>', 10, 60 * 60])
        self.filter.ignoreCommand = 'if [ "<F-USER>" = "tester" ]; then exit 0; fi; exit 1'
        for i in range(5):
            self.pruneLog()
            self.assertTrue(self.filter.inIgnoreIPList(FailTicket('tester', data={'user': 'tester'})))
            self.assertFalse(self.filter.inIgnoreIPList(FailTicket('root', data={'user': 'root'})))
            if not i:
                self.assertLogged('returned successfully')
            else:
                self.assertNotLogged('returned successfully')

    def testIgnoreCauseOK(self):
        if False:
            for i in range(10):
                print('nop')
        ip = '93.184.216.34'
        for ignore_source in ['dns', 'ip', 'command']:
            self.filter.logIgnoreIp(ip, True, ignore_source=ignore_source)
            self.assertLogged('[%s] Ignore %s by %s' % (self.jail.name, ip, ignore_source))

    def testIgnoreCauseNOK(self):
        if False:
            for i in range(10):
                print('nop')
        self.filter.logIgnoreIp('example.com', False, ignore_source='NOT_LOGGED')
        self.assertNotLogged('[%s] Ignore %s by %s' % (self.jail.name, 'example.com', 'NOT_LOGGED'))

class IgnoreIPDNS(LogCaptureTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        'Call before every test case.'
        unittest.F2B.SkipIfNoNetwork()
        LogCaptureTestCase.setUp(self)
        self.jail = DummyJail()
        self.filter = FileFilter(self.jail)

    def testIgnoreIPDNS(self):
        if False:
            for i in range(10):
                print('nop')
        for dns in ('www.epfl.ch', 'example.com'):
            self.filter.addIgnoreIP(dns)
            ips = DNSUtils.dnsToIp(dns)
            self.assertTrue(len(ips) > 0)
            for ip in ips:
                ip = str(ip)
                DefLogSys.debug('  ++ positive case for %s', ip)
                self.assertTrue(self.filter.inIgnoreIPList(ip))
                iparr = []
                ip2 = re.search('^([^.:]+)([.:])(.*?)([.:])([^.:]+)$', ip)
                if ip2:
                    ip2 = ip2.groups()
                    for o in (0, 4):
                        for i in (1, -1):
                            ipo = list(ip2)
                            if ipo[1] == '.':
                                ipo[o] = str(int(ipo[o]) + i)
                            else:
                                ipo[o] = '%x' % (int(ipo[o], 16) + i)
                            ipo = ''.join(ipo)
                            if ipo not in ips:
                                iparr.append(ipo)
                self.assertTrue(len(iparr) > 0)
                for ip in iparr:
                    DefLogSys.debug('  -- negative case for %s', ip)
                    self.assertFalse(self.filter.inIgnoreIPList(str(ip)))

    def testIgnoreCmdApacheFakegooglebot(self):
        if False:
            return 10
        unittest.F2B.SkipIfCfgMissing(stock=True)
        cmd = os.path.join(STOCK_CONF_DIR, 'filter.d/ignorecommands/apache-fakegooglebot')
        mod = Utils.load_python_module(cmd)
        self.assertFalse(mod.is_googlebot(*mod.process_args([cmd, '128.178.222.69'])))
        self.assertFalse(mod.is_googlebot(*mod.process_args([cmd, '192.0.2.1'])))
        self.assertFalse(mod.is_googlebot(*mod.process_args([cmd, '192.0.2.1', 0.1])))
        bot_ips = ['66.249.66.1']
        for ip in bot_ips:
            self.assertTrue(mod.is_googlebot(*mod.process_args([cmd, str(ip)])), 'test of googlebot ip %s failed' % ip)
        self.assertRaises(ValueError, lambda : mod.is_googlebot(*mod.process_args([cmd])))
        self.assertRaises(ValueError, lambda : mod.is_googlebot(*mod.process_args([cmd, '192.0'])))
        self.filter.ignoreCommand = cmd + ' <ip>'
        for ip in bot_ips:
            self.assertTrue(self.filter.inIgnoreIPList(str(ip)), 'test of googlebot ip %s failed' % ip)
            self.assertLogged('-- returned successfully')
            self.pruneLog()
        self.assertFalse(self.filter.inIgnoreIPList('192.0'))
        self.assertLogged('Argument must be a single valid IP.')
        self.pruneLog()
        self.filter.ignoreCommand = cmd + ' bad arguments <ip>'
        self.assertFalse(self.filter.inIgnoreIPList('192.0'))
        self.assertLogged('Usage')

class LogFile(LogCaptureTestCase):
    MISSING = 'testcases/missingLogFile'

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        LogCaptureTestCase.setUp(self)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        LogCaptureTestCase.tearDown(self)

    def testMissingLogFiles(self):
        if False:
            return 10
        self.filter = FilterPoll(None)
        self.assertRaises(IOError, self.filter.addLogPath, LogFile.MISSING)

    def testDecodeLineWarn(self):
        if False:
            while True:
                i = 10
        l = 'correct line\n'
        r = l.encode('utf-16le')
        self.assertEqual(FileContainer.decode_line('TESTFILE', 'utf-16le', r), l)
        self.assertEqual(FileContainer.decode_line('TESTFILE', 'utf-16le', r[0:-1]), l[0:-1])
        self.assertNotLogged('Error decoding line')
        r = b'incorrect \xc8\n line\n'
        l = r.decode('utf-8', 'replace')
        self.assertEqual(FileContainer.decode_line('TESTFILE', 'utf-8', r), l)
        self.assertLogged('Error decoding line')

class LogFileFilterPoll(unittest.TestCase):
    FILENAME = os.path.join(TEST_FILES_DIR, 'testcase01.log')

    def setUp(self):
        if False:
            return 10
        'Call before every test case.'
        super(LogFileFilterPoll, self).setUp()
        self.filter = FilterPoll(DummyJail())
        self.filter.addLogPath(LogFileFilterPoll.FILENAME)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        'Call after every test case.'
        super(LogFileFilterPoll, self).tearDown()

    def testIsModified(self):
        if False:
            print('Hello World!')
        self.assertTrue(self.filter.isModified(LogFileFilterPoll.FILENAME))
        self.assertFalse(self.filter.isModified(LogFileFilterPoll.FILENAME))

    def testSeekToTimeSmallFile(self):
        if False:
            print('Hello World!')
        self.filter.setDatePattern('^%ExY-%Exm-%Exd %ExH:%ExM:%ExS')
        fname = tempfile.mktemp(prefix='tmp_fail2ban', suffix='.log')
        time = 1417512352
        f = open(fname, 'wb')
        fc = None
        try:
            fc = FileContainer(fname, self.filter.getLogEncoding())
            fc.open()
            fc.setPos(0)
            self.filter.seekToTime(fc, time)
            f.flush()
            fc.setPos(0)
            self.filter.seekToTime(fc, time)
            self.assertEqual(fc.getPos(), 0)
            f.write(b'%s [sshd] error: PAM: failure len 1\n' % _tmb(time))
            f.flush()
            fc.setPos(0)
            self.filter.seekToTime(fc, time)
            f.seek(0)
            f.truncate()
            fc.close()
            fc = FileContainer(fname, self.filter.getLogEncoding())
            fc.open()
            for i in range(10):
                f.write(b'[sshd] error: PAM: failure len 1\n')
                f.flush()
                fc.setPos(0)
                self.filter.seekToTime(fc, time)
            f.seek(0)
            f.truncate()
            fc.close()
            fc = FileContainer(fname, self.filter.getLogEncoding())
            fc.open()
            f.write(b'%s [sshd] error: PAM: failure len 2\n' % _tmb(time - 10))
            f.flush()
            fc.setPos(0)
            self.filter.seekToTime(fc, time)
            self.assertEqual(fc.getPos(), 53)
            f.write(b'%s [sshd] error: PAM: failure len 3 2 1\n' % _tmb(time - 9))
            f.flush()
            fc.setPos(0)
            self.filter.seekToTime(fc, time)
            self.assertEqual(fc.getPos(), 110)
            f.write(b'%s [sshd] error: PAM: failure\n' % _tmb(time - 1))
            f.flush()
            self.assertEqual(fc.getFileSize(), 157)
            fc.setPos(0)
            self.filter.seekToTime(fc, time)
            self.assertEqual(fc.getPos(), 157)
            f.write(b'%s [sshd] error: PAM: Authentication failure\n' % _tmb(time))
            f.write(b'%s [sshd] error: PAM: failure len 1\n' % _tmb(time))
            f.flush()
            fc.setPos(0)
            self.filter.seekToTime(fc, time)
            self.assertEqual(fc.getPos(), 157)
            f.write(b'%s [sshd] error: PAM: failure len 3 2 1\n' % _tmb(time + 2))
            f.write(b'%s [sshd] error: PAM: Authentication failure\n' % _tmb(time + 3))
            f.flush()
            fc.setPos(0)
            self.filter.seekToTime(fc, time)
            self.assertEqual(fc.getPos(), 157)
            f.write(b'%s [sshd] error: PAM: failure\n' % _tmb(time + 9))
            f.write(b'%s [sshd] error: PAM: failure len 4 3 2\n' % _tmb(time + 9))
            f.flush()
            fc.setPos(0)
            self.filter.seekToTime(fc, time)
            self.assertEqual(fc.getPos(), 157)
            fc.setPos(157)
            self.filter.seekToTime(fc, time)
            self.assertEqual(fc.getPos(), 157)
            fc.setPos(110)
            self.filter.seekToTime(fc, time)
            self.assertEqual(fc.getPos(), 157)
        finally:
            if fc:
                fc.close()
            _killfile(f, fname)

    def testSeekToTimeLargeFile(self):
        if False:
            i = 10
            return i + 15
        self.filter.setDatePattern('^%ExY-%Exm-%Exd %ExH:%ExM:%ExS')
        fname = tempfile.mktemp(prefix='tmp_fail2ban', suffix='.log')
        time = 1417512352
        f = open(fname, 'wb')
        fc = None
        count = 1000 if unittest.F2B.fast else 10000
        try:
            fc = FileContainer(fname, self.filter.getLogEncoding())
            fc.open()
            f.seek(0)
            t = time - count - 1
            for i in range(count):
                f.write(b'%s [sshd] error: PAM: failure\n' % _tmb(t))
                t += 1
            f.flush()
            fc.setPos(0)
            self.filter.seekToTime(fc, time)
            self.assertEqual(fc.getPos(), 47 * count)
            for i in range(10):
                f.write(b'%s [sshd] error: PAM: failure\n' % _tmb(time))
            f.flush()
            fc.setPos(0)
            self.filter.seekToTime(fc, time)
            self.assertEqual(fc.getPos(), 47 * count)
            fc.setPos(4 * count)
            self.filter.seekToTime(fc, time)
            self.assertEqual(fc.getPos(), 47 * count)
            t = time + 1
            for i in range(count // 500):
                for j in range(500):
                    f.write(b'%s [sshd] error: PAM: failure\n' % _tmb(t))
                    t += 1
                f.flush()
                fc.setPos(0)
                self.filter.seekToTime(fc, time)
                self.assertEqual(fc.getPos(), 47 * count)
                fc.setPos(53)
                self.filter.seekToTime(fc, time)
                self.assertEqual(fc.getPos(), 47 * count)
        finally:
            if fc:
                fc.close()
            _killfile(f, fname)

class LogFileMonitor(LogCaptureTestCase):
    """Few more tests for FilterPoll API
	"""

    def setUp(self):
        if False:
            while True:
                i = 10
        'Call before every test case.'
        setUpMyTime()
        LogCaptureTestCase.setUp(self)
        self.filter = self.name = 'NA'
        (_, self.name) = tempfile.mkstemp('fail2ban', 'monitorfailures')
        self.file = open(self.name, 'ab')
        self.filter = FilterPoll(DummyJail())
        self.filter.addLogPath(self.name, autoSeek=False)
        self.filter.active = True
        self.filter.addFailRegex('(?:(?:Authentication failure|Failed [-/\\w+]+) for(?: [iI](?:llegal|nvalid) user)?|[Ii](?:llegal|nvalid) user|ROOT LOGIN REFUSED) .*(?: from|FROM) <HOST>')

    def tearDown(self):
        if False:
            return 10
        tearDownMyTime()
        LogCaptureTestCase.tearDown(self)
        _killfile(self.file, self.name)
        pass

    def isModified(self, delay=2):
        if False:
            return 10
        'Wait up to `delay` sec to assure that it was modified or not\n\t\t'
        return Utils.wait_for(lambda : self.filter.isModified(self.name), _maxWaitTime(delay))

    def notModified(self, delay=2):
        if False:
            return 10
        'Wait up to `delay` sec as long as it was not modified\n\t\t'
        return Utils.wait_for(lambda : not self.filter.isModified(self.name), _maxWaitTime(delay))

    def testUnaccessibleLogFile(self):
        if False:
            while True:
                i = 10
        os.chmod(self.name, 0)
        self.filter.getFailures(self.name)
        failure_was_logged = self._is_logged('Unable to open %s' % self.name)
        is_root = True
        try:
            with open(self.name) as f:
                f.read()
        except IOError:
            is_root = False
        self.assertTrue(failure_was_logged != is_root)

    def testNoLogFile(self):
        if False:
            for i in range(10):
                print('nop')
        _killfile(self.file, self.name)
        self.filter.getFailures(self.name)
        self.assertLogged('Unable to open %s' % self.name)

    def testErrorProcessLine(self):
        if False:
            return 10
        self.filter.setDatePattern('^%ExY-%Exm-%Exd %ExH:%ExM:%ExS')
        self.filter.sleeptime /= 1000.0
        _org_processLine = self.filter.processLine
        self.filter.processLine = None
        for i in range(100):
            self.file.write(b'line%d\n' % 1)
        self.file.flush()
        for i in range(100):
            self.filter.getFailures(self.name)
        self.assertLogged('Failed to process line:')
        self.assertLogged('Too many errors at once')
        self.pruneLog()
        self.assertTrue(self.filter.idle)
        self.filter.idle = False
        self.filter.getFailures(self.name)
        self.filter.processLine = _org_processLine
        self.file.write(b'line%d\n' % 1)
        self.file.flush()
        self.filter.getFailures(self.name)
        self.assertNotLogged('Failed to process line:')

    def testRemovingFailRegex(self):
        if False:
            print('Hello World!')
        self.filter.delFailRegex(0)
        self.assertNotLogged('Cannot remove regular expression. Index 0 is not valid')
        self.filter.delFailRegex(0)
        self.assertLogged('Cannot remove regular expression. Index 0 is not valid')

    def testRemovingIgnoreRegex(self):
        if False:
            for i in range(10):
                print('nop')
        self.filter.delIgnoreRegex(0)
        self.assertLogged('Cannot remove regular expression. Index 0 is not valid')

    def testNewChangeViaIsModified(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(self.isModified())
        self.assertTrue(self.notModified())
        self.assertTrue(self.notModified())
        mtimesleep()
        for i in range(4):
            self.file.write(b'line%d\n' % i)
            self.file.flush()
            self.assertTrue(self.isModified())
            self.assertTrue(self.notModified())
            mtimesleep()
        os.rename(self.name, self.name + '.old')
        self.assertTrue(self.notModified(1))
        f = open(self.name, 'ab')
        self.assertTrue(self.isModified())
        self.assertTrue(self.notModified())
        mtimesleep()
        f.write(b'line%d\n' % i)
        f.flush()
        self.assertTrue(self.isModified())
        self.assertTrue(self.notModified())
        _killfile(f, self.name)
        _killfile(self.name, self.name + '.old')
        pass

    def testNewChangeViaGetFailures_simple(self):
        if False:
            i = 10
            return i + 15
        self.filter.setDatePattern('^(?:%a )?%b %d %H:%M:%S(?:\\.%f)?(?: %ExY)?')
        self.filter.getFailures(self.name)
        self.assertRaises(FailManagerEmpty, self.filter.failManager.toBan)
        _copy_lines_between_files(GetFailures.FILENAME_01, self.file, n=5)
        self.filter.getFailures(self.name)
        self.assertRaises(FailManagerEmpty, self.filter.failManager.toBan)
        _copy_lines_between_files(GetFailures.FILENAME_01, self.file, skip=12, n=3)
        self.filter.getFailures(self.name)
        _assert_correct_last_attempt(self, self.filter, GetFailures.FAILURES_01)

    def testNewChangeViaGetFailures_rewrite(self):
        if False:
            return 10
        self.filter.setDatePattern('^(?:%a )?%b %d %H:%M:%S(?:\\.%f)?(?: %ExY)?')
        self.file.close()
        _copy_lines_between_files(GetFailures.FILENAME_01, self.name).close()
        self.filter.getFailures(self.name)
        _assert_correct_last_attempt(self, self.filter, GetFailures.FAILURES_01)
        self.file = _copy_lines_between_files(GetFailures.FILENAME_01, self.name, skip=12, n=3, mode='w')
        self.filter.getFailures(self.name)
        _assert_correct_last_attempt(self, self.filter, GetFailures.FAILURES_01)

    def testNewChangeViaGetFailures_move(self):
        if False:
            return 10
        self.filter.setDatePattern('^(?:%a )?%b %d %H:%M:%S(?:\\.%f)?(?: %ExY)?')
        self.file.close()
        self.file = _copy_lines_between_files(GetFailures.FILENAME_01, self.name, n=14, mode='w')
        self.filter.getFailures(self.name)
        self.assertRaises(FailManagerEmpty, self.filter.failManager.toBan)
        self.assertEqual(self.filter.failManager.getFailTotal(), 2)
        os.rename(self.name, self.name + '.bak')
        _copy_lines_between_files(GetFailures.FILENAME_01, self.name, skip=14, n=1).close()
        self.filter.getFailures(self.name)
        self.assertEqual(self.filter.failManager.getFailTotal(), 3)

class CommonMonitorTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        'Call before every test case.'
        super(CommonMonitorTestCase, self).setUp()
        self._failTotal = 0

    def tearDown(self):
        if False:
            print('Hello World!')
        super(CommonMonitorTestCase, self).tearDown()
        self.assertFalse(hasattr(self, '_unexpectedError'))

    def waitFailTotal(self, count, delay=1):
        if False:
            for i in range(10):
                print('nop')
        'Wait up to `delay` sec to assure that expected failure `count` reached\n\t\t'
        ret = Utils.wait_for(lambda : self.filter.failManager.getFailTotal() >= self._failTotal + count and self.jail.isFilled(), _maxWaitTime(delay))
        self._failTotal += count
        return ret

    def isFilled(self, delay=1):
        if False:
            print('Hello World!')
        'Wait up to `delay` sec to assure that it was modified or not\n\t\t'
        return Utils.wait_for(self.jail.isFilled, _maxWaitTime(delay))

    def isEmpty(self, delay=5):
        if False:
            return 10
        'Wait up to `delay` sec to assure that it empty again\n\t\t'
        return Utils.wait_for(self.jail.isEmpty, _maxWaitTime(delay))

    def waitForTicks(self, ticks, delay=2):
        if False:
            while True:
                i = 10
        'Wait up to `delay` sec to assure that it was modified or not\n\t\t'
        last_ticks = self.filter.ticks
        return Utils.wait_for(lambda : self.filter.ticks >= last_ticks + ticks, _maxWaitTime(delay))

    def commonFltError(self, reason='common', exc=None):
        if False:
            return 10
        ' Mock-up for default common error handler to find catched unhandled exceptions\n\t\tcould occur in filters\n\t\t'
        self._commonFltError(reason, exc)
        if reason == 'unhandled':
            DefLogSys.critical('Caught unhandled exception in main cycle of %r : %r', self.filter, exc, exc_info=True)
            self._unexpectedError = True

def get_monitor_failures_testcase(Filter_):
    if False:
        return 10
    "Generator of TestCase's for different filters/backends\n\t"
    testclass_name = tempfile.mktemp('fail2ban', 'monitorfailures_%s_' % (Filter_.__name__,))

    class MonitorFailures(CommonMonitorTestCase):
        count = 0

        def setUp(self):
            if False:
                i = 10
                return i + 15
            'Call before every test case.'
            super(MonitorFailures, self).setUp()
            setUpMyTime()
            self.filter = self.name = 'NA'
            self.name = '%s-%d' % (testclass_name, self.count)
            MonitorFailures.count += 1
            self.file = open(self.name, 'ab')
            self.jail = DummyJail()
            self.filter = Filter_(self.jail)
            (self._commonFltError, self.filter.commonError) = (self.filter.commonError, self.commonFltError)
            self.filter.addLogPath(self.name, autoSeek=False)
            self.filter.setDatePattern('^(?:%a )?%b %d %H:%M:%S(?:\\.%f)?(?: %ExY)?')
            self.filter.active = True
            self.filter.addFailRegex('(?:(?:Authentication failure|Failed [-/\\w+]+) for(?: [iI](?:llegal|nvalid) user)?|[Ii](?:llegal|nvalid) user|ROOT LOGIN REFUSED) .*(?: from|FROM) <HOST>')
            self.filter.start()
            self._sleep_4_poll()

        def tearDown(self):
            if False:
                while True:
                    i = 10
            tearDownMyTime()
            self.filter.stop()
            self.filter.join()
            _killfile(self.file, self.name)
            super(MonitorFailures, self).tearDown()

        def _sleep_4_poll(self):
            if False:
                print('Hello World!')
            if isinstance(self.filter, FilterPoll):
                Utils.wait_for(self.filter.isAlive, _maxWaitTime(5))

        def assert_correct_last_attempt(self, failures, count=None):
            if False:
                i = 10
                return i + 15
            self.assertTrue(self.waitFailTotal(count if count else failures[1], 10))
            _assert_correct_last_attempt(self, self.jail, failures, count=count)

        def test_grow_file(self):
            if False:
                i = 10
                return i + 15
            self._test_grow_file()

        def test_grow_file_in_idle(self):
            if False:
                i = 10
                return i + 15
            self._test_grow_file(True)

        def _test_grow_file(self, idle=False):
            if False:
                return 10
            if idle:
                self.filter.sleeptime /= 100.0
                self.filter.idle = True
                self.waitForTicks(1)
            self.assertRaises(FailManagerEmpty, self.filter.failManager.toBan)
            _copy_lines_between_files(GetFailures.FILENAME_01, self.file, n=12)
            self.assertRaises(FailManagerEmpty, self.filter.failManager.toBan)
            self.assertFalse(len(self.jail))
            _copy_lines_between_files(GetFailures.FILENAME_01, self.file, skip=12, n=3)
            if idle:
                self.waitForTicks(1)
                self.assertTrue(self.isEmpty(1))
                return
            self.assertTrue(self.isFilled(10))
            self.assertEqual(len(self.jail), 1)
            self.assertRaises(FailManagerEmpty, self.filter.failManager.toBan)
            self.assert_correct_last_attempt(GetFailures.FAILURES_01)
            self.assertEqual(len(self.jail), 0)
            _copy_lines_between_files(GetFailures.FILENAME_01, self.file, skip=12, n=3)
            self.assert_correct_last_attempt(GetFailures.FAILURES_01)

        def test_rewrite_file(self):
            if False:
                print('Hello World!')
            self.file.close()
            _copy_lines_between_files(GetFailures.FILENAME_01, self.name).close()
            self.assert_correct_last_attempt(GetFailures.FAILURES_01)
            self.file = _copy_lines_between_files(GetFailures.FILENAME_01, self.name, skip=12, n=3, mode='w')
            self.assert_correct_last_attempt(GetFailures.FAILURES_01)

        def _wait4failures(self, count=2, waitEmpty=True):
            if False:
                i = 10
                return i + 15
            if waitEmpty:
                self.assertTrue(self.isEmpty(_maxWaitTime(5)), 'Queue must be empty but it is not: %s.' % ', '.join([str(x) for x in self.jail.queue]))
                self.assertRaises(FailManagerEmpty, self.filter.failManager.toBan)
            Utils.wait_for(lambda : self.filter.failManager.getFailTotal() >= count, _maxWaitTime(10))
            self.assertEqual(self.filter.failManager.getFailTotal(), count)

        def test_move_file(self):
            if False:
                i = 10
                return i + 15
            self.file.close()
            self.file = _copy_lines_between_files(GetFailures.FILENAME_01, self.name, n=14, mode='w')
            self._wait4failures()
            os.rename(self.name, self.name + '.bak')
            _copy_lines_between_files(GetFailures.FILENAME_01, self.name, skip=14, n=1, lines=['Aug 14 11:59:59 [logrotate] rotation 1']).close()
            self.assert_correct_last_attempt(GetFailures.FAILURES_01)
            self.assertEqual(self.filter.failManager.getFailTotal(), 3)
            _killfile(None, self.name + '.bak')
            _copy_lines_between_files(GetFailures.FILENAME_01, self.name, skip=12, n=3, lines=['Aug 14 11:59:59 [logrotate] rotation 2']).close()
            self.assert_correct_last_attempt(GetFailures.FAILURES_01)
            self.assertEqual(self.filter.failManager.getFailTotal(), 6)

        def test_pyinotify_delWatch(self):
            if False:
                print('Hello World!')
            if hasattr(self.filter, '_delWatch'):
                m = self.filter._FilterPyinotify__monitor
                self.assertTrue(self.filter._delWatch(m.get_wd(self.name)))
                _org_get_path = m.get_path

                def _get_path(wd):
                    if False:
                        for i in range(10):
                            print('nop')
                    return 'test'
                m.get_path = _get_path
                self.assertFalse(self.filter._delWatch(2147483647))
                m.get_path = _org_get_path

        def test_del_file(self):
            if False:
                i = 10
                return i + 15
            self.file.close()
            self.waitForTicks(1)
            os.unlink(self.name)
            self.waitForTicks(2)
            if hasattr(self.filter, 'getPendingPaths'):
                self.assertTrue(Utils.wait_for(lambda : self.name in self.filter.getPendingPaths(), _maxWaitTime(10)))
                self.assertEqual(len(self.filter.getPendingPaths()), 1)

        @with_tmpdir
        def test_move_dir(self, tmp):
            if False:
                i = 10
                return i + 15
            self.file.close()
            self.filter.setMaxRetry(10)
            self.filter.delLogPath(self.name)
            _killfile(None, self.name)
            tmpsub1 = os.path.join(tmp, '1')
            tmpsub2 = os.path.join(tmp, '2')
            os.mkdir(tmpsub1)
            self.name = os.path.join(tmpsub1, os.path.basename(self.name))
            os.close(os.open(self.name, os.O_CREAT | os.O_APPEND))
            self.filter.addLogPath(self.name, autoSeek=False)
            self.file = _copy_lines_between_files(GetFailures.FILENAME_01, self.name, skip=12, n=1, mode='w')
            self.file.close()
            self._wait4failures(1)
            os.rename(tmpsub1, tmpsub2 + 'a')
            os.mkdir(tmpsub1)
            self.file = _copy_lines_between_files(GetFailures.FILENAME_01, self.name, skip=12, n=1, mode='w', lines=['Aug 14 11:59:59 [logrotate] rotation 1'])
            self.file.close()
            self._wait4failures(2)
            os.rename(tmpsub1, tmpsub2 + 'b')
            self.waitForTicks(2)
            os.mkdir(tmpsub1)
            self.waitForTicks(2)
            self.file = _copy_lines_between_files(GetFailures.FILENAME_01, self.name, skip=12, n=1, mode='w', lines=['Aug 14 11:59:59 [logrotate] rotation 2'])
            self.file.close()
            self._wait4failures(3)
            self.filter.stop()
            self.filter.join()

        def _test_move_into_file(self, interim_kill=False):
            if False:
                return 10
            _copy_lines_between_files(GetFailures.FILENAME_01, self.name).close()
            self.assert_correct_last_attempt(GetFailures.FAILURES_01)
            self.assertEqual(self.filter.failManager.getFailTotal(), 3)
            if interim_kill:
                _killfile(None, self.name)
                time.sleep(Utils.DEFAULT_SHORT_INTERVAL)
            _copy_lines_between_files(GetFailures.FILENAME_01, self.name + '.new', skip=12, n=3).close()
            os.rename(self.name + '.new', self.name)
            self.assert_correct_last_attempt(GetFailures.FAILURES_01)
            self.assertEqual(self.filter.failManager.getFailTotal(), 6)
            _copy_lines_between_files(GetFailures.FILENAME_01, self.name, skip=12, n=3).close()
            self.assert_correct_last_attempt(GetFailures.FAILURES_01)
            self.assertEqual(self.filter.failManager.getFailTotal(), 9)

        def test_move_into_file(self):
            if False:
                print('Hello World!')
            self._test_move_into_file(interim_kill=False)

        def test_move_into_file_after_removed(self):
            if False:
                return 10
            self._test_move_into_file(interim_kill=True)

        def test_new_bogus_file(self):
            if False:
                i = 10
                return i + 15
            _copy_lines_between_files(GetFailures.FILENAME_01, self.name, n=100).close()
            self.assert_correct_last_attempt(GetFailures.FAILURES_01)
            open(self.name + '.bak2', 'w').close()
            _copy_lines_between_files(GetFailures.FILENAME_01, self.name, skip=12, n=3).close()
            self.assert_correct_last_attempt(GetFailures.FAILURES_01)
            self.assertEqual(self.filter.failManager.getFailTotal(), 6)
            _killfile(None, self.name + '.bak2')

        def test_delLogPath(self):
            if False:
                print('Hello World!')
            _copy_lines_between_files(GetFailures.FILENAME_01, self.file, n=100)
            self.assert_correct_last_attempt(GetFailures.FAILURES_01)
            self.filter.delLogPath(self.name)
            self.waitForTicks(2)
            _copy_lines_between_files(GetFailures.FILENAME_01, self.file, n=100)
            self.assertTrue(self.isEmpty(10))
            self.filter.addLogPath(self.name, autoSeek=False)
            self.waitForTicks(2)
            self.assert_correct_last_attempt(GetFailures.FAILURES_01, count=3)
            _copy_lines_between_files(GetFailures.FILENAME_01, self.file, skip=12, n=3)
            self.assert_correct_last_attempt(GetFailures.FAILURES_01, count=3)
            self._wait4failures(12, False)
    cls = MonitorFailures
    cls.__qualname__ = cls.__name__ = 'MonitorFailures<%s>(%s)' % (Filter_.__name__, testclass_name)
    return cls

def get_monitor_failures_journal_testcase(Filter_):
    if False:
        print('Hello World!')
    "Generator of TestCase's for journal based filters/backends\n\t"
    testclass_name = 'monitorjournalfailures_%s' % (Filter_.__name__,)

    class MonitorJournalFailures(CommonMonitorTestCase):

        def setUp(self):
            if False:
                for i in range(10):
                    print('nop')
            'Call before every test case.'
            super(MonitorJournalFailures, self).setUp()
            self.test_file = os.path.join(TEST_FILES_DIR, 'testcase-journal.log')
            self.jail = DummyJail()
            self.filter = None
            self.test_uuid = str(uuid.uuid4())
            self.name = '%s-%s' % (testclass_name, self.test_uuid)
            self.journal_fields = {'TEST_FIELD': '1', 'TEST_UUID': self.test_uuid}

        def _initFilter(self, **kwargs):
            if False:
                print('Hello World!')
            self._getRuntimeJournal()
            self.filter = Filter_(self.jail, **kwargs)
            (self._commonFltError, self.filter.commonError) = (self.filter.commonError, self.commonFltError)
            self.filter.addJournalMatch(['SYSLOG_IDENTIFIER=fail2ban-testcases', 'TEST_FIELD=1', 'TEST_UUID=%s' % self.test_uuid])
            self.filter.addJournalMatch(['SYSLOG_IDENTIFIER=fail2ban-testcases', 'TEST_FIELD=2', 'TEST_UUID=%s' % self.test_uuid])
            self.filter.addFailRegex('(?:(?:Authentication failure|Failed [-/\\w+]+) for(?: [iI](?:llegal|nvalid) user)?|[Ii](?:llegal|nvalid) user|ROOT LOGIN REFUSED) .*(?: from|FROM) <HOST>')

        def tearDown(self):
            if False:
                print('Hello World!')
            if self.filter and self.filter.active:
                self.filter.stop()
                self.filter.join()
            super(MonitorJournalFailures, self).tearDown()

        def _getRuntimeJournal(self):
            if False:
                for i in range(10):
                    print('nop')
            'Retrieve current system journal path\n\n\t\t\tIf not found, SkipTest exception will be raised.\n\t\t\t'
            if not hasattr(MonitorJournalFailures, '_runtimeJournal'):
                for systemd_var in ('system-runtime-logs', 'system-state-logs'):
                    tmp = Utils.executeCmd('find "$(systemd-path %s)/journal" -name system.journal -readable' % systemd_var, timeout=10, shell=True, output=True)
                    self.assertTrue(tmp)
                    out = str(tmp[1].decode('utf-8')).split('\n')[0]
                    if out:
                        break
                if os.geteuid() != 0 and os.getenv('F2B_SYSTEMD_DEFAULT_FLAGS', None) is None:
                    os.environ['F2B_SYSTEMD_DEFAULT_FLAGS'] = '0'
                MonitorJournalFailures._runtimeJournal = out
            if MonitorJournalFailures._runtimeJournal:
                return MonitorJournalFailures._runtimeJournal
            raise unittest.SkipTest('systemd journal seems to be not available (e. g. no rights to read)')

        def testJournalFilesArg(self):
            if False:
                while True:
                    i = 10
            jrnlfile = self._getRuntimeJournal()
            self._initFilter(journalfiles=jrnlfile)

        def testJournalFilesAndFlagsArgs(self):
            if False:
                for i in range(10):
                    print('nop')
            jrnlfile = self._getRuntimeJournal()
            self._initFilter(journalfiles=jrnlfile, journalflags=0)

        def testJournalPathArg(self):
            if False:
                while True:
                    i = 10
            jrnlpath = self._getRuntimeJournal()
            jrnlpath = os.path.dirname(jrnlpath)
            self._initFilter(journalpath=jrnlpath)
            self.filter.seekToTime(datetime.datetime.now() - datetime.timedelta(days=1))
            self.filter.start()
            self.waitForTicks(2)
            self.assertTrue(self.isEmpty(1))
            self.assertEqual(len(self.jail), 0)
            self.assertRaises(FailManagerEmpty, self.filter.failManager.toBan)

        def testJournalFlagsArg(self):
            if False:
                for i in range(10):
                    print('nop')
            self._initFilter(journalflags=0)

        def assert_correct_ban(self, test_ip, test_attempts):
            if False:
                return 10
            self.assertTrue(self.waitFailTotal(test_attempts, 10))
            ticket = self.jail.getFailTicket()
            self.assertTrue(ticket)
            attempts = ticket.getAttempt()
            ip = ticket.getID()
            ticket.getMatches()
            self.assertEqual(ip, test_ip)
            self.assertEqual(attempts, test_attempts)

        def test_grow_file(self):
            if False:
                i = 10
                return i + 15
            self._test_grow_file()

        def test_grow_file_in_idle(self):
            if False:
                print('Hello World!')
            self._test_grow_file(True)

        def _test_grow_file(self, idle=False):
            if False:
                for i in range(10):
                    print('nop')
            self._initFilter()
            self.filter.start()
            if idle:
                self.filter.sleeptime /= 100.0
                self.filter.idle = True
            self.waitForTicks(1)
            self.assertRaises(FailManagerEmpty, self.filter.failManager.toBan)
            _copy_lines_to_journal(self.test_file, self.journal_fields, n=2)
            self.assertRaises(FailManagerEmpty, self.filter.failManager.toBan)
            self.assertFalse(len(self.jail))
            _copy_lines_to_journal(self.test_file, self.journal_fields, skip=2, n=3)
            if idle:
                self.waitForTicks(1)
                self.assertTrue(self.isEmpty(1))
                return
            self.assertTrue(self.isFilled(10))
            self.assertEqual(len(self.jail), 1)
            self.assertRaises(FailManagerEmpty, self.filter.failManager.toBan)
            self.assert_correct_ban('193.168.0.128', 3)
            self.assertEqual(len(self.jail), 0)
            _copy_lines_to_journal(self.test_file, self.journal_fields, skip=5, n=4)
            self.assert_correct_ban('193.168.0.128', 3)

        @with_alt_time
        def test_grow_file_with_db(self):
            if False:
                while True:
                    i = 10

            def _gen_falure(ip):
                if False:
                    while True:
                        i = 10
                fields = self.journal_fields
                fields.update(TEST_JOURNAL_FIELDS)
                journal.send(MESSAGE='error: PAM: Authentication failure for test from ' + ip, **fields)
                self.waitForTicks(1)
                self.assert_correct_ban(ip, 1)
            self.jail.database = getFail2BanDb(':memory:')
            self.jail.database.addJail(self.jail)
            MyTime.setTime(time.time())
            self._test_grow_file()
            self.filter.stop()
            self.filter.join()
            MyTime.setTime(time.time() + 10)
            self.jail.database.updateJournal(self.jail, 'systemd-journal', MyTime.time(), 'TEST')
            self._failTotal = 0
            self._initFilter()
            self.filter.setMaxRetry(1)
            self.filter.start()
            self.waitForTicks(2)
            _gen_falure('192.0.2.5')
            self.assertFalse(self.jail.getFailTicket())
            self.filter.stop()
            self.filter.join()
            MyTime.setTime(time.time() + 10000)
            self._failTotal = 0
            self._initFilter()
            self.filter.setMaxRetry(1)
            self.filter.start()
            self.waitForTicks(2)
            MyTime.setTime(time.time() + 20)
            _gen_falure('192.0.2.6')
            self.assertFalse(self.jail.getFailTicket())
            self.filter.stop()
            self.filter.join()
            self.jail.database.updateJournal(self.jail, 'systemd-journal', MyTime.time() - 10000, 'TEST')
            self._initFilter()
            self.filter.setMaxRetry(1)
            states = []

            def _state(*args):
                if False:
                    for i in range(10):
                        print('nop')
                try:
                    self.assertNotIn('** in operation', states)
                    self.assertFalse(self.filter.inOperation)
                    states.append('** process line: %r' % (args,))
                except Exception as e:
                    states.append('** failed: %r' % (e,))
                    raise
            self.filter.processLineAndAdd = _state

            def _inoper():
                if False:
                    print('Hello World!')
                try:
                    self.assertNotIn('** in operation', states)
                    self.assertEqual(len(states), 11)
                    states.append('** in operation')
                    self.filter.__class__.inOperationMode(self.filter)
                except Exception as e:
                    states.append('** failed: %r' % (e,))
                    raise
            self.filter.inOperationMode = _inoper
            self.filter.start()
            self.waitForTicks(12)
            self.assertTrue(Utils.wait_for(lambda : len(states) == 12, _maxWaitTime(10)))
            self.assertEqual(states[-1], '** in operation')

        def test_delJournalMatch(self):
            if False:
                return 10
            self._initFilter()
            self.filter.start()
            self.waitForTicks(1)
            _copy_lines_to_journal(self.test_file, self.journal_fields, n=5)
            self.assert_correct_ban('193.168.0.128', 3)
            self.filter.delJournalMatch(['SYSLOG_IDENTIFIER=fail2ban-testcases', 'TEST_FIELD=1', 'TEST_UUID=%s' % self.test_uuid])
            _copy_lines_to_journal(self.test_file, self.journal_fields, n=5, skip=5)
            self.assertTrue(self.isEmpty(10))
            self.filter.addJournalMatch(['SYSLOG_IDENTIFIER=fail2ban-testcases', 'TEST_FIELD=1', 'TEST_UUID=%s' % self.test_uuid])
            self.assert_correct_ban('193.168.0.128', 3)
            _copy_lines_to_journal(self.test_file, self.journal_fields, n=6, skip=10)
            self.assertTrue(self.isFilled(10))

        def test_WrongChar(self):
            if False:
                while True:
                    i = 10
            self._initFilter()
            self.filter.start()
            self.waitForTicks(1)
            _copy_lines_to_journal(self.test_file, self.journal_fields, skip=15, n=4)
            self.waitForTicks(1)
            self.assertTrue(self.isFilled(10))
            self.assert_correct_ban('87.142.124.10', 3)
            for l in ('error: PAM: Authentication failure for äöüß from 192.0.2.1', 'error: PAM: Authentication failure for äöüß from 192.0.2.1', b'error: PAM: Authentication failure for \xe4\xf6\xfc\xdf from 192.0.2.1'.decode('utf-8', 'replace'), 'error: PAM: Authentication failure for Ã¤Ã¶Ã¼Ã\x9f from 192.0.2.2', 'error: PAM: Authentication failure for Ã¤Ã¶Ã¼Ã\x9f from 192.0.2.2', b'error: PAM: Authentication failure for \xc3\xa4\xc3\xb6\xc3\xbc\xc3\x9f from 192.0.2.2'.decode('utf-8', 'replace')):
                fields = self.journal_fields
                fields.update(TEST_JOURNAL_FIELDS)
                journal.send(MESSAGE=l, **fields)
            self.waitForTicks(1)
            self.waitFailTotal(6, 10)
            self.assertTrue(Utils.wait_for(lambda : len(self.jail) == 2, 10))
            self.assertSortedEqual([self.jail.getFailTicket().getID(), self.jail.getFailTicket().getID()], ['192.0.2.1', '192.0.2.2'])
    cls = MonitorJournalFailures
    cls.__qualname__ = cls.__name__ = 'MonitorJournalFailures<%s>(%s)' % (Filter_.__name__, testclass_name)
    return cls

class GetFailures(LogCaptureTestCase):
    FILENAME_01 = os.path.join(TEST_FILES_DIR, 'testcase01.log')
    FILENAME_02 = os.path.join(TEST_FILES_DIR, 'testcase02.log')
    FILENAME_03 = os.path.join(TEST_FILES_DIR, 'testcase03.log')
    FILENAME_04 = os.path.join(TEST_FILES_DIR, 'testcase04.log')
    FILENAME_USEDNS = os.path.join(TEST_FILES_DIR, 'testcase-usedns.log')
    FILENAME_MULTILINE = os.path.join(TEST_FILES_DIR, 'testcase-multiline.log')
    FAILURES_01 = ('193.168.0.128', 3, 1124013599.0, ['Aug 14 11:59:59 [sshd] error: PAM: Authentication failure for kevin from 193.168.0.128'] * 3)

    def setUp(self):
        if False:
            print('Hello World!')
        'Call before every test case.'
        LogCaptureTestCase.setUp(self)
        setUpMyTime()
        self.jail = DummyJail()
        self.filter = FileFilter(self.jail)
        self.filter.active = True
        self.filter.setDatePattern('^(?:%a )?%b %d %H:%M:%S(?:\\.%f)?(?: %ExY)?')

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        'Call after every test case.'
        tearDownMyTime()
        LogCaptureTestCase.tearDown(self)

    def testFilterAPI(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.filter.getLogs(), [])
        self.assertEqual(self.filter.getLogCount(), 0)
        self.filter.addLogPath(GetFailures.FILENAME_01, tail=True)
        self.assertEqual(self.filter.getLogCount(), 1)
        self.assertEqual(self.filter.getLogPaths(), [GetFailures.FILENAME_01])
        self.filter.addLogPath(GetFailures.FILENAME_02, tail=True)
        self.assertEqual(self.filter.getLogCount(), 2)
        self.assertSortedEqual(self.filter.getLogPaths(), [GetFailures.FILENAME_01, GetFailures.FILENAME_02])

    def testTail(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.filter.getLogs(), [])
        self.filter.addLogPath(GetFailures.FILENAME_01, tail=True)
        self.assertEqual(self.filter.getLogs()[-1].getPos(), 1653)
        self.filter.getLogs()[-1].close()
        self.assertEqual(self.filter.getLogs()[-1].readline(), '')
        self.filter.delLogPath(GetFailures.FILENAME_01)
        self.assertEqual(self.filter.getLogs(), [])

    def testNoLogAdded(self):
        if False:
            for i in range(10):
                print('nop')
        self.filter.addLogPath(GetFailures.FILENAME_01, tail=True)
        self.assertTrue(self.filter.containsLogPath(GetFailures.FILENAME_01))
        self.filter.delLogPath(GetFailures.FILENAME_01)
        self.assertFalse(self.filter.containsLogPath(GetFailures.FILENAME_01))
        self.assertFalse(self.filter.containsLogPath('unknown.log'))
        self.filter.delLogPath('unknown.log')

    def testGetFailures01(self, filename=None, failures=None):
        if False:
            while True:
                i = 10
        filename = filename or GetFailures.FILENAME_01
        failures = failures or GetFailures.FAILURES_01
        self.filter.addLogPath(filename, autoSeek=0)
        self.filter.addFailRegex('(?:(?:Authentication failure|Failed [-/\\w+]+) for(?: [iI](?:llegal|nvalid) user)?|[Ii](?:llegal|nvalid) user|ROOT LOGIN REFUSED) .*(?: from|FROM) <HOST>$')
        self.filter.getFailures(filename)
        _assert_correct_last_attempt(self, self.filter, failures)

    def testCRLFFailures01(self):
        if False:
            for i in range(10):
                print('nop')
        fname = tempfile.mktemp(prefix='tmp_fail2ban', suffix='crlf')
        try:
            (fin, fout) = (open(GetFailures.FILENAME_01, 'rb'), open(fname, 'wb'))
            for l in fin.read().splitlines():
                fout.write(l + b'\r\n')
            fin.close()
            fout.close()
            self.testGetFailures01(filename=fname)
        finally:
            _killfile(fout, fname)

    def testNLCharAsPartOfUniChar(self):
        if False:
            return 10
        fname = tempfile.mktemp(prefix='tmp_fail2ban', suffix='uni')
        for enc in ('utf-16be', 'utf-16le'):
            self.pruneLog('[test-phase encoding=%s]' % enc)
            try:
                fout = open(fname, 'wb')
                tm = int(time.time())
                for l in ('%s € Failed auth: invalid user TestȊ from 192.0.2.1\n' % tm, '%s € Failed auth: invalid user TestI from 192.0.2.2\n' % tm):
                    fout.write(l.encode(enc))
                fout.close()
                self.filter.setLogEncoding(enc)
                self.filter.addLogPath(fname, autoSeek=0)
                self.filter.setDatePattern(('^EPOCH',))
                self.filter.addFailRegex('Failed .* from <HOST>')
                self.filter.getFailures(fname)
                self.assertLogged('[DummyJail] Found 192.0.2.1', '[DummyJail] Found 192.0.2.2', all=True, wait=True)
            finally:
                _killfile(fout, fname)
                self.filter.delLogPath(fname)
        self.assertEqual(self.filter.failManager.getFailCount(), (2, 4))

    def testGetFailures02(self):
        if False:
            for i in range(10):
                print('nop')
        output = ('141.3.81.106', 4, 1124013539.0, ['Aug 14 11:%d:59 i60p295 sshd[12365]: Failed publickey for roehl from ::ffff:141.3.81.106 port 51332 ssh2' % m for m in (53, 54, 57, 58)])
        self.filter.setMaxRetry(4)
        self.filter.addLogPath(GetFailures.FILENAME_02, autoSeek=0)
        self.filter.addFailRegex('Failed .* from <HOST>')
        self.filter.getFailures(GetFailures.FILENAME_02)
        _assert_correct_last_attempt(self, self.filter, output)

    def testGetFailures03(self):
        if False:
            return 10
        output = ('203.162.223.135', 6, 1124013600.0)
        self.filter.setMaxRetry(6)
        self.filter.addLogPath(GetFailures.FILENAME_03, autoSeek=0)
        self.filter.addFailRegex('error,relay=<HOST>,.*550 User unknown')
        self.filter.getFailures(GetFailures.FILENAME_03)
        _assert_correct_last_attempt(self, self.filter, output)

    def testGetFailures03_InOperation(self):
        if False:
            print('Hello World!')
        output = ('203.162.223.135', 9, 1124013600.0)
        self.filter.setMaxRetry(9)
        self.filter.addLogPath(GetFailures.FILENAME_03, autoSeek=0)
        self.filter.addFailRegex('error,relay=<HOST>,.*550 User unknown')
        self.filter.getFailures(GetFailures.FILENAME_03, inOperation=True)
        _assert_correct_last_attempt(self, self.filter, output)

    def testGetFailures03_Seek1(self):
        if False:
            while True:
                i = 10
        output = ('203.162.223.135', 3, 1124013600.0)
        self.filter.addLogPath(GetFailures.FILENAME_03, autoSeek=output[2] - 4 * 60)
        self.filter.addFailRegex('error,relay=<HOST>,.*550 User unknown')
        self.filter.getFailures(GetFailures.FILENAME_03)
        _assert_correct_last_attempt(self, self.filter, output)

    def testGetFailures03_Seek2(self):
        if False:
            i = 10
            return i + 15
        output = ('203.162.223.135', 2, 1124013600.0)
        self.filter.setMaxRetry(2)
        self.filter.addLogPath(GetFailures.FILENAME_03, autoSeek=output[2])
        self.filter.addFailRegex('error,relay=<HOST>,.*550 User unknown')
        self.filter.getFailures(GetFailures.FILENAME_03)
        _assert_correct_last_attempt(self, self.filter, output)

    def testGetFailures04(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(MyTime.time(), 1124013600)
        output = (('212.41.96.186', 2, 1124013480.0), ('212.41.96.186', 2, 1124013600.0), ('212.41.96.185', 2, 1124013598.0))
        self.filter.setDatePattern(('^%ExY(?P<_sep>[-/.])%m(?P=_sep)%d[T ]%H:%M:%S(?:[.,]%f)?(?:\\s*%z)?', '^(?:%a )?%b %d %H:%M:%S(?:\\.%f)?(?: %ExY)?', '^EPOCH'))
        self.filter.setMaxRetry(2)
        self.filter.addLogPath(GetFailures.FILENAME_04, autoSeek=0)
        self.filter.addFailRegex('Invalid user .* <HOST>')
        self.filter.getFailures(GetFailures.FILENAME_04)
        _assert_correct_last_attempt(self, self.filter, output)

    def testGetFailuresWrongChar(self):
        if False:
            while True:
                i = 10
        self.filter.checkFindTime = False
        fname = tempfile.mktemp(prefix='tmp_fail2ban', suffix='crlf')
        fout = fopen(fname, 'wb')
        try:
            for l in (b'2015-01-14 20:00:58 user "test\xf1ing" from "192.0.2.0"\n', b'2015-01-14 20:00:59 user "\xd1\xe2\xe5\xf2\xe0" from "192.0.2.0"\n', b'2015-01-14 20:01:00 user "testing" from "192.0.2.0"\n'):
                fout.write(l)
            fout.close()
            output = ('192.0.2.0', 3, 1421262060.0)
            failregex = '^\\s*user \\"[^\\"]*\\" from \\"<HOST>\\"\\s*$'
            for enc in (None, 'utf-8', 'ascii'):
                if enc is not None:
                    self.tearDown()
                    self.setUp()
                    if DefLogSys.getEffectiveLevel() > 7:
                        DefLogSys.setLevel(7)
                    self.filter.checkFindTime = False
                    self.filter.setLogEncoding(enc)
                self.filter.setDatePattern('^%ExY-%Exm-%Exd %ExH:%ExM:%ExS')
                self.assertNotLogged('Error decoding line')
                self.filter.addLogPath(fname)
                self.filter.addFailRegex(failregex)
                self.filter.getFailures(fname)
                _assert_correct_last_attempt(self, self.filter, output)
                self.assertLogged('Error decoding line')
                self.assertLogged('Continuing to process line ignoring invalid characters:', '2015-01-14 20:00:58 user ')
                self.assertLogged('Continuing to process line ignoring invalid characters:', '2015-01-14 20:00:59 user ')
        finally:
            _killfile(fout, fname)

    def testGetFailuresUseDNS(self):
        if False:
            print('Hello World!')
        output_yes = (('93.184.216.34', 1, 1124013299.0, ['Aug 14 11:54:59 i60p295 sshd[12365]: Failed publickey for roehl from example.com port 51332 ssh2']), ('93.184.216.34', 1, 1124013539.0, ['Aug 14 11:58:59 i60p295 sshd[12365]: Failed publickey for roehl from ::ffff:93.184.216.34 port 51332 ssh2']), ('2606:2800:220:1:248:1893:25c8:1946', 1, 1124013299.0, ['Aug 14 11:54:59 i60p295 sshd[12365]: Failed publickey for roehl from example.com port 51332 ssh2']))
        if not unittest.F2B.no_network and (not DNSUtils.IPv6IsAllowed()):
            output_yes = output_yes[0:2]
        output_no = ('93.184.216.34', 1, 1124013539.0, ['Aug 14 11:58:59 i60p295 sshd[12365]: Failed publickey for roehl from ::ffff:93.184.216.34 port 51332 ssh2'])
        for (useDns, output) in (('yes', output_yes), ('no', output_no), ('warn', output_yes)):
            self.pruneLog('[test-phase useDns=%s]' % useDns)
            jail = DummyJail()
            filter_ = FileFilter(jail, useDns=useDns)
            filter_.active = True
            filter_.failManager.setMaxRetry(1)
            filter_.addLogPath(GetFailures.FILENAME_USEDNS, autoSeek=False)
            filter_.addFailRegex('Failed .* from <HOST>')
            filter_.getFailures(GetFailures.FILENAME_USEDNS)
            _assert_correct_last_attempt(self, filter_, output)

    def testGetFailuresMultiRegex(self):
        if False:
            for i in range(10):
                print('nop')
        output = [('141.3.81.106', 8, 1124013541.0)]
        self.filter.setMaxRetry(8)
        self.filter.addLogPath(GetFailures.FILENAME_02, autoSeek=False)
        self.filter.addFailRegex('Failed .* from <HOST>')
        self.filter.addFailRegex('Accepted .* from <HOST>')
        self.filter.getFailures(GetFailures.FILENAME_02)
        _assert_correct_last_attempt(self, self.filter, output)

    def testGetFailuresIgnoreRegex(self):
        if False:
            for i in range(10):
                print('nop')
        self.filter.addLogPath(GetFailures.FILENAME_02, autoSeek=False)
        self.filter.addFailRegex('Failed .* from <HOST>')
        self.filter.addFailRegex('Accepted .* from <HOST>')
        self.filter.addIgnoreRegex('for roehl')
        self.filter.getFailures(GetFailures.FILENAME_02)
        self.assertRaises(FailManagerEmpty, self.filter.failManager.toBan)

    def testGetFailuresMultiLine(self):
        if False:
            print('Hello World!')
        output = [('192.0.43.10', 1, 1124013598.0), ('192.0.43.10', 1, 1124013599.0), ('192.0.43.11', 1, 1124013598.0)]
        self.filter.addLogPath(GetFailures.FILENAME_MULTILINE, autoSeek=False)
        self.filter.setMaxLines(100)
        self.filter.addFailRegex('^.*rsyncd\\[(?P<pid>\\d+)\\]: connect from .+ \\(<HOST>\\)$<SKIPLINES>^.+ rsyncd\\[(?P=pid)\\]: rsync error: .*$')
        self.filter.setMaxRetry(1)
        self.filter.getFailures(GetFailures.FILENAME_MULTILINE)
        _assert_correct_last_attempt(self, self.filter, output)

    def testGetFailuresMultiLineIgnoreRegex(self):
        if False:
            print('Hello World!')
        output = [('192.0.43.10', 1, 1124013598.0), ('192.0.43.10', 1, 1124013599.0)]
        self.filter.addLogPath(GetFailures.FILENAME_MULTILINE, autoSeek=False)
        self.filter.setMaxLines(100)
        self.filter.addFailRegex('^.*rsyncd\\[(?P<pid>\\d+)\\]: connect from .+ \\(<HOST>\\)$<SKIPLINES>^.+ rsyncd\\[(?P=pid)\\]: rsync error: .*$')
        self.filter.addIgnoreRegex('rsync error: Received SIGINT')
        self.filter.setMaxRetry(1)
        self.filter.getFailures(GetFailures.FILENAME_MULTILINE)
        _assert_correct_last_attempt(self, self.filter, output)
        self.assertRaises(FailManagerEmpty, self.filter.failManager.toBan)

    def testGetFailuresMultiLineMultiRegex(self):
        if False:
            while True:
                i = 10
        output = [('192.0.43.10', 1, 1124013598.0), ('192.0.43.10', 1, 1124013599.0), ('192.0.43.11', 1, 1124013598.0), ('192.0.43.15', 1, 1124013598.0)]
        self.filter.addLogPath(GetFailures.FILENAME_MULTILINE, autoSeek=False)
        self.filter.setMaxLines(100)
        self.filter.addFailRegex('^.*rsyncd\\[(?P<pid>\\d+)\\]: connect from .+ \\(<HOST>\\)$<SKIPLINES>^.+ rsyncd\\[(?P=pid)\\]: rsync error: .*$')
        self.filter.addFailRegex('^.* sendmail\\[.*, msgid=<(?P<msgid>[^>]+).*relay=\\[<HOST>\\].*$<SKIPLINES>^.+ spamd: result: Y \\d+ .*,mid=<(?P=msgid)>(,bayes=[.\\d]+)?(,autolearn=\\S+)?\\s*$')
        self.filter.setMaxRetry(1)
        self.filter.getFailures(GetFailures.FILENAME_MULTILINE)
        _assert_correct_last_attempt(self, self.filter, output)
        self.assertRaises(FailManagerEmpty, self.filter.failManager.toBan)

class DNSUtilsTests(unittest.TestCase):

    def testCache(self):
        if False:
            return 10
        c = Utils.Cache(maxCount=5, maxTime=60)
        self.assertTrue(c.get('a') is None)
        self.assertEqual(c.get('a', 'test'), 'test')
        for i in range(5):
            c.set(i, i)
        for i in range(5):
            self.assertEqual(c.get(i), i)
        c.unset('a')
        c.unset('a')

    def testCacheMaxSize(self):
        if False:
            return 10
        c = Utils.Cache(maxCount=5, maxTime=60)
        for i in range(5):
            c.set(i, i)
        self.assertEqual([c.get(i) for i in range(5)], [i for i in range(5)])
        self.assertNotIn(-1, (c.get(i, -1) for i in range(5)))
        c.set(10, i)
        self.assertIn(-1, (c.get(i, -1) for i in range(5)))
        for i in range(10):
            c.set(i, 1)
        self.assertEqual(len(c), 5)

    def testCacheMaxTime(self):
        if False:
            i = 10
            return i + 15
        c = Utils.Cache(maxCount=5, maxTime=0.0005)
        for i in range(10):
            c.set(i, 1)
        st = time.time()
        self.assertTrue(Utils.wait_for(lambda : time.time() >= st + 0.0005, 1))
        self.assertTrue(len(c) <= 5)
        for i in range(10):
            self.assertTrue(c.get(i) is None)
        self.assertEqual(len(c), 0)

    def testOverflowedIPCache(self):
        if False:
            for i in range(10):
                print('nop')
        from threading import Thread
        from random import shuffle
        _org_cache = IPAddr.CACHE_OBJ
        cache = IPAddr.CACHE_OBJ = Utils.Cache(maxCount=5, maxTime=60)
        result = list()
        count = 1 if unittest.F2B.fast else 50
        try:

            def _TestCacheStr2IP(forw=True, result=[], random=False):
                if False:
                    while True:
                        i = 10
                try:
                    c = count
                    while c:
                        c -= 1
                        s = range(0, 256, 1) if forw else range(255, -1, -1)
                        if random:
                            shuffle([i for i in s])
                        for i in s:
                            IPAddr('192.0.2.' + str(i), IPAddr.FAM_IPv4)
                            IPAddr('2001:db8::' + str(i), IPAddr.FAM_IPv6)
                    result.append(None)
                except Exception as e:
                    DefLogSys.debug(e, exc_info=True)
                    result.append(e)
            th1 = Thread(target=_TestCacheStr2IP, args=(True, result))
            th1.start()
            th2 = Thread(target=_TestCacheStr2IP, args=(False, result))
            th2.start()
            _TestCacheStr2IP(True, result, True)
        finally:
            th1.join()
            th2.join()
            IPAddr.CACHE_OBJ = _org_cache
        self.assertEqual(result, [None] * 3)
        self.assertTrue(len(cache) <= cache.maxCount)

class DNSUtilsNetworkTests(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        'Call before every test case.'
        super(DNSUtilsNetworkTests, self).setUp()
    EXAMPLE_ADDRS = ['93.184.216.34', '2606:2800:220:1:248:1893:25c8:1946'] if unittest.F2B.no_network or DNSUtils.IPv6IsAllowed() else ['93.184.216.34']

    def test_IPAddr(self):
        if False:
            while True:
                i = 10
        ip4 = IPAddr('192.0.2.1')
        ip6 = IPAddr('2001:DB8::')
        self.assertTrue(ip4.isIPv4)
        self.assertTrue(ip4.isSingle)
        self.assertTrue(ip6.isIPv6)
        self.assertTrue(ip6.isSingle)
        self.assertTrue(asip('192.0.2.1').isIPv4)
        self.assertTrue(id(asip(ip4)) == id(ip4))
        ip6 = IPAddr('::')
        self.assertTrue(ip6.isIPv6)
        self.assertTrue(ip6.isSingle)
        ip6 = IPAddr('::/32')
        self.assertTrue(ip6.isIPv6)
        self.assertFalse(ip6.isSingle)
        for s in ('some/path/as/id', 'other-path/24', '1.2.3.4/path'):
            r = IPAddr(s, IPAddr.CIDR_UNSPEC)
            self.assertEqual(r.raw, s)
            self.assertFalse(r.isIPv4)
            self.assertFalse(r.isIPv6)

    def test_IPAddr_Raw(self):
        if False:
            return 10
        r = IPAddr('xxx', IPAddr.CIDR_RAW)
        self.assertFalse(r.isIPv4)
        self.assertFalse(r.isIPv6)
        self.assertFalse(r.isSingle)
        self.assertTrue(r.isValid)
        self.assertEqual(r, 'xxx')
        self.assertEqual('xxx', str(r))
        self.assertNotEqual(r, IPAddr('xxx'))
        r = IPAddr('1:2', IPAddr.CIDR_RAW)
        self.assertFalse(r.isIPv4)
        self.assertFalse(r.isIPv6)
        self.assertFalse(r.isSingle)
        self.assertTrue(r.isValid)
        self.assertEqual(r, '1:2')
        self.assertEqual('1:2', str(r))
        self.assertNotEqual(r, IPAddr('1:2'))
        r = IPAddr('93.184.0.1', IPAddr.CIDR_RAW)
        ip4 = IPAddr('93.184.0.1')
        self.assertNotEqual(ip4, r)
        self.assertNotEqual(r, ip4)
        self.assertTrue(r < ip4)
        self.assertTrue(r < ip4)
        r = IPAddr('1::2', IPAddr.CIDR_RAW)
        ip6 = IPAddr('1::2')
        self.assertNotEqual(ip6, r)
        self.assertNotEqual(r, ip6)
        self.assertTrue(r < ip6)
        self.assertTrue(r < ip6)

    def testUseDns(self):
        if False:
            return 10
        res = DNSUtils.textToIp('www.example.com', 'no')
        self.assertSortedEqual(res, [])
        res = DNSUtils.textToIp('www.example.com', 'warn')
        self.assertSortedEqual(res, self.EXAMPLE_ADDRS)
        res = DNSUtils.textToIp('www.example.com', 'yes')
        self.assertSortedEqual(res, self.EXAMPLE_ADDRS)

    def testTextToIp(self):
        if False:
            while True:
                i = 10
        hostnames = ['www.example.com', 'doh1.2.3.4.buga.xxxxx.yyy.invalid', '1.2.3.4.buga.xxxxx.yyy.invalid']
        for s in hostnames:
            res = DNSUtils.textToIp(s, 'yes')
            if s == 'www.example.com':
                self.assertSortedEqual(res, self.EXAMPLE_ADDRS)
            else:
                self.assertSortedEqual(res, [])

    def testIpToIp(self):
        if False:
            print('Hello World!')
        for s in self.EXAMPLE_ADDRS:
            ips = DNSUtils.textToIp(s, 'yes')
            self.assertSortedEqual(ips, [s])
            for ip in ips:
                self.assertTrue(isinstance(ip, IPAddr))

    def testIpToName(self):
        if False:
            i = 10
            return i + 15
        res = DNSUtils.ipToName('8.8.4.4')
        self.assertTrue(res.endswith(('.google', '.google.com')))
        res = DNSUtils.ipToName(IPAddr('8.8.4.4'))
        self.assertTrue(res.endswith(('.google', '.google.com')))
        res = DNSUtils.ipToName('192.0.2.0')
        self.assertEqual(res, None)
        res = DNSUtils.ipToName('192.0.2.888')
        self.assertEqual(res, None)

    def testAddr2bin(self):
        if False:
            return 10
        res = IPAddr('10.0.0.0')
        self.assertEqual(res.addr, 167772160)
        res = IPAddr('10.0.0.0', cidr=None)
        self.assertEqual(res.addr, 167772160)
        res = IPAddr('10.0.0.0', cidr=32)
        self.assertEqual(res.addr, 167772160)
        res = IPAddr('10.0.0.1', cidr=32)
        self.assertEqual(res.addr, 167772161)
        self.assertTrue(res.isSingle)
        res = IPAddr('10.0.0.1', cidr=31)
        self.assertEqual(res.addr, 167772160)
        self.assertFalse(res.isSingle)
        self.assertEqual(IPAddr('10.0.0.0').hexdump, '0a000000')
        self.assertEqual(IPAddr('1::2').hexdump, '00010000000000000000000000000002')
        self.assertEqual(IPAddr('xxx').hexdump, '')
        self.assertEqual(IPAddr('192.0.2.0').getPTR(), '0.2.0.192.in-addr.arpa.')
        self.assertEqual(IPAddr('192.0.2.1').getPTR(), '1.2.0.192.in-addr.arpa.')
        self.assertEqual(IPAddr('2606:2800:220:1:248:1893:25c8:1946').getPTR(), '6.4.9.1.8.c.5.2.3.9.8.1.8.4.2.0.1.0.0.0.0.2.2.0.0.0.8.2.6.0.6.2.ip6.arpa.')

    def testIPAddr_Equal6(self):
        if False:
            print('Hello World!')
        self.assertEqual(IPAddr('2606:2800:220:1:248:1893::'), IPAddr('2606:2800:220:1:248:1893:0:0'))
        self.assertEqual(IPAddr('[2606:2800:220:1:248:1893::]'), IPAddr('2606:2800:220:1:248:1893:0:0'))

    def testIPAddr_InInet(self):
        if False:
            while True:
                i = 10
        ip4net = IPAddr('93.184.0.1/24')
        ip6net = IPAddr('2606:2800:220:1:248:1893:25c8:0/120')
        self.assertFalse(ip4net.isSingle)
        self.assertFalse(ip6net.isSingle)
        self.assertTrue(IPAddr('93.184.0.1').isInNet(ip4net))
        self.assertTrue(IPAddr('93.184.0.255').isInNet(ip4net))
        self.assertFalse(IPAddr('93.184.1.0').isInNet(ip4net))
        self.assertFalse(IPAddr('93.184.0.1').isInNet(ip6net))
        self.assertTrue(IPAddr('2606:2800:220:1:248:1893:25c8:1').isInNet(ip6net))
        self.assertTrue(IPAddr('2606:2800:220:1:248:1893:25c8:ff').isInNet(ip6net))
        self.assertFalse(IPAddr('2606:2800:220:1:248:1893:25c8:100').isInNet(ip6net))
        self.assertFalse(IPAddr('2606:2800:220:1:248:1893:25c8:100').isInNet(ip4net))
        self.assertFalse(IPAddr('93.184.0.1', IPAddr.CIDR_RAW).isInNet(ip4net))
        self.assertFalse(IPAddr('2606:2800:220:1:248:1893:25c8:1', IPAddr.CIDR_RAW).isInNet(ip6net))
        self.assertFalse(IPAddr('xxx').isInNet(ip4net))
        ip6net = IPAddr('::/32')
        self.assertTrue(IPAddr('::').isInNet(ip6net))
        self.assertTrue(IPAddr('::1').isInNet(ip6net))
        self.assertTrue(IPAddr('0000::').isInNet(ip6net))
        self.assertTrue(IPAddr('0000::0000').isInNet(ip6net))
        self.assertTrue(IPAddr('0000:0000:7777::').isInNet(ip6net))
        self.assertTrue(IPAddr('0000::7777:7777:7777:7777:7777:7777').isInNet(ip6net))
        self.assertTrue(IPAddr('0000:0000:ffff::').isInNet(ip6net))
        self.assertTrue(IPAddr('0000::ffff:ffff:ffff:ffff:ffff:ffff').isInNet(ip6net))
        self.assertFalse(IPAddr('0000:0001:ffff::').isInNet(ip6net))
        self.assertFalse(IPAddr('1::').isInNet(ip6net))

    def testIPAddr_Compare(self):
        if False:
            for i in range(10):
                print('nop')
        ip4 = [IPAddr('93.184.0.1'), IPAddr('93.184.216.1'), IPAddr('93.184.216.34')]
        ip6 = [IPAddr('2606:2800:220:1:248:1893::'), IPAddr('2606:2800:220:1:248:1893:25c8:0'), IPAddr('2606:2800:220:1:248:1893:25c8:1946')]
        self.assertNotEqual(ip4[0], None)
        self.assertTrue(ip4[0] is not None)
        self.assertFalse(ip4[0] is None)
        self.assertTrue(ip4[0] < ip4[1])
        self.assertTrue(ip4[1] < ip4[2])
        self.assertEqual(sorted(reversed(ip4)), ip4)
        self.assertNotEqual(ip6[0], None)
        self.assertTrue(ip6[0] is not None)
        self.assertFalse(ip6[0] is None)
        self.assertTrue(ip6[0] < ip6[1])
        self.assertTrue(ip6[1] < ip6[2])
        self.assertEqual(sorted(reversed(ip6)), ip6)
        self.assertNotEqual(ip4[0], ip6[0])
        self.assertTrue(ip4[0] < ip6[0])
        self.assertTrue(ip4[2] < ip6[2])
        self.assertEqual(sorted(reversed(ip4 + ip6)), ip4 + ip6)
        d = {'93.184.216.34': 'ip4-test', '2606:2800:220:1:248:1893:25c8:1946': 'ip6-test'}
        d2 = dict([(IPAddr(k), v) for (k, v) in d.items()])
        self.assertTrue(isinstance(list(d.keys())[0], str))
        self.assertTrue(isinstance(list(d2.keys())[0], IPAddr))
        self.assertEqual(d.get(ip4[2], ''), 'ip4-test')
        self.assertEqual(d.get(ip6[2], ''), 'ip6-test')
        self.assertEqual(d2.get(str(ip4[2]), ''), 'ip4-test')
        self.assertEqual(d2.get(str(ip6[2]), ''), 'ip6-test')
        self.assertEqual(d, d2)

    def testIPAddr_CIDR(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(str(IPAddr('93.184.0.1', 24)), '93.184.0.0/24')
        self.assertEqual(str(IPAddr('192.168.1.0/255.255.255.128')), '192.168.1.0/25')
        self.assertEqual(IPAddr('93.184.0.1', 24).ntoa, '93.184.0.0/24')
        self.assertEqual(IPAddr('192.168.1.0/255.255.255.128').ntoa, '192.168.1.0/25')
        self.assertEqual(IPAddr('93.184.0.1/32').ntoa, '93.184.0.1')
        self.assertEqual(IPAddr('93.184.0.1/255.255.255.255').ntoa, '93.184.0.1')
        self.assertEqual(str(IPAddr('2606:2800:220:1:248:1893:25c8::', 120)), '2606:2800:220:1:248:1893:25c8:0/120')
        self.assertEqual(IPAddr('2606:2800:220:1:248:1893:25c8::', 120).ntoa, '2606:2800:220:1:248:1893:25c8:0/120')
        self.assertEqual(str(IPAddr('2606:2800:220:1:248:1893:25c8:0/120')), '2606:2800:220:1:248:1893:25c8:0/120')
        self.assertEqual(IPAddr('2606:2800:220:1:248:1893:25c8:0/120').ntoa, '2606:2800:220:1:248:1893:25c8:0/120')
        self.assertEqual(str(IPAddr('2606:28ff:220:1:248:1893:25c8::', 25)), '2606:2880::/25')
        self.assertEqual(str(IPAddr('2606:28ff:220:1:248:1893:25c8::/ffff:ff80::')), '2606:2880::/25')
        self.assertEqual(str(IPAddr('2606:28ff:220:1:248:1893:25c8::/ffff:ffff:ffff:ffff:ffff:ffff:ffff::')), '2606:28ff:220:1:248:1893:25c8:0/112')
        self.assertEqual(str(IPAddr('2606:28ff:220:1:248:1893:25c8::/128')), '2606:28ff:220:1:248:1893:25c8:0')
        self.assertEqual(str(IPAddr('2606:28ff:220:1:248:1893:25c8::/ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff')), '2606:28ff:220:1:248:1893:25c8:0')

    def testIPAddr_CIDR_Wrong(self):
        if False:
            for i in range(10):
                print('nop')
        s = '2606:28ff:220:1:248:1893:25c8::/ffff::/::1'
        r = IPAddr(s)
        self.assertEqual(r.raw, s)
        self.assertFalse(r.isIPv4)
        self.assertFalse(r.isIPv6)

    def testIPAddr_CIDR_Repr(self):
        if False:
            print('Hello World!')
        self.assertEqual(['127.0.0.0/8', '::/32', '2001:db8::/32'], [IPAddr('127.0.0.0', 8), IPAddr('::1', 32), IPAddr('2001:db8::', 32)])

    def testIPAddr_CompareDNS(self):
        if False:
            i = 10
            return i + 15
        ips = IPAddr('example.com')
        self.assertTrue(IPAddr('93.184.216.34').isInNet(ips))
        self.assertEqual(IPAddr('2606:2800:220:1:248:1893:25c8:1946').isInNet(ips), '2606:2800:220:1:248:1893:25c8:1946' in self.EXAMPLE_ADDRS)

    def testIPAddr_wrongDNS_IP(self):
        if False:
            for i in range(10):
                print('nop')
        unittest.F2B.SkipIfNoNetwork()
        DNSUtils.dnsToIp('`this`.dns-is-wrong.`wrong-nic`-dummy')
        DNSUtils.ipToName('*')

    def testIPAddr_Cached(self):
        if False:
            print('Hello World!')
        ips = [DNSUtils.dnsToIp('example.com'), DNSUtils.dnsToIp('example.com')]
        for (ip1, ip2) in zip(ips, ips):
            self.assertEqual(id(ip1), id(ip2))
        ip1 = IPAddr('93.184.216.34')
        ip2 = IPAddr('93.184.216.34')
        self.assertEqual(id(ip1), id(ip2))
        ip1 = IPAddr('2606:2800:220:1:248:1893:25c8:1946')
        ip2 = IPAddr('2606:2800:220:1:248:1893:25c8:1946')
        self.assertEqual(id(ip1), id(ip2))

    def test_NetworkInterfacesAddrs(self):
        if False:
            return 10
        for withMask in (False, True):
            try:
                ips = IPAddrSet([a for (ni, a) in DNSUtils._NetworkInterfacesAddrs(withMask)])
                ip = IPAddr('127.0.0.1')
                self.assertEqual(ip in ips, any((ip in n for n in ips)))
                ip = IPAddr('::1')
                self.assertEqual(ip in ips, any((ip in n for n in ips)))
            except Exception as e:
                raise unittest.SkipTest(e)

    def test_IPAddrSet(self):
        if False:
            return 10
        ips = IPAddrSet([IPAddr('192.0.2.1/27'), IPAddr('2001:DB8::/32')])
        self.assertTrue(IPAddr('192.0.2.1') in ips)
        self.assertTrue(IPAddr('192.0.2.31') in ips)
        self.assertFalse(IPAddr('192.0.2.32') in ips)
        self.assertTrue(IPAddr('2001:DB8::1') in ips)
        self.assertTrue(IPAddr('2001:0DB8:FFFF:FFFF:FFFF:FFFF:FFFF:FFFF') in ips)
        self.assertFalse(IPAddr('2001:DB9::') in ips)
        for cov in ('ni', 'dns', 'last'):
            _org_NetworkInterfacesAddrs = None
            if cov == 'dns':
                _org_NetworkInterfacesAddrs = DNSUtils._NetworkInterfacesAddrs

                def _tmp_NetworkInterfacesAddrs():
                    if False:
                        print('Hello World!')
                    raise NotImplementedError()
                DNSUtils._NetworkInterfacesAddrs = staticmethod(_tmp_NetworkInterfacesAddrs)
            try:
                ips = DNSUtils.getSelfIPs()
                if ips:
                    ip = IPAddr('127.0.0.1')
                    self.assertEqual(ip in ips, any((ip in n for n in ips)))
                    ip = IPAddr('127.0.0.2')
                    self.assertEqual(ip in ips, any((ip in n for n in ips)))
                    ip = IPAddr('::1')
                    self.assertEqual(ip in ips, any((ip in n for n in ips)))
            finally:
                if _org_NetworkInterfacesAddrs:
                    DNSUtils._NetworkInterfacesAddrs = staticmethod(_org_NetworkInterfacesAddrs)
                if cov != 'last':
                    DNSUtils.CACHE_nameToIp.unset(DNSUtils._getSelfIPs_key)
                    DNSUtils.CACHE_nameToIp.unset(DNSUtils._getNetIntrfIPs_key)

    def testFQDN(self):
        if False:
            print('Hello World!')
        unittest.F2B.SkipIfNoNetwork()
        sname = DNSUtils.getHostname(fqdn=False)
        lname = DNSUtils.getHostname(fqdn=True)
        self.assertEqual(lname != 'localhost', sname != 'localhost')
        self.assertEqual(getfqdn(sname), lname)
        self.assertEqual(getfqdn(lname), lname)
        self.assertIn(getfqdn('localhost.'), ('localhost', 'localhost.'))

    def testFQDN_DNS(self):
        if False:
            while True:
                i = 10
        unittest.F2B.SkipIfNoNetwork()
        self.assertIn(getfqdn('as112.arpa.'), ('as112.arpa.', 'as112.arpa'))

class JailTests(unittest.TestCase):

    def testSetBackend_gh83(self):
        if False:
            while True:
                i = 10
        Jail('test', backend='polling')