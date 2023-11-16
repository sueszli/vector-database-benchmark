__author__ = 'Cyril Jaquier'
__copyright__ = 'Copyright (c) 2004 Cyril Jaquier'
__license__ = 'GPL'
import unittest
import time
import datetime
from ..server.datedetector import DateDetector
from ..server import datedetector
from ..server.datetemplate import DatePatternRegex, DateTemplate
from .utils import setUpMyTime, tearDownMyTime, LogCaptureTestCase
from ..helpers import getLogger
logSys = getLogger('fail2ban')

class DateDetectorTest(LogCaptureTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        'Call before every test case.'
        LogCaptureTestCase.setUp(self)
        setUpMyTime()
        self.__datedetector = None

    def tearDown(self):
        if False:
            print('Hello World!')
        'Call after every test case.'
        LogCaptureTestCase.tearDown(self)
        tearDownMyTime()

    @property
    def datedetector(self):
        if False:
            while True:
                i = 10
        if self.__datedetector is None:
            self.__datedetector = DateDetector()
            self.__datedetector.addDefaultTemplate()
        return self.__datedetector

    def testGetEpochTime(self):
        if False:
            return 10
        self.__datedetector = DateDetector()
        self.__datedetector.appendTemplate('EPOCH')
        for dateUnix in (1138049999, 32535244799):
            for date in ('%s', '[%s]', '[%s.555]', 'audit(%s.555:101)'):
                date = date % dateUnix
                log = date + ' [sshd] error: PAM: Authentication failure'
                datelog = self.datedetector.getTime(log)
                self.assertTrue(datelog, 'Parse epoch time for %s failed' % (date,))
                (datelog, matchlog) = datelog
                self.assertEqual(int(datelog), dateUnix)
                self.assertIn(matchlog.group(1), (str(dateUnix), str(dateUnix) + '.555'))
        for dateUnix in ('123456789', '9999999999999999', '1138049999A', 'A1138049999'):
            for date in ('%s', '[%s]', '[%s.555]', 'audit(%s.555:101)'):
                date = date % dateUnix
                log = date + ' [sshd] error: PAM: Authentication failure'
                datelog = self.datedetector.getTime(log)
                self.assertFalse(datelog)

    def testGetEpochMsTime(self):
        if False:
            while True:
                i = 10
        self.__datedetector = DateDetector()
        self.__datedetector.appendTemplate('LEPOCH')
        for fact in (1, 1000, 1000000):
            for dateUnix in (1138049999, 32535244799):
                for date in ('%s', '[%s]', '[%s]', 'audit(%s:101)'):
                    dateLong = dateUnix * fact
                    date = date % dateLong
                    log = date + ' [sshd] error: PAM: Authentication failure'
                    datelog = self.datedetector.getTime(log)
                    self.assertTrue(datelog, 'Parse epoch time for %s failed' % (date,))
                    (datelog, matchlog) = datelog
                    self.assertEqual(int(datelog), dateUnix)
                    self.assertEqual(matchlog.group(1), str(dateLong))
        for dateUnix in ('123456789', '999999999999999999', '1138049999A', 'A1138049999'):
            for date in ('%s', '[%s]', '[%s.555]', 'audit(%s.555:101)'):
                date = date % dateUnix
                log = date + ' [sshd] error: PAM: Authentication failure'
                datelog = self.datedetector.getTime(log)
                self.assertFalse(datelog)

    def testGetEpochPattern(self):
        if False:
            print('Hello World!')
        self.__datedetector = DateDetector()
        self.__datedetector.appendTemplate('(?<=\\|\\s){LEPOCH}(?=\\s\\|)')
        for fact in (1, 1000, 1000000):
            for dateUnix in (1138049999, 32535244799):
                dateLong = dateUnix * fact
                log = 'auth-error | %s | invalid password' % dateLong
                datelog = self.datedetector.getTime(log)
                self.assertTrue(datelog, 'Parse epoch time failed: %r' % (log,))
                (datelog, matchlog) = datelog
                self.assertEqual(int(datelog), dateUnix)
                self.assertEqual(matchlog.group(1), str(dateLong))
        for log in ('test%s123', 'test-right | %stest', 'test%s | test-left'):
            log = log % dateLong
            datelog = self.datedetector.getTime(log)
            self.assertFalse(datelog)

    def testGetEpochPatternCut(self):
        if False:
            print('Hello World!')
        self.__datedetector = DateDetector()
        self.__datedetector.appendTemplate('^type=\\S+ msg=audit\\(({EPOCH})')
        line = 'type=USER_AUTH msg=audit(1106513999.000:987)'
        datelog = self.datedetector.getTime(line)
        timeMatch = datelog[1]
        self.assertEqual([int(datelog[0]), line[timeMatch.start(1):timeMatch.end(1)]], [1106513999, '1106513999.000'])

    def testGetTime(self):
        if False:
            i = 10
            return i + 15
        log = 'Jan 23 21:59:59 [sshd] error: PAM: Authentication failure'
        dateUnix = 1106513999.0
        (datelog, matchlog) = self.datedetector.getTime(log)
        self.assertEqual(datelog, dateUnix)
        self.assertEqual(matchlog.group(1), 'Jan 23 21:59:59')

    def testDefaultTimeZone(self):
        if False:
            i = 10
            return i + 15
        dd = DateDetector()
        dd.appendTemplate('^%ExY-%Exm-%Exd %H:%M:%S(?: ?%Exz)?')
        dt = datetime.datetime
        logdt = '2017-01-23 15:00:00'
        dtUTC = dt(2017, 1, 23, 15, 0)
        for (tz, log, desired) in (('UTC+0300', logdt, dt(2017, 1, 23, 12, 0)), ('UTC', logdt, dtUTC), ('UTC-0430', logdt, dt(2017, 1, 23, 19, 30)), ('GMT+12', logdt, dt(2017, 1, 23, 3, 0)), (None, logdt, dt(2017, 1, 23, 14, 0)), ('CET', logdt, dt(2017, 1, 23, 14, 0)), ('+0100', logdt, dt(2017, 1, 23, 14, 0)), ('CEST-01', logdt, dt(2017, 1, 23, 14, 0)), ('CEST', logdt, dt(2017, 1, 23, 13, 0)), ('+0200', logdt, dt(2017, 1, 23, 13, 0)), ('CET+01', logdt, dt(2017, 1, 23, 13, 0)), ('CET+0100', logdt, dt(2017, 1, 23, 13, 0)), ('CET+0130', logdt, dt(2017, 1, 23, 12, 30)), ('UTC+0300', logdt + ' GMT', dtUTC), ('UTC', logdt + ' GMT', dtUTC), ('UTC-0430', logdt + ' GMT', dtUTC), (None, logdt + ' GMT', dtUTC), ('UTC', logdt + ' -1045', dt(2017, 1, 24, 1, 45)), (None, logdt + ' -10:45', dt(2017, 1, 24, 1, 45)), ('UTC', logdt + ' +0945', dt(2017, 1, 23, 5, 15)), (None, logdt + ' +09:45', dt(2017, 1, 23, 5, 15)), ('UTC+0300', logdt + ' Z', dtUTC), ('GMT+12', logdt + ' CET', dt(2017, 1, 23, 14, 0)), ('GMT+12', logdt + ' CEST', dt(2017, 1, 23, 13, 0)), ('GMT+12', logdt + ' CET+0130', dt(2017, 1, 23, 12, 30))):
            logSys.debug('== test %r with TZ %r', log, tz)
            dd.default_tz = tz
            (datelog, _) = dd.getTime(log)
            val = dt.utcfromtimestamp(datelog)
            self.assertEqual(val, desired, 'wrong offset %r != %r by %r with default TZ %r (%r)' % (val, desired, log, tz, dd.default_tz))
        self.assertRaises(ValueError, setattr, dd, 'default_tz', 'WRONG-TZ')
        dd.default_tz = None

    def testVariousTimes(self):
        if False:
            i = 10
            return i + 15
        'Test detection of various common date/time formats f2b should understand\n\t\t'
        dateUnix = 1106513999.0
        for (anchored, bound, sdate, rdate) in ((False, True, 'Jan 23 21:59:59', None), (False, False, 'Sun Jan 23 21:59:59 2005', None), (False, False, 'Sun Jan 23 21:59:59', None), (False, False, 'Sun Jan 23 2005 21:59:59', None), (False, True, '2005/01/23 21:59:59', None), (False, True, '2005.01.23 21:59:59', None), (False, True, '23/01/2005 21:59:59', None), (False, True, '23/01/05 21:59:59', None), (False, True, '23/Jan/2005:21:59:59', None), (False, True, '23/Jan/2005:21:59:59 +0100', None), (False, True, '01/23/2005:21:59:59', None), (False, True, '2005-01-23 21:59:59', None), (False, True, '2005-01-23 21:59:59,000', None), (False, True, '23-Jan-2005 21:59:59', None), (False, True, '23-Jan-2005 21:59:59.02', None), (False, True, '23-Jan-2005 21:59:59 +0100', None), (False, True, '23-01-2005 21:59:59', None), (True, True, '1106513999', None), (False, True, '01-23-2005 21:59:59.252', None), (False, False, '@4000000041f4104f00000000', None), (False, True, '2005-01-23T20:59:59.252Z', None), (False, True, '2005-01-23T15:59:59-05:00', None), (False, True, '2005-01-23 21:59:59', None), (False, True, '20050123T215959', None), (False, True, '20050123 215959', None), (True, True, '<01/23/05@21:59:59>', None), (False, True, '050123 21:59:59', None), (True, True, 'Jan-23-05 21:59:59', None), (False, True, 'Jan 23, 2005 9:59:59 PM', None), (True, True, '1106513999', None), (True, True, '1106513999.000', None), (True, True, '[1106513999.000]', '1106513999.000'), (False, True, 'audit(1106513999.000:987)', '1106513999.000'), (True, True, 'no date line', None)):
            if rdate is None and sdate != 'no date line':
                rdate = sdate
            logSys.debug('== test %r', (anchored, bound, sdate, rdate))
            for (should_match, prefix) in ((rdate is not None, ''), (not anchored, 'bogus-prefix '), (False, 'word-boundary')):
                log = prefix + sdate + '[sshd] error: PAM: Authentication failure'
                if not bound and prefix == 'word-boundary':
                    continue
                logSys.debug('  -- test %-5s for %r', should_match, log)
                logtime = self.datedetector.getTime(log)
                if should_match:
                    self.assertNotEqual(logtime, None, 'getTime retrieved nothing: failure for %s by prefix %r, anchored: %r, log: %s' % (sdate, prefix, anchored, log))
                    (logUnix, logMatch) = logtime
                    self.assertEqual(logUnix, dateUnix, 'getTime comparison failure for %s: by prefix %r "%s" is not "%s"' % (sdate, prefix, logUnix, dateUnix))
                    self.assertEqual(logMatch.group(1), rdate)
                else:
                    self.assertEqual(logtime, None, 'getTime should have not matched for %r by prefix %r Got: %s' % (sdate, prefix, logtime))
                (timeMatch, template) = matchTime = self.datedetector.matchTime(log)
                logtime = self.datedetector.getTime(log, matchTime)
                logSys.debug('  -- found - %r', template.name if timeMatch else False)
                if should_match:
                    self.assertNotEqual(logtime, None, 'getTime retrieved nothing: failure for %s by prefix %r, anchored: %r, log: %s' % (sdate, prefix, anchored, log))
                    (logUnix, logMatch) = logtime
                    self.assertEqual(logUnix, dateUnix, 'getTime comparison failure for %s by prefix %r: "%s" is not "%s"' % (sdate, prefix, logUnix, dateUnix))
                    self.assertEqual(logMatch.group(1), rdate)
                else:
                    self.assertEqual(logtime, None, 'getTime should have not matched for %r by prefix %r Got: %s' % (sdate, prefix, logtime))
                logSys.debug('  -- OK')

    def testAllUniqueTemplateNames(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(ValueError, self.datedetector.appendTemplate, self.datedetector.templates[0])

    def testFullYearMatch_gh130(self):
        if False:
            i = 10
            return i + 15
        mu = time.mktime(datetime.datetime(2012, 10, 11, 2, 37, 17).timetuple())
        logdate = self.datedetector.getTime('2012/10/11 02:37:17 [error] 18434#0')
        self.assertNotEqual(logdate, None)
        (logTime, logMatch) = logdate
        self.assertEqual(logTime, mu)
        self.assertEqual(logMatch.group(1), '2012/10/11 02:37:17')
        for i in range(10):
            (logTime, logMatch) = self.datedetector.getTime('11/10/2012 02:37:17 [error] 18434#0')
            self.assertEqual(logTime, mu)
            self.assertEqual(logMatch.group(1), '11/10/2012 02:37:17')
        (logTime, logMatch) = self.datedetector.getTime('2012/10/11 02:37:17 [error] 18434#0')
        self.assertEqual(logTime, mu)
        self.assertEqual(logMatch.group(1), '2012/10/11 02:37:17')

    def testDateTemplate(self):
        if False:
            while True:
                i = 10
        t = DateTemplate()
        t.setRegex('^a{3,5}b?c*$')
        self.assertEqual(t.regex, '^(a{3,5}b?c*)$')
        self.assertRaises(Exception, t.getDate, '')
        self.assertEqual(t.matchDate('aaaac').group(1), 'aaaac')
        t = DatePatternRegex()
        t.pattern = '(?iu)**time:%ExY%Exm%ExdT%ExH%ExM%ExS**'
        self.assertFalse('**' in t.regex)
        dt = 'TIME:20050102T010203'
        self.assertEqual(t.matchDate('X' + dt + 'X').group(1), dt)
        self.assertEqual(t.matchDate(dt).group(1), dt)
        dt = 'TIME:50050102T010203'
        self.assertFalse(t.matchDate(dt))
        t = DatePatternRegex()
        t.pattern = '{^LN-BEG}time:%ExY%Exm%ExdT%ExH%ExM%ExS'
        self.assertTrue('^' in t.regex)
        dt = 'time:20050102T010203'
        self.assertFalse(t.matchDate('X' + dt))
        self.assertFalse(t.matchDate(dt + 'X'))
        self.assertEqual(t.matchDate('##' + dt + '...').group(1), dt)
        self.assertEqual(t.matchDate(dt).group(1), dt)
        dt = 'TIME:20050102T010203'
        self.assertFalse(t.matchDate(dt))
        t = DatePatternRegex()
        t.pattern = '^%Y %b %d'
        self.assertTrue('(?iu)' in t.regex)
        dt = '2005 jun 03'
        self.assertEqual(t.matchDate(dt).group(1), dt)
        dt = '2005 Jun 03'
        self.assertEqual(t.matchDate(dt).group(1), dt)
        dt = '2005 JUN 03'
        self.assertEqual(t.matchDate(dt).group(1), dt)

    def testNotAnchoredCollision(self):
        if False:
            while True:
                i = 10
        for dp in ('%H:%M:%S', '{UNB}%H:%M:%S'):
            dd = DateDetector()
            dd.appendTemplate(dp)
            for fmt in ('%s test', '%8s test', 'test %s', 'test %8s'):
                for dt in ('00:01:02', '00:01:2', '00:1:2', '0:1:2', '00:1:2', '00:01:2', '00:01:02', '0:1:2', '00:01:02'):
                    t = dd.getTime(fmt % dt)
                    self.assertEqual((t[0], t[1].group()), (1123970462.0, dt))

    def testAmbiguousInOrderedTemplates(self):
        if False:
            print('Hello World!')
        dd = self.datedetector
        for (debit, line, cnt) in (('030324  0:03:59', 'some free text 030324  0:03:59 -- 2003-03-07 17:05:01 ...', 1), ('2003-03-07 17:05:01', 'some free text 2003-03-07 17:05:01 test ...', 15), ('030324  0:04:00', 'server mysqld[1000]: 030324  0:04:00 [Warning] Access denied ... foreign-input just some free text 2003-03-07 17:05:01 test', 10), ('Sep 16 21:30:26', 'server mysqld[1020]: Sep 16 21:30:26 server mysqld: 030916 21:30:26 [Warning] Access denied', 15), ('2005-10-07 06:09:42', 'server mysqld[5906]: 2005-10-07 06:09:42 5907 [Warning] Access denied', 20), ('2005-10-08T15:26:18.237955', 'server mysqld[5906]: 2005-10-08T15:26:18.237955 6 [Note] Access denied', 20), ('051009 10:05:30', 'server mysqld[1000]: 051009 10:05:30 [Warning] Access denied ...', 50)):
            logSys.debug('== test: %r', (debit, line, cnt))
            for i in range(cnt):
                logSys.debug('Line: %s', line)
                (match, template) = dd.matchTime(line)
                self.assertTrue(match)
                self.assertEqual(match.group(1), debit)

    def testLowLevelLogging(self):
        if False:
            i = 10
            return i + 15
        try:
            self.__old_eff_level = datedetector.logLevel
            if datedetector.logLevel < logSys.getEffectiveLevel() + 1:
                datedetector.logLevel = logSys.getEffectiveLevel() + 1
            dd = self.datedetector
            i = 0
            for (line, cnt) in (('server mysqld[5906]: 2005-10-07 06:09:%02i 5907 [Warning] Access denied', 2), ('server mysqld[5906]: 051007 06:10:%02i 5907 [Warning] Access denied', 5), ('server mysqld[5906]: 2005-10-07 06:09:%02i 5907 [Warning] Access denied', 10)):
                for i in range(i, i + cnt + 1):
                    logSys.debug('== test: %r', (line % i, cnt))
                    (match, template) = dd.matchTime(line % i)
                    self.assertTrue(match)
        finally:
            datedetector.logLevel = self.__old_eff_level

    def testWrongTemplate(self):
        if False:
            print('Hello World!')
        t = DatePatternRegex('(%ExY%Exm%Exd')
        self.assertRaises(Exception, t.matchDate, '(20050101')
        self.assertLogged('Compile %r failed' % t.name)
        t = DateTemplate()
        self.assertRaises(Exception, t.getDate, 'no date line')
iso8601 = DatePatternRegex('%Y-%m-%d[T ]%H:%M:%S(?:\\.%f)?%z')

class CustomDateFormatsTest(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        'Call before every test case.'
        unittest.TestCase.setUp(self)
        setUpMyTime()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        'Call after every test case.'
        unittest.TestCase.tearDown(self)
        tearDownMyTime()

    def testIso8601(self):
        if False:
            i = 10
            return i + 15
        date = datetime.datetime.utcfromtimestamp(iso8601.getDate('2007-01-25T12:00:00Z')[0])
        self.assertEqual(date, datetime.datetime(2007, 1, 25, 12, 0))
        self.assertRaises(TypeError, iso8601.getDate, None)
        self.assertRaises(TypeError, iso8601.getDate, date)
        self.assertEqual(iso8601.getDate(''), None)
        self.assertEqual(iso8601.getDate('Z'), None)
        self.assertEqual(iso8601.getDate('2007-01-01T120:00:00Z'), None)
        self.assertEqual(iso8601.getDate('2007-13-01T12:00:00Z'), None)
        date = datetime.datetime.utcfromtimestamp(iso8601.getDate('2007-01-25T12:00:00+0400')[0])
        self.assertEqual(date, datetime.datetime(2007, 1, 25, 8, 0))
        date = datetime.datetime.utcfromtimestamp(iso8601.getDate('2007-01-25T12:00:00+04:00')[0])
        self.assertEqual(date, datetime.datetime(2007, 1, 25, 8, 0))
        date = datetime.datetime.utcfromtimestamp(iso8601.getDate('2007-01-25T12:00:00-0400')[0])
        self.assertEqual(date, datetime.datetime(2007, 1, 25, 16, 0))
        date = datetime.datetime.utcfromtimestamp(iso8601.getDate('2007-01-25T12:00:00-04')[0])
        self.assertEqual(date, datetime.datetime(2007, 1, 25, 16, 0))

    def testAmbiguousDatePattern(self):
        if False:
            return 10
        defDD = DateDetector()
        defDD.addDefaultTemplate()
        for (matched, dp, line) in (('Jan 23 21:59:59', None, 'Test failure Jan 23 21:59:59 for 192.0.2.1'), (False, None, 'Test failure TestJan 23 21:59:59.011 2015 for 192.0.2.1'), (False, None, 'Test failure Jan 23 21:59:59123456789 for 192.0.2.1'), ('Aug 8 11:25:50', None, 'Aug 8 11:25:50 20030f2329b8 Authentication failed from 192.0.2.1'), ('Aug 8 11:25:50', None, '[Aug 8 11:25:50] 20030f2329b8 Authentication failed from 192.0.2.1'), ('Aug 8 11:25:50 2014', None, 'Aug 8 11:25:50 2014 20030f2329b8 Authentication failed from 192.0.2.1'), ('20:00:00 01.02.2003', '%H:%M:%S %d.%m.%Y$', '192.0.2.1 at 20:00:00 01.02.2003'), ('[20:00:00 01.02.2003]', '\\[%H:%M:%S %d.%m.%Y\\]', '192.0.2.1[20:00:00 01.02.2003]'), ('[20:00:00 01.02.2003]', '\\[%H:%M:%S %d.%m.%Y\\]', '[20:00:00 01.02.2003]192.0.2.1'), ('[20:00:00 01.02.2003]', '\\[%H:%M:%S %d.%m.%Y\\]$', '192.0.2.1[20:00:00 01.02.2003]'), ('[20:00:00 01.02.2003]', '^\\[%H:%M:%S %d.%m.%Y\\]', '[20:00:00 01.02.2003]192.0.2.1'), ('[17/Jun/2011 17:00:45]', '^\\[%d/%b/%Y %H:%M:%S\\]', '[17/Jun/2011 17:00:45] Attempt, IP address 192.0.2.1'), ('[17/Jun/2011 17:00:45]', '\\[%d/%b/%Y %H:%M:%S\\]', 'Attempt [17/Jun/2011 17:00:45] IP address 192.0.2.1'), ('[17/Jun/2011 17:00:45]', '\\[%d/%b/%Y %H:%M:%S\\]', 'Attempt IP address 192.0.2.1, date: [17/Jun/2011 17:00:45]'), (False, '%H:%M:%S %d.%m.%Y', '192.0.2.1x20:00:00 01.02.2003'), (False, '%H:%M:%S %d.%m.%Y', '20:00:00 01.02.2003x192.0.2.1'), ('20:00:00 01.02.2003', '**%H:%M:%S %d.%m.%Y**', '192.0.2.1x20:00:00 01.02.2003'), ('20:00:00 01.02.2003', '**%H:%M:%S %d.%m.%Y**', '20:00:00 01.02.2003x192.0.2.1'), ('*20:00:00 01.02.2003*', '\\**%H:%M:%S %d.%m.%Y\\**', 'test*20:00:00 01.02.2003*test'), ('20:00:00 01.02.2003', '%H:%M:%S %d.%m.%Y', '192.0.2.1 20:00:00 01.02.2003'), ('20:00:00 01.02.2003', '%H:%M:%S %d.%m.%Y', '20:00:00 01.02.2003 192.0.2.1'), (None, '%Y-%Exm-%Exd %ExH:%ExM:%ExS', '0000-12-30 00:00:00 - 2003-12-30 00:00:00'), ('2003-12-30 00:00:00', '%ExY-%Exm-%Exd %ExH:%ExM:%ExS', '0000-12-30 00:00:00 - 2003-12-30 00:00:00'), ('2003-12-30 00:00:00', None, '0000-12-30 00:00:00 - 2003-12-30 00:00:00'), ('200333 010203', '%Y%m%d %H%M%S', 'text:200333 010203 | date:20031230 010203'), ('20031230 010203', '%ExY%Exm%Exd %ExH%ExM%ExS', 'text:200333 010203 | date:20031230 010203'), ('20031230 010203', None, 'text:200333 010203 | date:20031230 010203'), ('20030101 000000', '%ExY%Exm%Exd %ExH%ExM%ExS', '00001230 010203 - 20030101 000000'), (None, '{^LN-BEG}%ExY%Exm%Exd %ExH%ExM%ExS', '00001230 010203 - 20030101 000000'), ('20031230 010203', '{^LN-BEG}%ExY%Exm%Exd %ExH%ExM%ExS', '20031230 010203 - 20030101 000000'), ('20031230010203', '{^LN-BEG}%ExY%Exm%Exd%ExH%ExM%ExS**', '2003123001020320030101000000'), ('20031230010203', '{^LN-BEG}%ExY%Exm%Exd%ExH%ExM%ExS**', '#2003123001020320030101000000'), ('20031230010203', '{^LN-BEG}%ExY%Exm%Exd%ExH%ExM%ExS**', '##2003123001020320030101000000'), ('20031230010203', '{^LN-BEG}%ExY%Exm%Exd%ExH%ExM%ExS', '[20031230010203]20030101000000'), (1072746123.0 - 3600, '{^LN-BEG}%ExY-%Exm-%Exd %ExH:%ExM:%ExS(?: %z)?', '[2003-12-30 01:02:03] server ...'), (1072746123.0 - 3600, '{^LN-BEG}%ExY-%Exm-%Exd %ExH:%ExM:%ExS(?: %Z)?', '[2003-12-30 01:02:03] server ...'), (1072746123.0, '{^LN-BEG}%ExY-%Exm-%Exd %ExH:%ExM:%ExS(?: %z)?', '[2003-12-30 01:02:03 UTC] server ...'), (1072746123.0, '{^LN-BEG}%ExY-%Exm-%Exd %ExH:%ExM:%ExS(?: %Z)?', '[2003-12-30 01:02:03 UTC] server ...'), (1072746123.0, '{^LN-BEG}%ExY-%Exm-%Exd %ExH:%ExM:%ExS(?: %z)?', '[2003-12-30 01:02:03 Z] server ...'), (1072746123.0, '{^LN-BEG}%ExY-%Exm-%Exd %ExH:%ExM:%ExS(?: %z)?', '[2003-12-30 01:02:03 +0000] server ...'), (1072746123.0, '{^LN-BEG}%ExY-%Exm-%Exd %ExH:%ExM:%ExS(?: %Z)?', '[2003-12-30 01:02:03 Z] server ...')):
            logSys.debug('== test: %r', (matched, dp, line))
            if dp is None:
                dd = defDD
            else:
                dd = DateDetector()
                dd.appendTemplate(dp)
            date = dd.getTime(line)
            if matched:
                self.assertTrue(date)
                if isinstance(matched, str):
                    self.assertEqual(matched, date[1].group(1))
                else:
                    self.assertEqual(matched, date[0])
            else:
                self.assertEqual(date, None)

    def testVariousFormatSpecs(self):
        if False:
            print('Hello World!')
        for (matched, dp, line) in ((1106438399.0, '^%B %Exd %I:%ExM:%ExS**', 'January 23 12:59:59'), (985208399.0, '^%y %U %A %ExH:%ExM:%ExS**', '01 11 Wednesday 21:59:59'), (984603599.0, '^%y %W %A %ExH:%ExM:%ExS**', '01 11 Wednesday 21:59:59'), (984949199.0, '^%y %W %w %ExH:%ExM:%ExS**', '01 11 0 21:59:59'), (984862799.0, '^%y %W %w %ExH:%ExM:%ExS**', '01 11 6 21:59:59'), (1123963199.0, '^%ExH:%ExM:%ExS**', '21:59:59'), (1123970401.0, '^%ExH:%ExM:%ExS**', '00:00:01'), (1094068799.0, '^%m/%d %ExH:%ExM:%ExS**', '09/01 21:59:59'), (1093989600.0, '^%Y-%m-%d**', '2004-09-01'), (1093996800.0, '^%Y-%m-%d%z**', '2004-09-01Z')):
            logSys.debug('== test: %r', (matched, dp, line))
            dd = DateDetector()
            dd.appendTemplate(dp)
            date = dd.getTime(line)
            if matched:
                self.assertTrue(date)
                if isinstance(matched, str):
                    self.assertEqual(matched, date[1].group(1))
                else:
                    self.assertEqual(matched, date[0])
            else:
                self.assertEqual(date, None)