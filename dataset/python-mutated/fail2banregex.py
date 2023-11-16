"""
Fail2Ban  reads log file that contains password failure report
and bans the corresponding IP addresses using firewall rules.

This tools can test regular expressions for "fail2ban".
"""
__author__ = 'Fail2Ban Developers'
__copyright__ = 'Copyright (c) 2004-2008 Cyril Jaquier, 2008- Fail2Ban Contributors\nCopyright of modifications held by their respective authors.\nLicensed under the GNU General Public License v2 (GPL).\n\nWritten by Cyril Jaquier <cyril.jaquier@fail2ban.org>.\nMany contributions by Yaroslav O. Halchenko, Steven Hiscocks, Sergey G. Brester (sebres).'
__license__ = 'GPL'
import getopt
import logging
import re
import os
import shlex
import sys
import time
import urllib.request, urllib.parse, urllib.error
from optparse import OptionParser, Option
from configparser import NoOptionError, NoSectionError, MissingSectionHeaderError
try:
    from ..server.filtersystemd import FilterSystemd
except ImportError:
    FilterSystemd = None
from ..version import version, normVersion
from .filterreader import FilterReader
from ..server.filter import Filter, FileContainer, MyTime
from ..server.failregex import Regex, RegexException
from ..helpers import str2LogLevel, getVerbosityFormat, FormatterWithTraceBack, getLogger, extractOptions, PREFER_ENC
logSys = getLogger('fail2ban')

def debuggexURL(sample, regex, multiline=False, useDns='yes'):
    if False:
        i = 10
        return i + 15
    args = {'re': Regex._resolveHostTag(regex, useDns=useDns), 'str': sample, 'flavor': 'python'}
    if multiline:
        args['flags'] = 'm'
    return 'https://www.debuggex.com/?' + urllib.parse.urlencode(args)

def output(args):
    if False:
        for i in range(10):
            print('nop')
    print(args)

def shortstr(s, l=53):
    if False:
        return 10
    'Return shortened string\n\t'
    if len(s) > l:
        return s[:l - 3] + '...'
    return s

def pprint_list(l, header=None):
    if False:
        while True:
            i = 10
    if not len(l):
        return
    if header:
        s = '|- %s\n' % header
    else:
        s = ''
    output(s + '|  ' + '\n|  '.join(l) + '\n`-')

def journal_lines_gen(flt, myjournal):
    if False:
        return 10
    while True:
        try:
            entry = myjournal.get_next()
        except OSError:
            continue
        if not entry:
            break
        yield flt.formatJournalEntry(entry)

def dumpNormVersion(*args):
    if False:
        while True:
            i = 10
    output(normVersion())
    sys.exit(0)
usage = lambda : '%s [OPTIONS] <LOG> <REGEX> [IGNOREREGEX]' % sys.argv[0]

class _f2bOptParser(OptionParser):

    def format_help(self, *args, **kwargs):
        if False:
            print('Hello World!')
        ' Overwritten format helper with full ussage.'
        self.usage = ''
        return 'Usage: ' + usage() + '\n' + __doc__ + "\nLOG:\n  string                a string representing a log line\n  filename              path to a log file (/var/log/auth.log)\n  systemd-journal       search systemd journal (systemd-python required),\n                        optionally with backend parameters, see `man jail.conf`\n                        for usage and examples (systemd-journal[journalflags=1]).\n\nREGEX:\n  string                a string representing a 'failregex'\n  filter                name of filter, optionally with options (sshd[mode=aggressive])\n  filename              path to a filter file (filter.d/sshd.conf)\n\nIGNOREREGEX:\n  string                a string representing an 'ignoreregex'\n  filename              path to a filter file (filter.d/sshd.conf)\n\n" + OptionParser.format_help(self, *args, **kwargs) + '\n\nReport bugs to https://github.com/fail2ban/fail2ban/issues\n\n' + __copyright__ + '\n'

def get_opt_parser():
    if False:
        for i in range(10):
            print('nop')
    p = _f2bOptParser(usage=usage(), version='%prog ' + version)
    p.add_options([Option('-c', '--config', default='/etc/fail2ban', help='set alternate config directory'), Option('-d', '--datepattern', help='set custom pattern used to match date/times'), Option('--timezone', '--TZ', action='store', default=None, help='set time-zone used by convert time format'), Option('-e', '--encoding', default=PREFER_ENC, help='File encoding. Default: system locale'), Option('-r', '--raw', action='store_true', default=False, help="Raw hosts, don't resolve dns"), Option('--usedns', action='store', default=None, help="DNS specified replacement of tags <HOST> in regexp ('yes' - matches all form of hosts, 'no' - IP addresses only)"), Option('-L', '--maxlines', type=int, default=0, help='maxlines for multi-line regex.'), Option('-m', '--journalmatch', help='journalctl style matches overriding filter file. "systemd-journal" only'), Option('-l', '--log-level', dest='log_level', default='critical', help='Log level for the Fail2Ban logger to use'), Option('-V', action='callback', callback=dumpNormVersion, help='get version in machine-readable short format'), Option('-v', '--verbose', action='count', dest='verbose', default=0, help='Increase verbosity'), Option('--verbosity', action='store', dest='verbose', type=int, help='Set numerical level of verbosity (0..4)'), Option('--verbose-date', '--VD', action='store_true', help='Verbose date patterns/regex in output'), Option('-D', '--debuggex', action='store_true', help='Produce debuggex.com urls for debugging there'), Option('--no-check-all', action='store_false', dest='checkAllRegex', default=True, help="Disable check for all regex's"), Option('-o', '--out', action='store', dest='out', default=None, help='Set token to print failure information only (row, id, ip, msg, host, ip4, ip6, dns, matches, ...)'), Option('--print-no-missed', action='store_true', help='Do not print any missed lines'), Option('--print-no-ignored', action='store_true', help='Do not print any ignored lines'), Option('--print-all-matched', action='store_true', help='Print all matched lines'), Option('--print-all-missed', action='store_true', help='Print all missed lines, no matter how many'), Option('--print-all-ignored', action='store_true', help='Print all ignored lines, no matter how many'), Option('-t', '--log-traceback', action='store_true', help='Enrich log-messages with compressed tracebacks'), Option('--full-traceback', action='store_true', help='Either to make the tracebacks full, not compressed (as by default)')])
    return p

class RegexStat(object):

    def __init__(self, failregex):
        if False:
            i = 10
            return i + 15
        self._stats = 0
        self._failregex = failregex
        self._ipList = list()

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return '%s(%r) %d failed: %s' % (self.__class__, self._failregex, self._stats, self._ipList)

    def inc(self):
        if False:
            while True:
                i = 10
        self._stats += 1

    def getStats(self):
        if False:
            for i in range(10):
                print('nop')
        return self._stats

    def getFailRegex(self):
        if False:
            print('Hello World!')
        return self._failregex

    def appendIP(self, value):
        if False:
            for i in range(10):
                print('nop')
        self._ipList.append(value)

    def getIPList(self):
        if False:
            i = 10
            return i + 15
        return self._ipList

class LineStats(object):
    """Just a convenience container for stats
	"""

    def __init__(self, opts):
        if False:
            i = 10
            return i + 15
        self.tested = self.matched = 0
        self.matched_lines = []
        self.missed = 0
        self.missed_lines = []
        self.ignored = 0
        self.ignored_lines = []
        if opts.debuggex:
            self.matched_lines_timeextracted = []
            self.missed_lines_timeextracted = []
            self.ignored_lines_timeextracted = []

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return '%(tested)d lines, %(ignored)d ignored, %(matched)d matched, %(missed)d missed' % self

    def __getitem__(self, key):
        if False:
            i = 10
            return i + 15
        return getattr(self, key) if hasattr(self, key) else ''

class Fail2banRegex(object):

    def __init__(self, opts):
        if False:
            i = 10
            return i + 15
        self.__dict__.update(dict((('_' + o, v) for (o, v) in opts.__dict__.items())))
        self._opts = opts
        self._maxlines_set = False
        self._datepattern_set = False
        self._journalmatch = None
        self.share_config = dict()
        self._filter = Filter(None)
        self._prefREMatched = 0
        self._prefREGroups = list()
        self._ignoreregex = list()
        self._failregex = list()
        self._time_elapsed = None
        self._line_stats = LineStats(opts)
        if opts.maxlines:
            self.setMaxLines(opts.maxlines)
        else:
            self._maxlines = 20
        if opts.journalmatch is not None:
            self.setJournalMatch(shlex.split(opts.journalmatch))
        if opts.timezone:
            self._filter.setLogTimeZone(opts.timezone)
        self._filter.checkFindTime = False
        if True:
            MyTime.setAlternateNow(0)
            from ..server.strptime import _updateTimeRE
            _updateTimeRE()
        if opts.datepattern:
            self.setDatePattern(opts.datepattern)
        if opts.usedns:
            self._filter.setUseDns(opts.usedns)
        self._filter.returnRawHost = opts.raw
        self._filter.checkAllRegex = opts.checkAllRegex and (not opts.out)
        self._filter.ignorePending = bool(opts.out)
        self._filter.onIgnoreRegex = self._onIgnoreRegex
        self._backend = 'auto'

    def output(self, line):
        if False:
            for i in range(10):
                print('nop')
        if not self._opts.out:
            output(line)

    def encode_line(self, line):
        if False:
            return 10
        return line.encode(self._encoding, 'ignore')

    def setDatePattern(self, pattern):
        if False:
            for i in range(10):
                print('nop')
        if not self._datepattern_set:
            self._filter.setDatePattern(pattern)
            self._datepattern_set = True
            if pattern is not None:
                self.output('Use      datepattern : %s : %s' % (pattern, self._filter.getDatePattern()[1]))

    def setMaxLines(self, v):
        if False:
            while True:
                i = 10
        if not self._maxlines_set:
            self._filter.setMaxLines(int(v))
            self._maxlines_set = True
            self.output('Use         maxlines : %d' % self._filter.getMaxLines())

    def setJournalMatch(self, v):
        if False:
            i = 10
            return i + 15
        self._journalmatch = v

    def _dumpRealOptions(self, reader, fltOpt):
        if False:
            print('Hello World!')
        realopts = {}
        combopts = reader.getCombined()
        for k in ['logtype', 'datepattern'] + list(fltOpt.keys()):
            try:
                realopts[k] = combopts[k] if k in combopts else reader.get('Definition', k)
            except NoOptionError:
                pass
        self.output('Real  filter options : %r' % realopts)

    def readRegex(self, value, regextype):
        if False:
            return 10
        assert regextype in ('fail', 'ignore')
        regex = regextype + 'regex'
        basedir = self._opts.config
        fltName = value
        fltFile = None
        fltOpt = {}
        if regextype == 'fail':
            if re.search('(?ms)^/{0,3}[\\w/_\\-.]+(?:\\[.*\\])?$', value):
                try:
                    (fltName, fltOpt) = extractOptions(value)
                    if '.' in fltName[~5:]:
                        tryNames = (fltName,)
                    else:
                        tryNames = (fltName, fltName + '.conf', fltName + '.local')
                    for fltFile in tryNames:
                        if not '/' in fltFile:
                            if os.path.basename(basedir) == 'filter.d':
                                fltFile = os.path.join(basedir, fltFile)
                            else:
                                fltFile = os.path.join(basedir, 'filter.d', fltFile)
                        else:
                            basedir = os.path.dirname(fltFile)
                        if os.path.isfile(fltFile):
                            break
                        fltFile = None
                except Exception as e:
                    output('ERROR: Wrong filter name or options: %s' % (str(e),))
                    output('       while parsing: %s' % (value,))
                    if self._verbose:
                        raise e
                    return False
        if fltFile is not None:
            if basedir == self._opts.config or os.path.basename(basedir) == 'filter.d' or ('.' not in fltName[~5:] and '/' not in fltName):
                if os.path.basename(basedir) == 'filter.d':
                    basedir = os.path.dirname(basedir)
                fltName = os.path.splitext(os.path.basename(fltName))[0]
                self.output('Use %11s filter file : %s, basedir: %s' % (regex, fltName, basedir))
            else:
                self.output('Use %11s file : %s' % (regex, fltName))
                basedir = None
                if not os.path.isabs(fltName):
                    fltName = os.path.abspath(fltName)
            if fltOpt:
                self.output('Use   filter options : %r' % fltOpt)
            reader = FilterReader(fltName, 'fail2ban-regex-jail', fltOpt, share_config=self.share_config, basedir=basedir)
            ret = None
            try:
                if basedir is not None:
                    ret = reader.read()
                else:
                    reader.setBaseDir(None)
                    ret = reader.readexplicit()
            except Exception as e:
                output('Wrong config file: %s' % (str(e),))
                if self._verbose:
                    raise e
            if not ret:
                output('ERROR: failed to load filter %s' % value)
                return False
            reader.applyAutoOptions(self._backend)
            reader.getOptions(None)
            if self._verbose > 1 or logSys.getEffectiveLevel() <= logging.DEBUG:
                self._dumpRealOptions(reader, fltOpt)
            readercommands = reader.convert()
            regex_values = {}
            for opt in readercommands:
                if opt[0] == 'multi-set':
                    optval = opt[3]
                elif opt[0] == 'set':
                    optval = opt[3:]
                else:
                    continue
                try:
                    if opt[2] == 'prefregex':
                        for optval in optval:
                            self._filter.prefRegex = optval
                    elif opt[2] == 'addfailregex':
                        stor = regex_values.get('fail')
                        if not stor:
                            stor = regex_values['fail'] = list()
                        for optval in optval:
                            stor.append(RegexStat(optval))
                    elif opt[2] == 'addignoreregex':
                        stor = regex_values.get('ignore')
                        if not stor:
                            stor = regex_values['ignore'] = list()
                        for optval in optval:
                            stor.append(RegexStat(optval))
                    elif opt[2] == 'maxlines':
                        for optval in optval:
                            self.setMaxLines(optval)
                    elif opt[2] == 'datepattern':
                        for optval in optval:
                            self.setDatePattern(optval)
                    elif opt[2] == 'addjournalmatch':
                        if self._opts.journalmatch is None:
                            self.setJournalMatch(optval)
                except ValueError as e:
                    output('ERROR: Invalid value for %s (%r) read from %s: %s' % (opt[2], optval, value, e))
                    return False
        else:
            self.output('Use %11s line : %s' % (regex, shortstr(value)))
            regex_values = {regextype: [RegexStat(value)]}
        for (regextype, regex_values) in regex_values.items():
            regex = regextype + 'regex'
            setattr(self, '_' + regex, regex_values)
            for regex in regex_values:
                getattr(self._filter, 'add%sRegex' % regextype.title())(regex.getFailRegex())
        return True

    def _onIgnoreRegex(self, idx, ignoreRegex):
        if False:
            print('Hello World!')
        self._lineIgnored = True
        self._ignoreregex[idx].inc()

    def testRegex(self, line, date=None):
        if False:
            for i in range(10):
                print('nop')
        orgLineBuffer = self._filter._Filter__lineBuffer
        if self._filter.getMaxLines() > 1:
            orgLineBuffer = orgLineBuffer[:]
        fullBuffer = len(orgLineBuffer) >= self._filter.getMaxLines()
        is_ignored = self._lineIgnored = False
        try:
            found = self._filter.processLine(line, date)
            lines = []
            ret = []
            for match in found:
                if not self._opts.out:
                    match.append(len(ret) > 1)
                    regex = self._failregex[match[0]]
                    regex.inc()
                    regex.appendIP(match)
                if not match[3].get('nofail'):
                    ret.append(match)
                else:
                    is_ignored = True
            if self._opts.out:
                return (None, ret, None)
            if self._filter.prefRegex:
                pre = self._filter.prefRegex
                if pre.hasMatched():
                    self._prefREMatched += 1
                    if self._verbose:
                        if len(self._prefREGroups) < self._maxlines:
                            self._prefREGroups.append(pre.getGroups())
                        elif len(self._prefREGroups) == self._maxlines:
                            self._prefREGroups.append('...')
        except RegexException as e:
            output('ERROR: %s' % e)
            return (None, 0, None)
        if self._filter.getMaxLines() > 1:
            for bufLine in orgLineBuffer[int(fullBuffer):]:
                if bufLine not in self._filter._Filter__lineBuffer:
                    try:
                        self._line_stats.missed_lines.pop(self._line_stats.missed_lines.index(''.join(bufLine)))
                        if self._debuggex:
                            self._line_stats.missed_lines_timeextracted.pop(self._line_stats.missed_lines_timeextracted.index(''.join(bufLine[::2])))
                    except ValueError:
                        pass
                    if self._print_all_matched:
                        if not self._debuggex:
                            self._line_stats.matched_lines.append(''.join(bufLine))
                        else:
                            lines.append(bufLine[0] + bufLine[2])
                    self._line_stats.matched += 1
                    self._line_stats.missed -= 1
        if lines:
            lines.append(self._filter.processedLine())
            line = '\n'.join(lines)
        return (line, ret, is_ignored or self._lineIgnored)

    def _prepaireOutput(self):
        if False:
            while True:
                i = 10
        "Prepares output- and fetch-function corresponding given '--out' option (format)"
        ofmt = self._opts.out
        if ofmt in ('id', 'fid'):

            def _out(ret):
                if False:
                    return 10
                for r in ret:
                    output(r[1])
        elif ofmt == 'ip':

            def _out(ret):
                if False:
                    i = 10
                    return i + 15
                for r in ret:
                    output(r[3].get('ip', r[1]))
        elif ofmt == 'msg':

            def _out(ret):
                if False:
                    i = 10
                    return i + 15
                for r in ret:
                    for r in r[3].get('matches'):
                        if not isinstance(r, str):
                            r = ''.join((r for r in r))
                        output(r)
        elif ofmt == 'row':

            def _out(ret):
                if False:
                    i = 10
                    return i + 15
                for r in ret:
                    output('[%r,\t%r,\t%r],' % (r[1], r[2], dict(((k, v) for (k, v) in r[3].items() if k != 'matches'))))
        elif '<' not in ofmt:

            def _out(ret):
                if False:
                    for i in range(10):
                        print('nop')
                for r in ret:
                    output(r[3].get(ofmt))
        else:
            from ..server.actions import Actions, CommandAction, BanTicket

            def _escOut(t, v):
                if False:
                    for i in range(10):
                        print('nop')
                if t not in ('msg',):
                    return v.replace('\x00', '\\x00')
                return v

            def _out(ret):
                if False:
                    while True:
                        i = 10
                rows = []
                wrap = {'NL': 0}
                for r in ret:
                    ticket = BanTicket(r[1], time=r[2], data=r[3])
                    aInfo = Actions.ActionInfo(ticket)

                    def _get_msg(self):
                        if False:
                            print('Hello World!')
                        if not wrap['NL'] and len(r[3].get('matches', [])) <= 1:
                            return self['matches']
                        else:
                            wrap['NL'] = 1
                            return '\x00msg\x00'
                    aInfo['msg'] = _get_msg
                    v = CommandAction.replaceDynamicTags(ofmt, aInfo, escapeVal=_escOut)
                    if wrap['NL']:
                        rows.append((r, v))
                        continue
                    output(v)
                for (r, v) in rows:
                    for r in r[3].get('matches'):
                        if not isinstance(r, str):
                            r = ''.join((r for r in r))
                        r = v.replace('\x00msg\x00', r)
                        output(r)
        return _out

    def process(self, test_lines):
        if False:
            return 10
        t0 = time.time()
        if self._opts.out:
            out = self._prepaireOutput()
        for line in test_lines:
            if isinstance(line, tuple):
                (line_datetimestripped, ret, is_ignored) = self.testRegex(line[0], line[1])
                line = ''.join(line[0])
            else:
                line = line.rstrip('\r\n')
                if line.startswith('#') or not line:
                    continue
                (line_datetimestripped, ret, is_ignored) = self.testRegex(line)
            if self._opts.out:
                if len(ret) > 0 and (not is_ignored):
                    out(ret)
                continue
            if is_ignored:
                self._line_stats.ignored += 1
                if not self._print_no_ignored and (self._print_all_ignored or self._line_stats.ignored <= self._maxlines + 1):
                    self._line_stats.ignored_lines.append(line)
                    if self._debuggex:
                        self._line_stats.ignored_lines_timeextracted.append(line_datetimestripped)
            elif len(ret) > 0:
                self._line_stats.matched += 1
                if self._print_all_matched:
                    self._line_stats.matched_lines.append(line)
                    if self._debuggex:
                        self._line_stats.matched_lines_timeextracted.append(line_datetimestripped)
            else:
                self._line_stats.missed += 1
                if not self._print_no_missed and (self._print_all_missed or self._line_stats.missed <= self._maxlines + 1):
                    self._line_stats.missed_lines.append(line)
                    if self._debuggex:
                        self._line_stats.missed_lines_timeextracted.append(line_datetimestripped)
            self._line_stats.tested += 1
        self._time_elapsed = time.time() - t0

    def printLines(self, ltype):
        if False:
            return 10
        lstats = self._line_stats
        assert lstats.missed == lstats.tested - (lstats.matched + lstats.ignored)
        lines = lstats[ltype]
        l = lstats[ltype + '_lines']
        multiline = self._filter.getMaxLines() > 1
        if lines:
            header = '%s line(s):' % (ltype.capitalize(),)
            if self._debuggex:
                if ltype == 'missed' or ltype == 'matched':
                    regexlist = self._failregex
                else:
                    regexlist = self._ignoreregex
                l = lstats[ltype + '_lines_timeextracted']
                if lines < self._maxlines or getattr(self, '_print_all_' + ltype):
                    ans = [[]]
                    for arg in [l, regexlist]:
                        ans = [x + [y] for x in ans for y in arg]
                    b = [a[0] + ' | ' + a[1].getFailRegex() + ' |  ' + debuggexURL(self.encode_line(a[0]), a[1].getFailRegex(), multiline, self._opts.usedns) for a in ans]
                    pprint_list([x.rstrip() for x in b], header)
                else:
                    output('%s too many to print.  Use --print-all-%s to print all %d lines' % (header, ltype, lines))
            elif lines < self._maxlines or getattr(self, '_print_all_' + ltype):
                pprint_list([x.rstrip() for x in l], header)
            else:
                output('%s too many to print.  Use --print-all-%s to print all %d lines' % (header, ltype, lines))

    def printStats(self):
        if False:
            return 10
        if self._opts.out:
            return True
        output('')
        output('Results')
        output('=======')

        def print_failregexes(title, failregexes):
            if False:
                while True:
                    i = 10
            (total, out) = (0, [])
            for (cnt, failregex) in enumerate(failregexes):
                match = failregex.getStats()
                total += match
                if match or self._verbose:
                    out.append('%2d) [%d] %s' % (cnt + 1, match, failregex.getFailRegex()))
                if self._verbose and len(failregex.getIPList()):
                    for ip in failregex.getIPList():
                        timeTuple = time.localtime(ip[2])
                        timeString = time.strftime('%a %b %d %H:%M:%S %Y', timeTuple)
                        out.append('    %s  %s%s' % (ip[1], timeString, ip[-1] and ' (multiple regex matched)' or ''))
            output('\n%s: %d total' % (title, total))
            pprint_list(out, ' #) [# of hits] regular expression')
            return total
        if self._filter.prefRegex:
            pre = self._filter.prefRegex
            out = [pre.getRegex()]
            if self._verbose:
                for grp in self._prefREGroups:
                    out.append('    %s' % (grp,))
            output('\n%s: %d total' % ('Prefregex', self._prefREMatched))
            pprint_list(out)
        total = print_failregexes('Failregex', self._failregex)
        _ = print_failregexes('Ignoreregex', self._ignoreregex)
        if self._filter.dateDetector is not None:
            output('\nDate template hits:')
            out = []
            for template in self._filter.dateDetector.templates:
                if self._verbose or template.hits:
                    out.append('[%d] %s' % (template.hits, template.name))
                    if self._verbose_date:
                        out.append('    # weight: %.3f (%.3f), pattern: %s' % (template.weight, template.template.weight, getattr(template, 'pattern', '')))
                        out.append('    # regex:   %s' % (getattr(template, 'regex', ''),))
            pprint_list(out, '[# of hits] date format')
        output('\nLines: %s' % self._line_stats)
        if self._time_elapsed is not None:
            output('[processed in %.2f sec]' % self._time_elapsed)
        output('')
        if self._print_all_matched:
            self.printLines('matched')
        if not self._print_no_ignored:
            self.printLines('ignored')
        if not self._print_no_missed:
            self.printLines('missed')
        return True

    def start(self, args):
        if False:
            print('Hello World!')
        (cmd_log, cmd_regex) = args[:2]
        if cmd_log.startswith('systemd-journal'):
            self._backend = 'systemd'
        try:
            if not self.readRegex(cmd_regex, 'fail'):
                return False
            if len(args) == 3 and (not self.readRegex(args[2], 'ignore')):
                return False
        except RegexException as e:
            output('ERROR: %s' % e)
            return False
        if os.path.isfile(cmd_log):
            try:
                test_lines = FileContainer(cmd_log, self._encoding, doOpen=True)
                self.output('Use         log file : %s' % cmd_log)
                self.output('Use         encoding : %s' % self._encoding)
            except IOError as e:
                output(e)
                return False
        elif cmd_log.startswith('systemd-journal'):
            if not FilterSystemd:
                output('Error: systemd library not found. Exiting...')
                return False
            self.output('Use         systemd journal')
            self.output('Use         encoding : %s' % self._encoding)
            (backend, beArgs) = extractOptions(cmd_log)
            flt = FilterSystemd(None, **beArgs)
            flt.setLogEncoding(self._encoding)
            myjournal = flt.getJournalReader()
            journalmatch = self._journalmatch
            self.setDatePattern(None)
            if journalmatch:
                flt.addJournalMatch(journalmatch)
                self.output('Use    journal match : %s' % ' '.join(journalmatch))
            test_lines = journal_lines_gen(flt, myjournal)
        elif self._filter.getMaxLines() <= 1 and '\n' not in cmd_log:
            self.output('Use      single line : %s' % shortstr(cmd_log.replace('\n', '\\n')))
            test_lines = [cmd_log]
        else:
            test_lines = cmd_log.split('\n')
            self.output('Use      multi line : %s line(s)' % len(test_lines))
            for (i, l) in enumerate(test_lines):
                if i >= 5:
                    self.output('| ...')
                    break
                self.output('| %2.2s: %s' % (i + 1, shortstr(l)))
            self.output('`-')
        self.output('')
        self.process(test_lines)
        if not self.printStats():
            return False
        return True

def exec_command_line(*args):
    if False:
        for i in range(10):
            print('nop')
    logging.exitOnIOError = True
    parser = get_opt_parser()
    (opts, args) = parser.parse_args(*args)
    errors = []
    if opts.print_no_missed and opts.print_all_missed:
        errors.append('ERROR: --print-no-missed and --print-all-missed are mutually exclusive.')
    if opts.print_no_ignored and opts.print_all_ignored:
        errors.append('ERROR: --print-no-ignored and --print-all-ignored are mutually exclusive.')
    if not len(args) in (2, 3):
        errors.append('ERROR: provide both <LOG> and <REGEX>.')
    if errors:
        parser.print_help()
        sys.stderr.write('\n' + '\n'.join(errors) + '\n')
        sys.exit(255)
    if not opts.out:
        output('')
        output('Running tests')
        output('=============')
        output('')
    opts.log_level = str2LogLevel(opts.log_level)
    logSys.setLevel(opts.log_level)
    stdout = logging.StreamHandler(sys.stdout)
    fmt = '%(levelname)-1.1s: %(message)s' if opts.verbose <= 1 else ' %(message)s'
    if opts.log_traceback:
        Formatter = FormatterWithTraceBack
        fmt = (opts.full_traceback and ' %(tb)s' or ' %(tbc)s') + fmt
    else:
        Formatter = logging.Formatter
    stdout.setFormatter(Formatter(getVerbosityFormat(opts.verbose, fmt)))
    logSys.addHandler(stdout)
    try:
        fail2banRegex = Fail2banRegex(opts)
    except Exception as e:
        if opts.verbose or logSys.getEffectiveLevel() <= logging.DEBUG:
            logSys.critical(e, exc_info=True)
        else:
            output('ERROR: %s' % e)
        sys.exit(255)
    if not fail2banRegex.start(args):
        sys.exit(255)