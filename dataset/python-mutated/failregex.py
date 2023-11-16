__author__ = 'Cyril Jaquier'
__copyright__ = 'Copyright (c) 2004 Cyril Jaquier'
__license__ = 'GPL'
import re
import sys
from .ipdns import IPAddr
FTAG_CRE = re.compile('</?[\\w\\-]+/?>')
FCUSTNAME_CRE = re.compile('^(/?)F-([A-Z0-9_\\-]+)$')
R_HOST = ['(?:::f{4,6}:)?(?P<ip4>%s)' % (IPAddr.IP_4_RE,), '(?P<ip6>%s)' % (IPAddr.IP_6_RE,), '(?P<dns>[\\w\\-.^_]*\\w)', '', '', '(?P<cidr>\\d+)', '']
RI_IPV4 = 0
RI_IPV6 = 1
RI_DNS = 2
RI_ADDR = 3
RI_HOST = 4
RI_CIDR = 5
RI_SUBNET = 6
R_HOST[RI_ADDR] = '\\[?(?:%s|%s)\\]?' % (R_HOST[RI_IPV4], R_HOST[RI_IPV6])
R_HOST[RI_HOST] = '(?:%s|%s)' % (R_HOST[RI_ADDR], R_HOST[RI_DNS])
R_HOST[RI_SUBNET] = '\\[?(?:%s|%s)(?:/%s)?\\]?' % (R_HOST[RI_IPV4], R_HOST[RI_IPV6], R_HOST[RI_CIDR])
RH4TAG = {'IP4': R_HOST[RI_IPV4], 'F-IP4/': R_HOST[RI_IPV4], 'IP6': R_HOST[RI_IPV6], 'F-IP6/': R_HOST[RI_IPV6], 'ADDR': R_HOST[RI_ADDR], 'F-ADDR/': R_HOST[RI_ADDR], 'CIDR': R_HOST[RI_CIDR], 'F-CIDR/': R_HOST[RI_CIDR], 'SUBNET': R_HOST[RI_SUBNET], 'F-SUBNET/': R_HOST[RI_SUBNET], 'DNS': R_HOST[RI_DNS], 'F-DNS/': R_HOST[RI_DNS], 'F-ID/': '(?P<fid>\\S+)', 'F-PORT/': '(?P<fport>\\w+)'}
R_MAP = {'id': 'fid', 'port': 'fport'}
try:
    re.search('^re(?i:val)$', 'reVAL')
    R_GLOB2LOCFLAGS = (re.compile('(?<!\\\\)\\((?:\\?:)?(\\(\\?[a-z]+)\\)'), '\\1:')
except:
    R_GLOB2LOCFLAGS = ()

def mapTag2Opt(tag):
    if False:
        while True:
            i = 10
    tag = tag.lower()
    return R_MAP.get(tag, tag)
ALTNAME_PRE = 'alt_'
TUPNAME_PRE = 'tuple_'
COMPLNAME_PRE = (ALTNAME_PRE, TUPNAME_PRE)
COMPLNAME_CRE = re.compile('^(' + '|'.join(COMPLNAME_PRE) + ')(.*?)(?:_\\d+)?$')

class Regex:

    def __init__(self, regex, multiline=False, **kwargs):
        if False:
            i = 10
            return i + 15
        self._matchCache = None
        regex = Regex._resolveHostTag(regex, **kwargs)
        if regex.lstrip() == '':
            raise RegexException('Cannot add empty regex')
        if R_GLOB2LOCFLAGS:
            regex = R_GLOB2LOCFLAGS[0].sub(R_GLOB2LOCFLAGS[1], regex)
        try:
            self._regexObj = re.compile(regex, re.MULTILINE if multiline else 0)
            self._regex = regex
            self._altValues = []
            self._tupleValues = []
            for k in [k for k in self._regexObj.groupindex if len(k) > len(COMPLNAME_PRE[0])]:
                n = COMPLNAME_CRE.match(k)
                if n:
                    (g, n) = (n.group(1), mapTag2Opt(n.group(2)))
                    if g == ALTNAME_PRE:
                        self._altValues.append((k, n))
                    else:
                        self._tupleValues.append((k, n))
            self._altValues.sort()
            self._tupleValues.sort()
            self._altValues = self._altValues if len(self._altValues) else None
            self._tupleValues = self._tupleValues if len(self._tupleValues) else None
        except re.error as e:
            raise RegexException("Unable to compile regular expression '%s':\n%s" % (regex, e))
        self.getGroups = self._getGroupsWithAlt if self._altValues or self._tupleValues else self._getGroups

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return '%s(%r)' % (self.__class__.__name__, self._regex)

    @staticmethod
    def _resolveHostTag(regex, useDns='yes'):
        if False:
            for i in range(10):
                print('nop')
        openTags = dict()
        props = {'nl': 0}

        def substTag(m):
            if False:
                i = 10
                return i + 15
            tag = m.group()
            tn = tag[1:-1]
            if tn == 'HOST':
                return R_HOST[RI_HOST if useDns not in ('no',) else RI_ADDR]
            if tn == 'SKIPLINES':
                nl = props['nl']
                props['nl'] = nl + 1
                return '\\n(?P<skiplines%i>(?:(?:.*\\n)*?))' % (nl,)
            try:
                return RH4TAG[tn]
            except KeyError:
                pass
            m = FCUSTNAME_CRE.match(tn)
            if m:
                m = m.groups()
                tn = m[1]
                if m[0]:
                    if openTags.get(tn):
                        return ')'
                    return tag
                openTags[tn] = 1
                tn = mapTag2Opt(tn)
                return '(?P<%s>' % (tn,)
            return tag
        return FTAG_CRE.sub(substTag, regex)

    def getRegex(self):
        if False:
            for i in range(10):
                print('nop')
        return self._regex

    @staticmethod
    def _tupleLinesBuf(tupleLines):
        if False:
            return 10
        return '\n'.join([''.join(v[::2]) for v in tupleLines]) + '\n'

    def search(self, tupleLines, orgLines=None):
        if False:
            print('Hello World!')
        buf = tupleLines
        if not isinstance(tupleLines, str):
            buf = Regex._tupleLinesBuf(tupleLines)
        self._matchCache = self._regexObj.search(buf)
        if self._matchCache:
            if orgLines is None:
                orgLines = tupleLines
            if len(orgLines) <= 1:
                self._matchedTupleLines = orgLines
                self._unmatchedTupleLines = []
            else:
                try:
                    matchLineStart = self._matchCache.string.rindex('\n', 0, self._matchCache.start() + 1) + 1
                except ValueError:
                    matchLineStart = 0
                try:
                    matchLineEnd = self._matchCache.string.index('\n', self._matchCache.end() - 1) + 1
                except ValueError:
                    matchLineEnd = len(self._matchCache.string)
                lineCount1 = self._matchCache.string.count('\n', 0, matchLineStart)
                lineCount2 = self._matchCache.string.count('\n', 0, matchLineEnd)
                self._matchedTupleLines = orgLines[lineCount1:lineCount2]
                self._unmatchedTupleLines = orgLines[:lineCount1]
                n = 0
                for skippedLine in self.getSkippedLines():
                    for (m, matchedTupleLine) in enumerate(self._matchedTupleLines[n:]):
                        if ''.join(matchedTupleLine[::2]) == skippedLine:
                            self._unmatchedTupleLines.append(self._matchedTupleLines.pop(n + m))
                            n += m
                            break
                self._unmatchedTupleLines.extend(orgLines[lineCount2:])

    def hasMatched(self):
        if False:
            return 10
        if self._matchCache:
            return True
        else:
            return False

    def _getGroups(self):
        if False:
            i = 10
            return i + 15
        return self._matchCache.groupdict()

    def _getGroupsWithAlt(self):
        if False:
            while True:
                i = 10
        fail = self._matchCache.groupdict()
        if self._altValues:
            for (k, n) in self._altValues:
                v = fail.get(k)
                if v and (not fail.get(n)):
                    fail[n] = v
        if self._tupleValues:
            for (k, n) in self._tupleValues:
                v = fail.get(k)
                t = fail.get(n)
                if isinstance(t, tuple):
                    t += (v,)
                else:
                    t = (t, v)
                fail[n] = t
        return fail

    def getGroups(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def getSkippedLines(self):
        if False:
            while True:
                i = 10
        if not self._matchCache:
            return []
        skippedLines = ''
        n = 0
        while True:
            try:
                if self._matchCache.group('skiplines%i' % n) is not None:
                    skippedLines += self._matchCache.group('skiplines%i' % n)
                n += 1
            except IndexError:
                break
            except KeyError:
                if 'PyPy' not in sys.version:
                    raise
                break
        return skippedLines.splitlines(False)

    def getUnmatchedTupleLines(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.hasMatched():
            return []
        else:
            return self._unmatchedTupleLines

    def getUnmatchedLines(self):
        if False:
            while True:
                i = 10
        if not self.hasMatched():
            return []
        else:
            return [''.join(line) for line in self._unmatchedTupleLines]

    def getMatchedTupleLines(self):
        if False:
            print('Hello World!')
        if not self.hasMatched():
            return []
        else:
            return self._matchedTupleLines

    def getMatchedLines(self):
        if False:
            print('Hello World!')
        if not self.hasMatched():
            return []
        else:
            return [''.join(line) for line in self._matchedTupleLines]

class RegexException(Exception):
    pass
FAILURE_ID_GROPS = ('fid', 'ip4', 'ip6', 'dns')
FAILURE_ID_PRESENTS = FAILURE_ID_GROPS + ('mlfid',)

class FailRegex(Regex):

    def __init__(self, regex, prefRegex=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        Regex.__init__(self, regex, **kwargs)
        if not [grp for grp in FAILURE_ID_PRESENTS if grp in self._regexObj.groupindex] and (prefRegex is None or not [grp for grp in FAILURE_ID_PRESENTS if grp in prefRegex._regexObj.groupindex]):
            raise RegexException("No failure-id group in '%s'" % self._regex)

    def getFailID(self, groups=FAILURE_ID_GROPS):
        if False:
            while True:
                i = 10
        fid = None
        for grp in groups:
            try:
                fid = self._matchCache.group(grp)
            except (IndexError, KeyError):
                continue
            if fid is not None:
                break
        if fid is None:
            s = self._matchCache.string
            r = self._matchCache.re
            raise RegexException("No group found in '%s' using '%s'" % (s, r))
        return str(fid)

    def getHost(self):
        if False:
            for i in range(10):
                print('nop')
        return self.getFailID(('ip4', 'ip6', 'dns'))

    def getIP(self):
        if False:
            return 10
        fail = self.getGroups()
        return IPAddr(self.getFailID(('ip4', 'ip6')), int(fail.get('cidr') or IPAddr.CIDR_UNSPEC))