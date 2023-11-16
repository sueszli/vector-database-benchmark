__author__ = 'Cyril Jaquier'
__copyright__ = 'Copyright (c) 2004 Cyril Jaquier'
__license__ = 'GPL'
import re, time
from abc import abstractmethod
from .strptime import reGroupDictStrptime, timeRE, getTimePatternRE
from ..helpers import getLogger
logSys = getLogger(__name__)
RE_GROUPED = re.compile('(?<!(?:\\(\\?))(?<!\\\\)\\((?!\\?)')
RE_GROUP = (re.compile('^((?:\\(\\?\\w+\\))?\\^?(?:\\(\\?\\w+\\))?)(.*?)(\\$?)$'), '\\1(\\2)\\3')
RE_GLOBALFLAGS = re.compile('((?:^|(?!<\\\\))\\(\\?[a-z]+\\))')
RE_EXLINE_NO_BOUNDS = re.compile('^\\{UNB\\}')
RE_EXLINE_BOUND_BEG = re.compile('^\\{\\^LN-BEG\\}')
RE_EXSANC_BOUND_BEG = re.compile('^\\((?:\\?:)?\\^\\|\\\\b\\|\\\\W\\)')
RE_EXEANC_BOUND_BEG = re.compile('\\(\\?=\\\\b\\|\\\\W\\|\\$\\)$')
RE_NO_WRD_BOUND_BEG = re.compile('^\\(*(?:\\(\\?\\w+\\))?(?:\\^|\\(*\\*\\*|\\((?:\\?:)?\\^)')
RE_NO_WRD_BOUND_END = re.compile('(?<!\\\\)(?:\\$\\)?|\\\\b|\\\\s|\\*\\*\\)*)$')
RE_DEL_WRD_BOUNDS = (re.compile('^\\(*(?:\\(\\?\\w+\\))?\\(*\\*\\*|(?<!\\\\)\\*\\*\\)*$'), lambda m: m.group().replace('**', ''))
RE_LINE_BOUND_BEG = re.compile('^(?:\\(\\?\\w+\\))?(?:\\^|\\((?:\\?:)?\\^(?!\\|))')
RE_LINE_BOUND_END = re.compile('(?<![\\\\\\|])(?:\\$\\)?)$')
RE_ALPHA_PATTERN = re.compile('(?<!\\%)\\%[aAbBpc]')
RE_EPOCH_PATTERN = re.compile('(?<!\\\\)\\{L?EPOCH\\}', re.IGNORECASE)

class DateTemplate(object):
    """A template which searches for and returns a date from a log line.

	This is an not functional abstract class which other templates should
	inherit from.

	Attributes
	----------
	name
	regex
	"""
    LINE_BEGIN = 8
    LINE_END = 4
    WORD_BEGIN = 2
    WORD_END = 1

    def __init__(self):
        if False:
            return 10
        self.name = ''
        self.weight = 1.0
        self.flags = 0
        self.hits = 0
        self.time = 0
        self._regex = ''
        self._cRegex = None

    def getRegex(self):
        if False:
            while True:
                i = 10
        return self._regex

    def setRegex(self, regex, wordBegin=True, wordEnd=True):
        if False:
            print('Hello World!')
        "Sets regex to use for searching for date in log line.\n\n\t\tParameters\n\t\t----------\n\t\tregex : str\n\t\t\tThe regex the template will use for searching for a date.\n\t\twordBegin : bool\n\t\t\tDefines whether the regex should be modified to search at beginning of a\n\t\t\tword, by adding special boundary r'(?=^|\\b|\\W)' to start of regex.\n\t\t\tCan be disabled with specifying of ** at front of regex.\n\t\t\tDefault True.\n\t\twordEnd : bool\n\t\t\tDefines whether the regex should be modified to search at end of a word,\n\t\t\tby adding special boundary r'(?=\\b|\\W|$)' to end of regex.\n\t\t\tCan be disabled with specifying of ** at end of regex.\n\t\t\tDefault True.\n\n\t\tRaises\n\t\t------\n\t\tre.error\n\t\t\tIf regular expression fails to compile\n\t\t"
        regex = regex.strip()
        gf = RE_GLOBALFLAGS.search(regex)
        if gf:
            regex = RE_GLOBALFLAGS.sub('', regex, count=1)
        boundBegin = wordBegin and (not RE_NO_WRD_BOUND_BEG.search(regex))
        boundEnd = wordEnd and (not RE_NO_WRD_BOUND_END.search(regex))
        if not RE_GROUPED.search(regex):
            regex = RE_GROUP[0].sub(RE_GROUP[1], regex)
        self.flags = 0
        if boundBegin:
            self.flags |= DateTemplate.WORD_BEGIN if wordBegin != 'start' else DateTemplate.LINE_BEGIN
            if wordBegin != 'start':
                regex = '(?=^|\\b|\\W)' + regex
            else:
                regex = '^(?:\\W{0,2})?' + regex
                if not self.name.startswith('{^LN-BEG}'):
                    self.name = '{^LN-BEG}' + self.name
        if boundEnd:
            self.flags |= DateTemplate.WORD_END
            regex += '(?=\\b|\\W|$)'
        if not self.flags & DateTemplate.LINE_BEGIN and RE_LINE_BOUND_BEG.search(regex):
            self.flags |= DateTemplate.LINE_BEGIN
        if not self.flags & DateTemplate.LINE_END and RE_LINE_BOUND_END.search(regex):
            self.flags |= DateTemplate.LINE_END
        regex = RE_DEL_WRD_BOUNDS[0].sub(RE_DEL_WRD_BOUNDS[1], regex)
        if gf:
            regex = gf.group(1) + regex
        self._regex = regex
        logSys.log(4, '  constructed regex %s', regex)
        self._cRegex = None
    regex = property(getRegex, setRegex, doc='Regex used to search for date.\n\t\t')

    def _compileRegex(self):
        if False:
            return 10
        'Compile regex by first usage.\n\t\t'
        if not self._cRegex:
            try:
                self._cRegex = re.compile(self.regex)
            except Exception as e:
                logSys.error('Compile %r failed, expression %r', self.name, self.regex)
                raise e

    def matchDate(self, line, *args):
        if False:
            while True:
                i = 10
        'Check if regex for date matches on a log line.\n\t\t'
        if not self._cRegex:
            self._compileRegex()
        logSys.log(4, '   search %s', self.regex)
        dateMatch = self._cRegex.search(line, *args)
        if dateMatch:
            self.hits += 1
        return dateMatch

    @abstractmethod
    def getDate(self, line, dateMatch=None, default_tz=None):
        if False:
            print('Hello World!')
        'Abstract method, which should return the date for a log line\n\n\t\tThis should return the date for a log line, typically taking the\n\t\tdate from the part of the line which matched the templates regex.\n\t\tThis requires abstraction, therefore just raises exception.\n\n\t\tParameters\n\t\t----------\n\t\tline : str\n\t\t\tLog line, of which the date should be extracted from.\n\t\tdefault_tz: if no explicit time zone is present in the line\n                            passing this will interpret it as in that time zone.\n\n\t\tRaises\n\t\t------\n\t\tNotImplementedError\n\t\t\tAbstract method, therefore always returns this.\n\t\t'
        raise NotImplementedError('getDate() is abstract')

    @staticmethod
    def unboundPattern(pattern):
        if False:
            while True:
                i = 10
        return RE_EXEANC_BOUND_BEG.sub('', RE_EXSANC_BOUND_BEG.sub('', RE_EXLINE_BOUND_BEG.sub('', RE_EXLINE_NO_BOUNDS.sub('', pattern))))

class DateEpoch(DateTemplate):
    """A date template which searches for Unix timestamps.

	This includes Unix timestamps which appear at start of a line, optionally
	within square braces (nsd), or on SELinux audit log lines.

	Attributes
	----------
	name
	regex
	"""

    def __init__(self, lineBeginOnly=False, pattern=None, longFrm=False):
        if False:
            while True:
                i = 10
        DateTemplate.__init__(self)
        self.name = 'Epoch' if not pattern else pattern
        self._longFrm = longFrm
        self._grpIdx = 1
        epochRE = '\\d{10,11}\\b(?:\\.\\d{3,6})?'
        if longFrm:
            self.name = 'LongEpoch' if not pattern else pattern
            epochRE = '\\d{10,11}(?:\\d{3}(?:\\.\\d{1,6}|\\d{3})?)?'
        if pattern:
            regex = RE_EPOCH_PATTERN.sub(lambda v: '(%s)' % epochRE, pattern)
            if not RE_GROUPED.search(pattern):
                regex = '(' + regex + ')'
            self._grpIdx = 2
            self.setRegex(regex)
        elif not lineBeginOnly:
            regex = '((?:^|(?P<square>(?<=^\\[))|(?P<selinux>(?<=\\baudit\\()))%s)(?:(?(selinux)(?=:\\d+\\)))|(?(square)(?=\\])))' % epochRE
            self.setRegex(regex, wordBegin=False)
        else:
            regex = '((?P<square>(?<=^\\[))?%s)(?(square)(?=\\]))' % epochRE
            self.setRegex(regex, wordBegin='start', wordEnd=True)

    def getDate(self, line, dateMatch=None, default_tz=None):
        if False:
            i = 10
            return i + 15
        'Method to return the date for a log line.\n\n\t\tParameters\n\t\t----------\n\t\tline : str\n\t\t\tLog line, of which the date should be extracted from.\n\t\tdefault_tz: ignored, Unix timestamps are time zone independent\n\n\t\tReturns\n\t\t-------\n\t\t(float, str)\n\t\t\tTuple containing a Unix timestamp, and the string of the date\n\t\t\twhich was matched and in turned used to calculated the timestamp.\n\t\t'
        if not dateMatch:
            dateMatch = self.matchDate(line)
        if dateMatch:
            v = dateMatch.group(self._grpIdx)
            if self._longFrm and len(v) >= 13:
                if len(v) >= 16 and '.' not in v:
                    v = float(v) / 1000000
                else:
                    v = float(v) / 1000
            return (float(v), dateMatch)

class DatePatternRegex(DateTemplate):
    """Date template, with regex/pattern

	Parameters
	----------
	pattern : str
		Sets the date templates pattern.

	Attributes
	----------
	name
	regex
	pattern
	"""
    (_patternRE, _patternName) = getTimePatternRE()
    _patternRE = re.compile(_patternRE)

    def __init__(self, pattern=None, **kwargs):
        if False:
            while True:
                i = 10
        super(DatePatternRegex, self).__init__()
        self._pattern = None
        if pattern is not None:
            self.setRegex(pattern, **kwargs)

    @property
    def pattern(self):
        if False:
            i = 10
            return i + 15
        'The pattern used for regex with strptime "%" time fields.\n\n\t\tThis should be a valid regular expression, of which matching string\n\t\twill be extracted from the log line. strptime style "%" fields will\n\t\tbe replaced by appropriate regular expressions, or custom regex\n\t\tgroups with names as per the strptime fields can also be used\n\t\tinstead.\n\t\t'
        return self._pattern

    @pattern.setter
    def pattern(self, pattern):
        if False:
            while True:
                i = 10
        self.setRegex(pattern)

    def setRegex(self, pattern, wordBegin=True, wordEnd=True):
        if False:
            while True:
                i = 10
        self._pattern = pattern
        if RE_EXLINE_NO_BOUNDS.search(pattern):
            pattern = RE_EXLINE_NO_BOUNDS.sub('', pattern)
            wordBegin = wordEnd = False
        if wordBegin and RE_EXLINE_BOUND_BEG.search(pattern):
            pattern = RE_EXLINE_BOUND_BEG.sub('', pattern)
            wordBegin = 'start'
        try:
            fmt = self._patternRE.sub('%(\\1)s', pattern)
            self.name = fmt % self._patternName
            regex = fmt % timeRE
            if RE_ALPHA_PATTERN.search(pattern):
                regex = '(?iu)' + regex
            super(DatePatternRegex, self).setRegex(regex, wordBegin, wordEnd)
        except Exception as e:
            raise TypeError("Failed to set datepattern '%s' (may be an invalid format or unescaped percent char): %s" % (pattern, e))

    def getDate(self, line, dateMatch=None, default_tz=None):
        if False:
            return 10
        'Method to return the date for a log line.\n\n\t\tThis uses a custom version of strptime, using the named groups\n\t\tfrom the instances `pattern` property.\n\n\t\tParameters\n\t\t----------\n\t\tline : str\n\t\t\tLog line, of which the date should be extracted from.\n\t\tdefault_tz: optionally used to correct timezone\n\n\t\tReturns\n\t\t-------\n\t\t(float, str)\n\t\t\tTuple containing a Unix timestamp, and the string of the date\n\t\t\twhich was matched and in turned used to calculated the timestamp.\n\t\t'
        if not dateMatch:
            dateMatch = self.matchDate(line)
        if dateMatch:
            return (reGroupDictStrptime(dateMatch.groupdict(), default_tz=default_tz), dateMatch)

class DateTai64n(DateTemplate):
    """A date template which matches TAI64N formate timestamps.

	Attributes
	----------
	name
	regex
	"""

    def __init__(self, wordBegin=False):
        if False:
            return 10
        DateTemplate.__init__(self)
        self.name = 'TAI64N'
        self.setRegex('@[0-9a-f]{24}', wordBegin=wordBegin)

    def getDate(self, line, dateMatch=None, default_tz=None):
        if False:
            print('Hello World!')
        'Method to return the date for a log line.\n\n\t\tParameters\n\t\t----------\n\t\tline : str\n\t\t\tLog line, of which the date should be extracted from.\n\t\tdefault_tz: ignored, since TAI is time zone independent\n\n\t\tReturns\n\t\t-------\n\t\t(float, str)\n\t\t\tTuple containing a Unix timestamp, and the string of the date\n\t\t\twhich was matched and in turned used to calculated the timestamp.\n\t\t'
        if not dateMatch:
            dateMatch = self.matchDate(line)
        if dateMatch:
            value = dateMatch.group(1)
            seconds_since_epoch = value[2:17]
            return (int(seconds_since_epoch, 16), dateMatch)