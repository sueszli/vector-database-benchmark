__author__ = 'Cyril Jaquier and Fail2Ban Contributors'
__copyright__ = 'Copyright (c) 2004 Cyril Jaquier'
__license__ = 'GPL'
import copy
import time
from threading import Lock
from .datetemplate import re, DateTemplate, DatePatternRegex, DateTai64n, DateEpoch, RE_EPOCH_PATTERN
from .strptime import validateTimeZone
from .utils import Utils
from ..helpers import getLogger
logSys = getLogger(__name__)
logLevel = 5
RE_DATE_PREMATCH = re.compile('(?<!\\\\)\\{DATE\\}', re.IGNORECASE)
DD_patternCache = Utils.Cache(maxCount=1000, maxTime=60 * 60)

def _getPatternTemplate(pattern, key=None):
    if False:
        i = 10
        return i + 15
    if key is None:
        key = pattern
        if '%' not in pattern:
            key = pattern.upper()
    template = DD_patternCache.get(key)
    if not template:
        if 'EPOCH' in key:
            if RE_EPOCH_PATTERN.search(pattern):
                template = DateEpoch(pattern=pattern, longFrm='LEPOCH' in key)
            elif key in ('EPOCH', '{^LN-BEG}EPOCH', '^EPOCH'):
                template = DateEpoch(lineBeginOnly=key != 'EPOCH')
            elif key in ('LEPOCH', '{^LN-BEG}LEPOCH', '^LEPOCH'):
                template = DateEpoch(lineBeginOnly=key != 'LEPOCH', longFrm=True)
        if template is None:
            if key in ('TAI64N', '{^LN-BEG}TAI64N', '^TAI64N'):
                template = DateTai64n(wordBegin='start' if key != 'TAI64N' else False)
            else:
                template = DatePatternRegex(pattern)
    DD_patternCache.set(key, template)
    return template

def _getAnchoredTemplate(template, wrap=lambda s: '{^LN-BEG}' + s):
    if False:
        while True:
            i = 10
    name = wrap(template.name)
    template2 = DD_patternCache.get(name)
    if not template2:
        regex = wrap(getattr(template, 'pattern', template.regex))
        if hasattr(template, 'pattern'):
            template2 = DD_patternCache.get(regex)
        if not template2:
            if not hasattr(template, 'pattern'):
                template2 = _getPatternTemplate(name)
            else:
                template2 = _getPatternTemplate(regex)
    return template2

class DateDetectorCache(object):
    """Implements the caching of the default templates list.
	"""

    def __init__(self):
        if False:
            while True:
                i = 10
        self.__lock = Lock()
        self.__templates = list()

    @property
    def templates(self):
        if False:
            print('Hello World!')
        'List of template instances managed by the detector.\n\t\t'
        if self.__templates:
            return self.__templates
        with self.__lock:
            if self.__templates:
                return self.__templates
            self._addDefaultTemplate()
            return self.__templates

    def _cacheTemplate(self, template):
        if False:
            while True:
                i = 10
        "Cache Fail2Ban's default template.\n\n\t\t"
        name = template.name
        if not name.startswith('{^LN-BEG}') and (not name.startswith('^')) and hasattr(template, 'regex'):
            template2 = _getAnchoredTemplate(template)
            if template2.name != name:
                template2.weight = 100.0
                self.__tmpcache[0].append(template2)
        self.__tmpcache[1].append(template)
    DEFAULT_TEMPLATES = ['%ExY(?P<_sep>[-/.])%m(?P=_sep)%d(?:T|  ?)%H:%M:%S(?:[.,]%f)?(?:\\s*%z)?', '(?:%a )?%b %d %k:%M:%S(?:\\.%f)?(?: %ExY)?', '(?:%a )?%b %d %ExY %k:%M:%S(?:\\.%f)?', '%d(?P<_sep>[-/])%m(?P=_sep)(?:%ExY|%Exy) %k:%M:%S', '%d(?P<_sep>[-/])%b(?P=_sep)%ExY[ :]?%H:%M:%S(?:\\.%f)?(?: %z)?', '%m/%d/%ExY:%H:%M:%S', '%m-%d-%ExY %k:%M:%S(?:\\.%f)?', 'EPOCH', '{^LN-BEG}%H:%M:%S', '^<%m/%d/%Exy@%H:%M:%S>', '%Exy%Exm%Exd  ?%H:%M:%S', '%b %d, %ExY %I:%M:%S %p', '^%b-%d-%Exy %k:%M:%S', '%ExY%Exm%Exd(?:T|  ?)%ExH%ExM%ExS(?:[.,]%f)?(?:\\s*%z)?', '(?:%Z )?(?:%a )?%b %d %k:%M:%S(?:\\.%f)?(?: %ExY)?', '(?:%z )?(?:%a )?%b %d %k:%M:%S(?:\\.%f)?(?: %ExY)?', 'TAI64N']

    @property
    def defaultTemplates(self):
        if False:
            print('Hello World!')
        if isinstance(DateDetectorCache.DEFAULT_TEMPLATES[0], str):
            for (i, dt) in enumerate(DateDetectorCache.DEFAULT_TEMPLATES):
                dt = _getPatternTemplate(dt)
                DateDetectorCache.DEFAULT_TEMPLATES[i] = dt
        return DateDetectorCache.DEFAULT_TEMPLATES

    def _addDefaultTemplate(self):
        if False:
            while True:
                i = 10
        "Add resp. cache Fail2Ban's default set of date templates.\n\t\t"
        self.__tmpcache = ([], [])
        for dt in self.defaultTemplates:
            self._cacheTemplate(dt)
        self.__templates = self.__tmpcache[0] + self.__tmpcache[1]
        del self.__tmpcache

class DateDetectorTemplate(object):
    """Used for "shallow copy" of the template object.

	Prevents collectively usage of hits/lastUsed in cached templates
	"""
    __slots__ = ('template', 'hits', 'lastUsed', 'distance')

    def __init__(self, template):
        if False:
            return 10
        self.template = template
        self.hits = 0
        self.lastUsed = 0
        self.distance = 2147483647

    @property
    def weight(self):
        if False:
            while True:
                i = 10
        return self.hits * self.template.weight / max(1, self.distance)

    def __getattr__(self, name):
        if False:
            for i in range(10):
                print('nop')
        ' Returns attribute of template (called for parameters not in slots)\n\t\t'
        return getattr(self.template, name)

class DateDetector(object):
    """Manages one or more date templates to find a date within a log line.

	Attributes
	----------
	templates
	"""
    _defCache = DateDetectorCache()

    def __init__(self):
        if False:
            return 10
        self.__templates = list()
        self.__known_names = set()
        self.__unusedTime = 300
        self.__lastPos = (1, None)
        self.__lastEndPos = (2147483647, None)
        self.__lastTemplIdx = 2147483647
        self.__firstUnused = 0
        self.__preMatch = None
        self.__default_tz = None

    def _appendTemplate(self, template, ignoreDup=False):
        if False:
            i = 10
            return i + 15
        name = template.name
        if name in self.__known_names:
            if ignoreDup:
                return
            raise ValueError('There is already a template with name %s' % name)
        self.__known_names.add(name)
        self.__templates.append(DateDetectorTemplate(template))

    def appendTemplate(self, template):
        if False:
            print('Hello World!')
        'Add a date template to manage and use in search of dates.\n\n\t\tParameters\n\t\t----------\n\t\ttemplate : DateTemplate or str\n\t\t\tCan be either a `DateTemplate` instance, or a string which will\n\t\t\tbe used as the pattern for the `DatePatternRegex` template. The\n\t\t\ttemplate will then be added to the detector.\n\n\t\tRaises\n\t\t------\n\t\tValueError\n\t\t\tIf a template already exists with the same name.\n\t\t'
        if isinstance(template, str):
            key = pattern = template
            if '%' not in pattern:
                key = pattern.upper()
            template = DD_patternCache.get(key)
            if not template:
                if key in ('{^LN-BEG}', '{DEFAULT}'):
                    flt = lambda template: template.flags & DateTemplate.LINE_BEGIN if key == '{^LN-BEG}' else None
                    self.addDefaultTemplate(flt)
                    return
                elif '{DATE}' in key:
                    self.addDefaultTemplate(preMatch=pattern, allDefaults=False)
                    return
                elif key == '{NONE}':
                    template = _getPatternTemplate('{UNB}^', key)
                else:
                    template = _getPatternTemplate(pattern, key)
            DD_patternCache.set(key, template)
        self._appendTemplate(template)
        logSys.info('  date pattern `%r`: `%s`', getattr(template, 'pattern', ''), template.name)
        logSys.debug('  date pattern regex for %r: %s', getattr(template, 'pattern', ''), template.regex)

    def addDefaultTemplate(self, filterTemplate=None, preMatch=None, allDefaults=True):
        if False:
            print('Hello World!')
        "Add Fail2Ban's default set of date templates.\n\t\t"
        ignoreDup = len(self.__templates) > 0
        for template in DateDetector._defCache.templates if allDefaults else DateDetector._defCache.defaultTemplates:
            if filterTemplate is not None and (not filterTemplate(template)):
                continue
            if preMatch is not None:
                template = _getAnchoredTemplate(template, wrap=lambda s: RE_DATE_PREMATCH.sub(lambda m: DateTemplate.unboundPattern(s), preMatch))
            self._appendTemplate(template, ignoreDup=ignoreDup)

    @property
    def templates(self):
        if False:
            return 10
        'List of template instances managed by the detector.\n\t\t'
        return self.__templates

    def matchTime(self, line):
        if False:
            print('Hello World!')
        "Attempts to find date on a log line using templates.\n\n\t\tThis uses the templates' `matchDate` method in an attempt to find\n\t\ta date. It also increments the match hit count for the winning\n\t\ttemplate.\n\n\t\tParameters\n\t\t----------\n\t\tline : str\n\t\t\tLine which is searched by the date templates.\n\n\t\tReturns\n\t\t-------\n\t\tre.MatchObject, DateTemplate\n\t\t\tThe regex match returned from the first successfully matched\n\t\t\ttemplate.\n\t\t"
        if not len(self.__templates):
            self.addDefaultTemplate()
        log = logSys.log if logSys.getEffectiveLevel() <= logLevel else lambda *args: None
        log(logLevel - 1, 'try to match time for line: %.120s', line)
        match = None
        found = (None, 2147483647, 2147483647, -1)
        ignoreBySearch = 2147483647
        i = self.__lastTemplIdx
        if i < len(self.__templates):
            ddtempl = self.__templates[i]
            template = ddtempl.template
            if template.flags & (DateTemplate.LINE_BEGIN | DateTemplate.LINE_END):
                log(logLevel - 1, '  try to match last anchored template #%02i ...', i)
                match = template.matchDate(line)
                ignoreBySearch = i
            else:
                (distance, endpos) = (self.__lastPos[0], self.__lastEndPos[0])
                log(logLevel - 1, '  try to match last template #%02i (from %r to %r): ...%r==%r %s %r==%r...', i, distance, endpos, line[distance - 1:distance], self.__lastPos[1], line[distance:endpos], line[endpos:endpos + 1], self.__lastEndPos[2])
                if (line[distance - 1:distance] == self.__lastPos[1] or (line[distance:distance + 1] == self.__lastPos[2] and (not self.__lastPos[2].isalnum()))) and (line[endpos:endpos + 1] == self.__lastEndPos[2] or (line[endpos - 1:endpos] == self.__lastEndPos[1] and (not self.__lastEndPos[1].isalnum()))):
                    log(logLevel - 1, '  boundaries are correct, search in part %r', line[distance:endpos])
                    match = template.matchDate(line, distance, endpos)
                else:
                    log(logLevel - 1, '  boundaries show conflict, try whole search')
                    match = template.matchDate(line)
                    ignoreBySearch = i
            if match:
                distance = match.start()
                endpos = match.end()
                if len(self.__templates) == 1 or template.flags & (DateTemplate.LINE_BEGIN | DateTemplate.LINE_END) or (distance == self.__lastPos[0] and endpos == self.__lastEndPos[0]):
                    log(logLevel, '  matched last time template #%02i', i)
                else:
                    log(logLevel, '  ** last pattern collision - pattern change, reserve & search ...')
                    found = (match, distance, endpos, i)
                    match = None
            else:
                log(logLevel, '  ** last pattern not found - pattern change, search ...')
        if not match:
            log(logLevel, ' search template (%i) ...', len(self.__templates))
            i = 0
            for ddtempl in self.__templates:
                if i == ignoreBySearch:
                    i += 1
                    continue
                log(logLevel - 1, '  try template #%02i: %s', i, ddtempl.name)
                template = ddtempl.template
                match = template.matchDate(line)
                if match:
                    distance = match.start()
                    endpos = match.end()
                    log(logLevel, '  matched time template #%02i (at %r <= %r, %r) %s', i, distance, ddtempl.distance, self.__lastPos[0], template.name)
                    if i + 1 >= len(self.__templates):
                        break
                    if template.flags & (DateTemplate.LINE_BEGIN | DateTemplate.LINE_END):
                        break
                    if (distance == 0 and ddtempl.hits) and (not self.__templates[i + 1].template.hits):
                        break
                    if distance > ddtempl.distance or distance > self.__lastPos[0]:
                        log(logLevel, '  ** distance collision - pattern change, reserve')
                        if distance < found[1]:
                            found = (match, distance, endpos, i)
                        match = None
                        i += 1
                        continue
                    break
                i += 1
            if not match and found[0]:
                (match, distance, endpos, i) = found
                log(logLevel, '  use best time template #%02i', i)
                ddtempl = self.__templates[i]
                template = ddtempl.template
        if match:
            ddtempl.hits += 1
            ddtempl.lastUsed = time.time()
            ddtempl.distance = distance
            if self.__firstUnused == i:
                self.__firstUnused += 1
            self.__lastPos = (distance, line[distance - 1:distance], line[distance])
            self.__lastEndPos = (endpos, line[endpos - 1], line[endpos:endpos + 1])
            if i and i != self.__lastTemplIdx:
                i = self._reorderTemplate(i)
            self.__lastTemplIdx = i
            return (match, template)
        log(logLevel, ' no template.')
        return (None, None)

    @property
    def default_tz(self):
        if False:
            print('Hello World!')
        return self.__default_tz

    @default_tz.setter
    def default_tz(self, value):
        if False:
            i = 10
            return i + 15
        self.__default_tz = validateTimeZone(value)

    def getTime(self, line, timeMatch=None):
        if False:
            i = 10
            return i + 15
        "Attempts to return the date on a log line using templates.\n\n\t\tThis uses the templates' `getDate` method in an attempt to find\n\t\ta date. \n\t\tFor the faster usage, always specify a parameter timeMatch (the previous tuple result\n\t\tof the matchTime), then this will work without locking and without cycle over templates.\n\n\t\tParameters\n\t\t----------\n\t\tline : str\n\t\t\tLine which is searched by the date templates.\n\n\t\tReturns\n\t\t-------\n\t\tfloat\n\t\t\tThe Unix timestamp returned from the first successfully matched\n\t\t\ttemplate or None if not found.\n\t\t"
        if timeMatch is None:
            timeMatch = self.matchTime(line)
        template = timeMatch[1]
        if template is not None:
            try:
                date = template.getDate(line, timeMatch[0], default_tz=self.__default_tz)
                if date is not None:
                    if logSys.getEffectiveLevel() <= logLevel:
                        logSys.log(logLevel, '  got time %f for %r using template %s', date[0], date[1].group(1), template.name)
                    return date
            except ValueError:
                pass
        return None

    def _reorderTemplate(self, num):
        if False:
            while True:
                i = 10
        'Reorder template (bubble up) in template list if hits grows enough.\n\n\t\tParameters\n\t\t----------\n\t\tnum : int\n\t\t\tIndex of template should be moved.\n\t\t'
        if num:
            templates = self.__templates
            ddtempl = templates[num]
            if logSys.getEffectiveLevel() <= logLevel:
                logSys.log(logLevel, '  -> reorder template #%02i, hits: %r', num, ddtempl.hits)
            untime = ddtempl.lastUsed - self.__unusedTime
            weight = ddtempl.weight
            pos = self.__firstUnused if self.__firstUnused < num else num // 2

            def _moveable():
                if False:
                    for i in range(10):
                        print('nop')
                pweight = templates[pos].weight
                if logSys.getEffectiveLevel() <= logLevel:
                    logSys.log(logLevel, '  -> compare template #%02i & #%02i, weight %.3f > %.3f, hits %r > %r', num, pos, weight, pweight, ddtempl.hits, templates[pos].hits)
                return weight > pweight or untime > templates[pos].lastUsed
            if not _moveable():
                if pos == num - 1:
                    return num
                pos = num - 1
                if not _moveable():
                    return num
            del templates[num]
            templates[pos:0] = [ddtempl]
            while self.__firstUnused < len(templates) and templates[self.__firstUnused].hits:
                self.__firstUnused += 1
            if logSys.getEffectiveLevel() <= logLevel:
                logSys.log(logLevel, '  -> moved template #%02i -> #%02i', num, pos)
            return pos
        return num