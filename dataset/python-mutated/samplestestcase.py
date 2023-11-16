__copyright__ = 'Copyright (c) 2013 Steven Hiscocks'
__license__ = 'GPL'
import datetime
import inspect
import json
import os
import re
import sys
import time
import unittest
from ..server.failregex import Regex
from ..server.filter import Filter, FileContainer
from ..client.filterreader import FilterReader
from .utils import setUpMyTime, tearDownMyTime, TEST_NOW, CONFIG_DIR
TEST_NOW_STR = datetime.datetime.utcfromtimestamp(TEST_NOW).isoformat()
TEST_CONFIG_DIR = os.path.join(os.path.dirname(__file__), 'config')
TEST_FILES_DIR = os.path.join(os.path.dirname(__file__), 'files')
RE_HOST = Regex._resolveHostTag('<HOST>')
RE_WRONG_GREED = re.compile('\\.[+\\*](?!\\?)[^\\$\\^]*' + re.escape(RE_HOST) + '.*(?:\\.[+\\*].*|[^\\$])$')

class FilterSamplesRegex(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        'Call before every test case.'
        super(FilterSamplesRegex, self).setUp()
        self._filters = dict()
        self._filterTests = None
        setUpMyTime()

    def tearDown(self):
        if False:
            return 10
        'Call after every test case.'
        super(FilterSamplesRegex, self).tearDown()
        tearDownMyTime()

    def testFiltersPresent(self):
        if False:
            for i in range(10):
                print('nop')
        'Check to ensure some tests exist'
        self.assertTrue(len([test for test in inspect.getmembers(self) if test[0].startswith('testSampleRegexs')]) >= 10, 'Expected more FilterSampleRegexs tests')

    def testReWrongGreedyCatchAll(self):
        if False:
            i = 10
            return i + 15
        'Tests regexp RE_WRONG_GREED is intact (positive/negative)'
        self.assertTrue(RE_WRONG_GREED.search('greedy .* test' + RE_HOST + ' test not hard-anchored'))
        self.assertTrue(RE_WRONG_GREED.search('greedy .+ test' + RE_HOST + ' test vary .* anchored$'))
        self.assertFalse(RE_WRONG_GREED.search('greedy .* test' + RE_HOST + ' test no catch-all, hard-anchored$'))
        self.assertFalse(RE_WRONG_GREED.search('non-greedy .*? test' + RE_HOST + ' test not hard-anchored'))
        self.assertFalse(RE_WRONG_GREED.search('non-greedy .+? test' + RE_HOST + ' test vary catch-all .* anchored$'))

    def _readFilter(self, fltName, name, basedir, opts=None):
        if False:
            for i in range(10):
                print('nop')
        flt = self._filters.get(fltName)
        if flt:
            return flt
        flt = Filter(None)
        flt.returnRawHost = True
        flt.checkAllRegex = True
        flt.checkFindTime = False
        flt.active = True
        if opts is None:
            opts = dict()
        opts = opts.copy()
        filterConf = FilterReader(name, 'jail', opts, basedir=basedir, share_config=unittest.F2B.share_config)
        self.assertEqual(filterConf.getFile(), name)
        self.assertEqual(filterConf.getJailName(), 'jail')
        filterConf.read()
        filterConf.getOptions({})
        for opt in filterConf.convert():
            if opt[0] == 'multi-set':
                optval = opt[3]
            elif opt[0] == 'set':
                optval = [opt[3]]
            else:
                self.fail('Unexpected config-token %r in stream' % (opt,))
            for optval in optval:
                if opt[2] == 'prefregex':
                    flt.prefRegex = optval
                elif opt[2] == 'addfailregex':
                    flt.addFailRegex(optval)
                elif opt[2] == 'addignoreregex':
                    flt.addIgnoreRegex(optval)
                elif opt[2] == 'maxlines':
                    flt.setMaxLines(optval)
                elif opt[2] == 'datepattern':
                    flt.setDatePattern(optval)
        regexList = flt.getFailRegex()
        for fr in regexList:
            if RE_WRONG_GREED.search(fr):
                raise AssertionError('Following regexp of "%s" contains greedy catch-all before <HOST>, that is not hard-anchored at end or has not precise sub expression after <HOST>:\n%s' % (fltName, str(fr).replace(RE_HOST, '<HOST>')))
        flt = [flt, set()]
        self._filters[fltName] = flt
        return flt

    @staticmethod
    def _filterOptions(opts):
        if False:
            return 10
        return dict(((k, v) for (k, v) in opts.items() if not k.startswith('test.')))

def testSampleRegexsFactory(name, basedir):
    if False:
        while True:
            i = 10

    def testFilter(self):
        if False:
            return 10
        self.assertTrue(os.path.isfile(os.path.join(TEST_FILES_DIR, 'logs', name)), "No sample log file available for '%s' filter" % name)
        filenames = [name]
        regexsUsedRe = set()
        commonOpts = {}
        faildata = {}
        i = 0
        while i < len(filenames):
            filename = filenames[i]
            i += 1
            logFile = FileContainer(os.path.join(TEST_FILES_DIR, 'logs', filename), 'UTF-8', doOpen=True)
            logFile.waitForLineEnd = False
            ignoreBlock = False
            lnnum = 0
            for line in logFile:
                lnnum += 1
                jsonREMatch = re.match('^#+ ?(failJSON|(?:file|filter)Options|addFILE):(.+)$', line)
                if jsonREMatch:
                    try:
                        faildata = json.loads(jsonREMatch.group(2))
                        if jsonREMatch.group(1) == 'fileOptions':
                            commonOpts = faildata
                            continue
                        if jsonREMatch.group(1) == 'filterOptions':
                            self._filterTests = []
                            ignoreBlock = False
                            for faildata in faildata if isinstance(faildata, list) else [faildata]:
                                if commonOpts:
                                    opts = commonOpts.copy()
                                    opts.update(faildata)
                                else:
                                    opts = faildata
                                self.assertTrue(isinstance(opts, dict))
                                if opts.get('test.condition'):
                                    ignoreBlock = not eval(opts.get('test.condition'))
                                if not ignoreBlock:
                                    fltOpts = self._filterOptions(opts)
                                    fltName = opts.get('test.filter-name')
                                    if not fltName:
                                        fltName = str(fltOpts) if fltOpts else ''
                                    fltName = name + fltName
                                    flt = self._readFilter(fltName, name, basedir, opts=fltOpts)
                                    self._filterTests.append((fltName, flt, opts))
                            continue
                        if jsonREMatch.group(1) == 'addFILE':
                            filenames.append(faildata)
                            continue
                    except ValueError as e:
                        raise ValueError('%s: %s:%i' % (e, logFile.getFileName(), lnnum))
                    line = next(logFile)
                elif ignoreBlock or line.startswith('#') or (not line.strip()):
                    continue
                else:
                    faildata = {}
                if ignoreBlock:
                    continue
                if not self._filterTests:
                    fltName = name
                    flt = self._readFilter(fltName, name, basedir, opts=None)
                    self._filterTests = [(fltName, flt, {})]
                line = line.rstrip('\r\n')
                for (fltName, flt, opts) in self._filterTests:
                    if faildata.get('constraint') and (not eval(faildata['constraint'])):
                        continue
                    (flt, regexsUsedIdx) = flt
                    regexList = flt.getFailRegex()
                    failregex = -1
                    try:
                        fail = {}
                        if opts.get('logtype') != 'journal':
                            ret = flt.processLine(line)
                        else:
                            if opts.get('test.prefix-line'):
                                line = opts.get('test.prefix-line') + line
                            ret = flt.processLine(('', TEST_NOW_STR, line), TEST_NOW)
                        if ret:
                            found = []
                            for ret in ret:
                                (failregex, fid, fail2banTime, fail) = ret
                                if fid is None or fail.get('nofail'):
                                    regexsUsedIdx.add(failregex)
                                    regexsUsedRe.add(regexList[failregex])
                                    continue
                                found.append(ret)
                            ret = found
                        if not ret:
                            self.assertFalse(faildata.get('match', False), 'Line not matched when should have')
                            continue
                        self.assertTrue(faildata.get('match', False), "Line matched when shouldn't have")
                        self.assertEqual(len(ret), 1, 'Multiple regexs matched %r' % [x[0] for x in ret])
                        for ret in ret:
                            (failregex, fid, fail2banTime, fail) = ret
                            for (k, v) in faildata.items():
                                if k not in ('time', 'match', 'desc', 'constraint'):
                                    fv = fail.get(k, None)
                                    if fv is None:
                                        if k == 'host':
                                            fv = fid
                                        if k == 'attempts':
                                            fv = len(fail.get('matches', {}))
                                    if isinstance(fv, (set, list, dict)):
                                        self.assertSortedEqual(fv, v)
                                        continue
                                    self.assertEqual(fv, v)
                            t = faildata.get('time', None)
                            if t is not None:
                                try:
                                    jsonTimeLocal = datetime.datetime.strptime(t, '%Y-%m-%dT%H:%M:%S')
                                except ValueError:
                                    jsonTimeLocal = datetime.datetime.strptime(t, '%Y-%m-%dT%H:%M:%S.%f')
                                jsonTime = time.mktime(jsonTimeLocal.timetuple())
                                jsonTime += jsonTimeLocal.microsecond / 1000000.0
                                self.assertEqual(fail2banTime, jsonTime, 'UTC Time  mismatch %s (%s) != %s (%s)  (diff %.3f seconds)' % (fail2banTime, time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(fail2banTime)), jsonTime, time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(jsonTime)), fail2banTime - jsonTime))
                            regexsUsedIdx.add(failregex)
                            regexsUsedRe.add(regexList[failregex])
                    except AssertionError as e:
                        import pprint
                        raise AssertionError('%s: %s on: %s:%i, line:\n  %s\nregex (%s):\n  %s\nfaildata: %s\nfail: %s' % (fltName, e, logFile.getFileName(), lnnum, line, failregex, regexList[failregex] if failregex != -1 else None, '\n'.join(pprint.pformat(faildata).splitlines()), '\n'.join(pprint.pformat(fail).splitlines())))
        for (fltName, flt) in self._filters.items():
            (flt, regexsUsedIdx) = flt
            regexList = flt.getFailRegex()
            for (failRegexIndex, failRegex) in enumerate(regexList):
                self.assertTrue(failRegexIndex in regexsUsedIdx or failRegex in regexsUsedRe, '%s: Regex has no samples: %i: %r' % (fltName, failRegexIndex, failRegex))
    return testFilter
for (basedir_, filter_) in ((CONFIG_DIR, lambda x: not x.endswith('common.conf') and x.endswith('.conf')), (TEST_CONFIG_DIR, lambda x: x.startswith('zzz-') and x.endswith('.conf'))):
    for filter_ in filter(filter_, os.listdir(os.path.join(basedir_, 'filter.d'))):
        filterName = filter_.rpartition('.')[0]
        if not filterName.startswith('.'):
            setattr(FilterSamplesRegex, 'testSampleRegexs%s' % filterName.upper(), testSampleRegexsFactory(filterName, basedir_))