__author__ = 'Cyril Jaquier'
__copyright__ = 'Copyright (c) 2004 Cyril Jaquier'
__license__ = 'GPL'
import os
import shlex
from .configreader import DefinitionInitConfigReader
from ..helpers import getLogger
logSys = getLogger(__name__)

class FilterReader(DefinitionInitConfigReader):
    _configOpts = {'usedns': ['string', None], 'prefregex': ['string', None], 'ignoreregex': ['string', None], 'failregex': ['string', None], 'maxlines': ['int', None], 'datepattern': ['string', None], 'journalmatch': ['string', None]}

    def setFile(self, fileName):
        if False:
            i = 10
            return i + 15
        self.__file = fileName
        DefinitionInitConfigReader.setFile(self, os.path.join('filter.d', fileName))

    def getFile(self):
        if False:
            print('Hello World!')
        return self.__file

    def applyAutoOptions(self, backend):
        if False:
            while True:
                i = 10
        if not self._initOpts.get('logtype') and (not self.has_option('Definition', 'logtype', False)):
            self._initOpts['logtype'] = ['file', 'journal'][int(backend.startswith('systemd'))]

    def convert(self):
        if False:
            i = 10
            return i + 15
        stream = list()
        opts = self.getCombined()
        if not len(opts):
            return stream
        return FilterReader._fillStream(stream, opts, self._jailName)

    @staticmethod
    def _fillStream(stream, opts, jailName):
        if False:
            for i in range(10):
                print('nop')
        prio0idx = 0
        for (opt, value) in opts.items():
            if value is None:
                continue
            if opt in ('failregex', 'ignoreregex'):
                multi = []
                for regex in value.split('\n'):
                    if regex != '':
                        multi.append(regex)
                if len(multi) > 1:
                    stream.append(['multi-set', jailName, 'add' + opt, multi])
                elif len(multi):
                    stream.append(['set', jailName, 'add' + opt, multi[0]])
            elif opt in ('usedns', 'maxlines', 'prefregex'):
                stream.insert(0 if opt == 'usedns' else prio0idx, ['set', jailName, opt, value])
                prio0idx += 1
            elif opt in 'datepattern':
                stream.append(['set', jailName, opt, value])
            elif opt == 'journalmatch':
                for match in value.split('\n'):
                    if match == '':
                        continue
                    stream.append(['set', jailName, 'addjournalmatch'] + shlex.split(match))
        return stream