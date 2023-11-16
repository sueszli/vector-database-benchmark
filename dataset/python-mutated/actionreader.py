__author__ = 'Cyril Jaquier'
__copyright__ = 'Copyright (c) 2004 Cyril Jaquier'
__license__ = 'GPL'
import os
from .configreader import DefinitionInitConfigReader
from ..helpers import getLogger
from ..server.action import CommandAction
logSys = getLogger(__name__)

class ActionReader(DefinitionInitConfigReader):
    _configOpts = {'actionstart': ['string', None], 'actionstart_on_demand': ['bool', None], 'actionstop': ['string', None], 'actionflush': ['string', None], 'actionreload': ['string', None], 'actioncheck': ['string', None], 'actionrepair': ['string', None], 'actionrepair_on_unban': ['bool', None], 'actionban': ['string', None], 'actionprolong': ['string', None], 'actionreban': ['string', None], 'actionunban': ['string', None], 'norestored': ['bool', None]}

    def __init__(self, file_, jailName, initOpts, **kwargs):
        if False:
            i = 10
            return i + 15
        n = initOpts.get('name')
        if n is None:
            initOpts['name'] = n = jailName
        actname = initOpts.get('actname')
        if actname is None:
            actname = file_
            if n != jailName:
                actname += n[len(jailName):] if n.startswith(jailName) else '-' + n
            initOpts['actname'] = actname
        self._name = actname
        DefinitionInitConfigReader.__init__(self, file_, jailName, initOpts, **kwargs)

    def setFile(self, fileName):
        if False:
            while True:
                i = 10
        self.__file = fileName
        DefinitionInitConfigReader.setFile(self, os.path.join('action.d', fileName))

    def getFile(self):
        if False:
            while True:
                i = 10
        return self.__file

    def setName(self, name):
        if False:
            i = 10
            return i + 15
        self._name = name

    def getName(self):
        if False:
            for i in range(10):
                print('nop')
        return self._name

    def convert(self):
        if False:
            print('Hello World!')
        opts = self.getCombined(ignore=CommandAction._escapedTags | set(('timeout', 'bantime')))
        head = ['set', self._jailName]
        stream = list()
        stream.append(head + ['addaction', self._name])
        multi = []
        for (opt, optval) in opts.items():
            if opt in self._configOpts and (not opt.startswith('known/')):
                multi.append([opt, optval])
        if self._initOpts:
            for (opt, optval) in self._initOpts.items():
                if opt not in self._configOpts and (not opt.startswith('known/')):
                    multi.append([opt, optval])
        if len(multi) > 1:
            stream.append(['multi-set', self._jailName, 'action', self._name, multi])
        elif len(multi):
            stream.append(['set', self._jailName, 'action', self._name] + multi[0])
        return stream