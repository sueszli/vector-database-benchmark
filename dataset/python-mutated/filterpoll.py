__author__ = 'Cyril Jaquier, Yaroslav Halchenko'
__copyright__ = 'Copyright (c) 2004 Cyril Jaquier; 2012 Yaroslav Halchenko'
__license__ = 'GPL'
import os
import time
from .filter import FileFilter
from .utils import Utils
from ..helpers import getLogger, logging
logSys = getLogger(__name__)

class FilterPoll(FileFilter):

    def __init__(self, jail):
        if False:
            i = 10
            return i + 15
        FileFilter.__init__(self, jail)
        self.__prevStats = dict()
        self.__file404Cnt = dict()
        logSys.debug('Created FilterPoll')

    def _addLogPath(self, path):
        if False:
            while True:
                i = 10
        self.__prevStats[path] = (0, None, None)
        self.__file404Cnt[path] = 0

    def _delLogPath(self, path):
        if False:
            print('Hello World!')
        del self.__prevStats[path]
        del self.__file404Cnt[path]

    def getModified(self, modlst):
        if False:
            print('Hello World!')
        for filename in self.getLogPaths():
            if self.isModified(filename):
                modlst.append(filename)
        return modlst

    def run(self):
        if False:
            return 10
        while self.active:
            try:
                if logSys.getEffectiveLevel() <= 4:
                    logSys.log(4, 'Woke up idle=%s with %d files monitored', self.idle, self.getLogCount())
                if self.idle:
                    if not Utils.wait_for(lambda : not self.active or not self.idle, self.sleeptime * 10, self.sleeptime):
                        self.ticks += 1
                        continue
                modlst = []
                Utils.wait_for(lambda : not self.active or self.getModified(modlst), self.sleeptime)
                if not self.active:
                    break
                for filename in modlst:
                    self.getFailures(filename)
                self.ticks += 1
                if self.ticks % 10 == 0:
                    self.performSvc()
            except Exception as e:
                if not self.active:
                    break
                logSys.error('Caught unhandled exception in main cycle: %r', e, exc_info=logSys.getEffectiveLevel() <= logging.DEBUG)
                self.commonError('unhandled', e)
        logSys.debug('[%s] filter terminated', self.jailName)
        return True

    def isModified(self, filename):
        if False:
            i = 10
            return i + 15
        try:
            logStats = os.stat(filename)
            stats = (logStats.st_mtime, logStats.st_ino, logStats.st_size)
            pstats = self.__prevStats.get(filename, (0,))
            if logSys.getEffectiveLevel() <= 4:
                dt = logStats.st_mtime - pstats[0]
                logSys.log(4, 'Checking %s for being modified. Previous/current stats: %s / %s. dt: %s', filename, pstats, stats, dt)
            self.__file404Cnt[filename] = 0
            if pstats == stats:
                return False
            logSys.debug('%s has been modified', filename)
            self.__prevStats[filename] = stats
            return True
        except Exception as e:
            if not self.getLog(filename) or self.__prevStats.get(filename) is None:
                logSys.warning('Log %r seems to be down: %s', filename, e)
                return False
            if self.__file404Cnt[filename] < 2:
                if e.errno == 2:
                    logSys.debug('Log absence detected (possibly rotation) for %s, reason: %s', filename, e)
                else:
                    logSys.error('Unable to get stat on %s because of: %s', filename, e, exc_info=logSys.getEffectiveLevel() <= logging.DEBUG)
            self.__file404Cnt[filename] += 1
            self.commonError()
            if self.__file404Cnt[filename] > 50:
                logSys.warning('Too many errors. Remove file %r from monitoring process', filename)
                self.__file404Cnt[filename] = 0
                self.delLogPath(filename)
            return False

    def getPendingPaths(self):
        if False:
            i = 10
            return i + 15
        return list(self.__file404Cnt.keys())