from panda3d.core import TrueClock
from direct.directnotify.DirectNotifyGlobal import directNotify
from direct.showbase.PythonUtil import StdoutCapture, _installProfileCustomFuncs, _removeProfileCustomFuncs, _getProfileResultFileInfo, _setProfileResultsFileInfo, Default
import profile
import pstats
import builtins

class PercentStats(pstats.Stats):

    def setTotalTime(self, tt):
        if False:
            return 10
        self._totalTime = tt

    def add(self, *args, **kArgs):
        if False:
            while True:
                i = 10
        pstats.Stats.add(self, *args, **kArgs)
        self.files = []

    def print_stats(self, *amount):
        if False:
            while True:
                i = 10
        for filename in self.files:
            print(filename)
        if self.files:
            print()
        indent = ' ' * 8
        for func in self.top_level:
            print(indent, pstats.func_get_function_name(func))
        print(indent, self.total_calls, 'function calls', end=' ')
        if self.total_calls != self.prim_calls:
            print('(%d primitive calls)' % self.prim_calls, end=' ')
        print('in %s CPU milliseconds' % (self.total_tt * 1000.0))
        if self._totalTime != self.total_tt:
            print(indent, 'percentages are of %s CPU milliseconds' % (self._totalTime * 1000))
        print()
        (width, list) = self.get_print_list(amount)
        if list:
            self.print_title()
            for func in list:
                self.print_line(func)
            print()
        return self

    def f8(self, x):
        if False:
            return 10
        if self._totalTime == 0.0:
            return '    Inf%'
        return '%7.2f%%' % (x * 100.0 / self._totalTime)

    @staticmethod
    def func_std_string(func_name):
        if False:
            for i in range(10):
                print('nop')
        return '%s:%d(%s)' % func_name

    def print_line(self, func):
        if False:
            print('Hello World!')
        (cc, nc, tt, ct, callers) = self.stats[func]
        c = str(nc)
        f8 = self.f8
        if nc != cc:
            c = c + '/' + str(cc)
        print(c.rjust(9), end=' ')
        print(f8(tt), end=' ')
        if nc == 0:
            print(' ' * 8, end=' ')
        else:
            print(f8(tt / nc), end=' ')
        print(f8(ct), end=' ')
        if cc == 0:
            print(' ' * 8, end=' ')
        else:
            print(f8(ct / cc), end=' ')
        print(PercentStats.func_std_string(func))

class ProfileSession:
    TrueClock = TrueClock.getGlobalPtr()
    notify = directNotify.newCategory('ProfileSession')

    def __init__(self, name, func=None, logAfterProfile=False):
        if False:
            print('Hello World!')
        self._func = func
        self._name = name
        self._logAfterProfile = logAfterProfile
        self._filenameBase = 'profileData-%s-%s' % (self._name, id(self))
        self._refCount = 0
        self._aggregate = False
        self._lines = 500
        self._sorts = ['cumulative', 'time', 'calls']
        self._callInfo = True
        self._totalTime = None
        self._reset()
        self.acquire()

    def getReference(self):
        if False:
            for i in range(10):
                print('nop')
        self.acquire()
        return self

    def acquire(self):
        if False:
            print('Hello World!')
        self._refCount += 1

    def release(self):
        if False:
            for i in range(10):
                print('nop')
        self._refCount -= 1
        if not self._refCount:
            self._destroy()

    def _destroy(self):
        if False:
            i = 10
            return i + 15
        del self._func
        del self._name
        del self._filenameBase
        del self._filenameCounter
        del self._filenames
        del self._duration
        del self._filename2ramFile
        del self._resultCache
        del self._successfulProfiles

    def _reset(self):
        if False:
            for i in range(10):
                print('nop')
        self._filenameCounter = 0
        self._filenames = []
        self._statFileCounter = 0
        self._successfulProfiles = 0
        self._duration = None
        self._filename2ramFile = {}
        self._stats = None
        self._resultCache = {}

    def _getNextFilename(self):
        if False:
            i = 10
            return i + 15
        filename = '%s-%s' % (self._filenameBase, self._filenameCounter)
        self._filenameCounter += 1
        return filename

    def run(self):
        if False:
            while True:
                i = 10
        self.acquire()
        if not self._aggregate:
            self._reset()
        if 'globalProfileSessionFunc' in builtins.__dict__:
            self.notify.warning('could not profile %s' % self._func)
            result = self._func()
            if self._duration is None:
                self._duration = 0.0
        else:
            assert hasattr(self._func, '__call__')
            builtins.globalProfileSessionFunc = self._func
            builtins.globalProfileSessionResult = [None]
            self._filenames.append(self._getNextFilename())
            filename = self._filenames[-1]
            _installProfileCustomFuncs(filename)
            Profile = profile.Profile
            statement = 'globalProfileSessionResult[0]=globalProfileSessionFunc()'
            sort = -1
            retVal = None
            prof = Profile()
            try:
                prof = prof.run(statement)
            except SystemExit:
                pass
            prof.dump_stats(filename)
            del prof.dispatcher
            profData = _getProfileResultFileInfo(filename)
            self._filename2ramFile[filename] = profData
            maxTime = 0.0
            for (cc, nc, tt, ct, callers) in profData[1].values():
                if ct > maxTime:
                    maxTime = ct
            self._duration = maxTime
            _removeProfileCustomFuncs(filename)
            result = builtins.globalProfileSessionResult[0]
            del builtins.globalProfileSessionFunc
            del builtins.globalProfileSessionResult
            self._successfulProfiles += 1
            if self._logAfterProfile:
                self.notify.info(self.getResults())
        self.release()
        return result

    def getDuration(self):
        if False:
            return 10
        return self._duration

    def profileSucceeded(self):
        if False:
            print('Hello World!')
        return self._successfulProfiles > 0

    def _restoreRamFile(self, filename):
        if False:
            i = 10
            return i + 15
        _installProfileCustomFuncs(filename)
        _setProfileResultsFileInfo(filename, self._filename2ramFile[filename])

    def _discardRamFile(self, filename):
        if False:
            for i in range(10):
                print('nop')
        _removeProfileCustomFuncs(filename)
        del self._filename2ramFile[filename]

    def setName(self, name):
        if False:
            i = 10
            return i + 15
        self._name = name

    def getName(self):
        if False:
            i = 10
            return i + 15
        return self._name

    def setFunc(self, func):
        if False:
            i = 10
            return i + 15
        self._func = func

    def getFunc(self):
        if False:
            return 10
        return self._func

    def setAggregate(self, aggregate):
        if False:
            return 10
        self._aggregate = aggregate

    def getAggregate(self):
        if False:
            for i in range(10):
                print('nop')
        return self._aggregate

    def setLogAfterProfile(self, logAfterProfile):
        if False:
            while True:
                i = 10
        self._logAfterProfile = logAfterProfile

    def getLogAfterProfile(self):
        if False:
            i = 10
            return i + 15
        return self._logAfterProfile

    def setLines(self, lines):
        if False:
            return 10
        self._lines = lines

    def getLines(self):
        if False:
            print('Hello World!')
        return self._lines

    def setSorts(self, sorts):
        if False:
            while True:
                i = 10
        self._sorts = sorts

    def getSorts(self):
        if False:
            i = 10
            return i + 15
        return self._sorts

    def setShowCallInfo(self, showCallInfo):
        if False:
            return 10
        self._showCallInfo = showCallInfo

    def getShowCallInfo(self):
        if False:
            i = 10
            return i + 15
        return self._showCallInfo

    def setTotalTime(self, totalTime=None):
        if False:
            while True:
                i = 10
        self._totalTime = totalTime

    def resetTotalTime(self):
        if False:
            while True:
                i = 10
        self._totalTime = None

    def getTotalTime(self):
        if False:
            for i in range(10):
                print('nop')
        return self._totalTime

    def aggregate(self, other):
        if False:
            i = 10
            return i + 15
        other._compileStats()
        self._compileStats()
        self._stats.add(other._stats)

    def _compileStats(self):
        if False:
            while True:
                i = 10
        statsChanged = self._statFileCounter < len(self._filenames)
        if self._stats is None:
            for filename in self._filenames:
                self._restoreRamFile(filename)
            self._stats = PercentStats(*self._filenames)
            self._statFileCounter = len(self._filenames)
            for filename in self._filenames:
                self._discardRamFile(filename)
        else:
            while self._statFileCounter < len(self._filenames):
                filename = self._filenames[self._statFileCounter]
                self._restoreRamFile(filename)
                self._stats.add(filename)
                self._discardRamFile(filename)
        if statsChanged:
            self._stats.strip_dirs()
            self._resultCache = {}
        return statsChanged

    def getResults(self, lines=Default, sorts=Default, callInfo=Default, totalTime=Default):
        if False:
            return 10
        if not self.profileSucceeded():
            output = '%s: profiler already running, could not profile' % self._name
        else:
            if lines is Default:
                lines = self._lines
            if sorts is Default:
                sorts = self._sorts
            if callInfo is Default:
                callInfo = self._callInfo
            if totalTime is Default:
                totalTime = self._totalTime
            self._compileStats()
            if totalTime is None:
                totalTime = self._stats.total_tt
            lines = int(lines)
            sorts = list(sorts)
            callInfo = bool(callInfo)
            totalTime = float(totalTime)
            k = str((lines, sorts, callInfo, totalTime))
            if k in self._resultCache:
                output = self._resultCache[k]
            else:
                sc = StdoutCapture()
                s = self._stats
                s.setTotalTime(totalTime)
                for sort in sorts:
                    s.sort_stats(sort)
                    s.print_stats(lines)
                    if callInfo:
                        s.print_callees(lines)
                        s.print_callers(lines)
                output = sc.getString()
                sc.destroy()
                self._resultCache[k] = output
        return output