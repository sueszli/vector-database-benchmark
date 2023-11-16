"""Contains utility classes for debugging memory leaks."""
__all__ = ['FakeObject', '_createGarbage', 'GarbageReport', 'GarbageLogger']
from direct.directnotify.DirectNotifyGlobal import directNotify
from direct.showbase.PythonUtil import ScratchPad, Stack, AlphabetCounter
from direct.showbase.PythonUtil import itype, deeptype, fastRepr
from direct.showbase.Job import Job
from direct.showbase.JobManagerGlobal import jobMgr
from direct.showbase.MessengerGlobal import messenger
from panda3d.core import ConfigVariableBool
import gc
GarbageCycleCountAnnounceEvent = 'announceGarbageCycleDesc2num'

class FakeObject:
    pass

class FakeDelObject:

    def __del__(self):
        if False:
            for i in range(10):
                print('nop')
        pass

def _createGarbage(num=1):
    if False:
        for i in range(10):
            print('nop')
    for i in range(num):
        a = FakeObject()
        b = FakeObject()
        a.other = b
        b.other = a
        a = FakeDelObject()
        b = FakeDelObject()
        a.other = b
        b.other = a

class GarbageReport(Job):
    """Detects leaked Python objects (via gc.collect()) and reports on garbage
    items, garbage-to-garbage references, and garbage cycles.
    If you just want to dump the report to the log, use GarbageLogger."""
    notify = directNotify.newCategory('GarbageReport')

    def __init__(self, name, log=True, verbose=False, fullReport=False, findCycles=True, threaded=False, doneCallback=None, autoDestroy=False, priority=None, safeMode=False, delOnly=False, collect=True):
        if False:
            print('Hello World!')
        Job.__init__(self, name)
        self._args = ScratchPad(name=name, log=log, verbose=verbose, fullReport=fullReport, findCycles=findCycles, doneCallback=doneCallback, autoDestroy=autoDestroy, safeMode=safeMode, delOnly=delOnly, collect=collect)
        if priority is not None:
            self.setPriority(priority)
        jobMgr.add(self)
        if not threaded:
            jobMgr.finish(self)

    def run(self):
        if False:
            while True:
                i = 10
        oldFlags = gc.get_debug()
        if self._args.delOnly:
            gc.set_debug(0)
            if self._args.collect:
                gc.collect()
            garbageInstances = gc.garbage[:]
            del gc.garbage[:]
            if len(garbageInstances) > 0:
                yield None
            if self.notify.getDebug():
                self.notify.debug('garbageInstances == %s' % fastRepr(garbageInstances))
            self.numGarbageInstances = len(garbageInstances)
            self.garbageInstanceIds = set()
            for i in range(len(garbageInstances)):
                self.garbageInstanceIds.add(id(garbageInstances[i]))
                if i % 20 == 0:
                    yield None
            del garbageInstances
        else:
            self.garbageInstanceIds = set()
        gc.set_debug(gc.DEBUG_SAVEALL)
        if self._args.collect:
            gc.collect()
        self.garbage = gc.garbage[:]
        del gc.garbage[:]
        if len(self.garbage) > 0:
            yield None
        if self.notify.getDebug():
            self.notify.debug('self.garbage == %s' % fastRepr(self.garbage))
        gc.set_debug(oldFlags)
        self.numGarbage = len(self.garbage)
        if self.numGarbage > 0:
            yield None
        if self._args.verbose:
            self.notify.info('found %s garbage items' % self.numGarbage)
        self._id2index = {}
        self.referrersByReference = {}
        self.referrersByNumber = {}
        self.referentsByReference = {}
        self.referentsByNumber = {}
        self._id2garbageInfo = {}
        self.cycles = []
        self.cyclesBySyntax = []
        self.uniqueCycleSets = set()
        self.cycleIds = set()
        for i in range(self.numGarbage):
            self._id2index[id(self.garbage[i])] = i
            if i % 20 == 0:
                yield None
        if self._args.fullReport and self.numGarbage != 0:
            if self._args.verbose:
                self.notify.info('getting referrers...')
            for i in range(self.numGarbage):
                yield None
                for result in self._getReferrers(self.garbage[i]):
                    yield None
                (byNum, byRef) = result
                self.referrersByNumber[i] = byNum
                self.referrersByReference[i] = byRef
        if self.numGarbage > 0:
            if self._args.verbose:
                self.notify.info('getting referents...')
            for i in range(self.numGarbage):
                yield None
                for result in self._getReferents(self.garbage[i]):
                    yield None
                (byNum, byRef) = result
                self.referentsByNumber[i] = byNum
                self.referentsByReference[i] = byRef
        for i in range(self.numGarbage):
            if hasattr(self.garbage[i], '_garbageInfo') and callable(self.garbage[i]._garbageInfo):
                try:
                    info = self.garbage[i]._garbageInfo()
                except Exception as e:
                    info = str(e)
                self._id2garbageInfo[id(self.garbage[i])] = info
                yield None
            elif i % 20 == 0:
                yield None
        if self._args.findCycles and self.numGarbage > 0:
            if self._args.verbose:
                self.notify.info('calculating cycles...')
            for i in range(self.numGarbage):
                yield None
                for newCycles in self._getCycles(i, self.uniqueCycleSets):
                    yield None
                self.cycles.extend(newCycles)
                newCyclesBySyntax = []
                for cycle in newCycles:
                    cycleBySyntax = ''
                    objs = []
                    for index in cycle[:-1]:
                        objs.append(self.garbage[index])
                        yield None
                    numObjs = len(objs) - 1
                    objs.extend(objs)
                    numToSkip = 0
                    objAlreadyRepresented = False
                    startIndex = 0
                    endIndex = numObjs + 1
                    if type(objs[0]) is dict and hasattr(objs[-1], '__dict__'):
                        startIndex -= 1
                        endIndex -= 1
                    for index in range(startIndex, endIndex):
                        if numToSkip:
                            numToSkip -= 1
                            continue
                        obj = objs[index]
                        if hasattr(obj, '__dict__'):
                            if not objAlreadyRepresented:
                                cycleBySyntax += '%s' % obj.__class__.__name__
                            cycleBySyntax += '.'
                            numToSkip += 1
                            member = objs[index + 2]
                            for (key, value) in obj.__dict__.items():
                                if value is member:
                                    break
                                yield None
                            else:
                                key = '<unknown member name>'
                            cycleBySyntax += '%s' % key
                            objAlreadyRepresented = True
                        elif type(obj) is dict:
                            cycleBySyntax += '{'
                            val = objs[index + 1]
                            for (key, value) in obj.items():
                                if value is val:
                                    break
                                yield None
                            else:
                                key = '<unknown key>'
                            cycleBySyntax += '%s}' % fastRepr(key)
                            objAlreadyRepresented = True
                        elif type(obj) in (tuple, list):
                            brackets = {tuple: '()', list: '[]'}[type(obj)]
                            nextObj = objs[index + 1]
                            cycleBySyntax += brackets[0]
                            for index in range(len(obj)):
                                if obj[index] is nextObj:
                                    index = str(index)
                                    break
                                yield None
                            else:
                                index = '<unknown index>'
                            cycleBySyntax += '%s%s' % (index, brackets[1])
                            objAlreadyRepresented = True
                        else:
                            cycleBySyntax += '%s --> ' % itype(obj)
                            objAlreadyRepresented = False
                    newCyclesBySyntax.append(cycleBySyntax)
                    yield None
                self.cyclesBySyntax.extend(newCyclesBySyntax)
                if not self._args.fullReport:
                    for cycle in newCycles:
                        yield None
                        self.cycleIds.update(set(cycle))
        self.numCycles = len(self.cycles)
        if self._args.findCycles:
            s = ["===== GarbageReport: '%s' (%s %s) =====" % (self._args.name, self.numCycles, 'cycle' if self.numCycles == 1 else 'cycles')]
        else:
            s = ["===== GarbageReport: '%s' =====" % self._args.name]
        if self.numGarbage > 0:
            if self._args.fullReport:
                garbageIndices = range(self.numGarbage)
            else:
                garbageIndices = sorted(self.cycleIds)
            numGarbage = len(garbageIndices)
            if not self._args.fullReport:
                abbrev = '(abbreviated) '
            else:
                abbrev = ''
            s.append('===== Garbage Items %s=====' % abbrev)
            digits = 0
            n = numGarbage
            while n > 0:
                yield None
                digits += 1
                n = n // 10
            format = '%0' + '%s' % digits + 'i:%s \t%s'
            for i in range(numGarbage):
                yield None
                idx = garbageIndices[i]
                if self._args.safeMode:
                    objStr = repr(itype(self.garbage[idx]))
                else:
                    objStr = fastRepr(self.garbage[idx])
                maxLen = 5000
                if len(objStr) > maxLen:
                    snip = '<SNIP>'
                    objStr = '%s%s' % (objStr[:maxLen - len(snip)], snip)
                s.append(format % (idx, itype(self.garbage[idx]), objStr))
            s.append('===== Garbage Item Types %s=====' % abbrev)
            for i in range(numGarbage):
                yield None
                idx = garbageIndices[i]
                objStr = str(deeptype(self.garbage[idx]))
                maxLen = 5000
                if len(objStr) > maxLen:
                    snip = '<SNIP>'
                    objStr = '%s%s' % (objStr[:maxLen - len(snip)], snip)
                s.append(format % (idx, itype(self.garbage[idx]), objStr))
            if self._args.findCycles:
                s.append('===== Garbage Cycles (Garbage Item Numbers) =====')
                ac = AlphabetCounter()
                for i in range(self.numCycles):
                    yield None
                    s.append('%s:%s' % (ac.next(), self.cycles[i]))
            if self._args.findCycles:
                s.append('===== Garbage Cycles (Python Syntax) =====')
                ac = AlphabetCounter()
                for i in range(len(self.cyclesBySyntax)):
                    yield None
                    s.append('%s:%s' % (ac.next(), self.cyclesBySyntax[i]))
            if len(self._id2garbageInfo) > 0:
                s.append('===== Garbage Custom Info =====')
                ac = AlphabetCounter()
                for i in range(len(self.cyclesBySyntax)):
                    yield None
                    counter = ac.next()
                    _id = id(self.garbage[i])
                    if _id in self._id2garbageInfo:
                        s.append('%s:%s' % (counter, self._id2garbageInfo[_id]))
            if self._args.fullReport:
                format = '%0' + '%s' % digits + 'i:%s'
                s.append('===== Referrers By Number (what is referring to garbage item?) =====')
                for i in range(numGarbage):
                    yield None
                    s.append(format % (i, self.referrersByNumber[i]))
                s.append('===== Referents By Number (what is garbage item referring to?) =====')
                for i in range(numGarbage):
                    yield None
                    s.append(format % (i, self.referentsByNumber[i]))
                s.append('===== Referrers (what is referring to garbage item?) =====')
                for i in range(numGarbage):
                    yield None
                    s.append(format % (i, self.referrersByReference[i]))
                s.append('===== Referents (what is garbage item referring to?) =====')
                for i in range(numGarbage):
                    yield None
                    s.append(format % (i, self.referentsByReference[i]))
        self._report = s
        if self._args.log:
            self.printingBegin()
            for i in range(len(self._report)):
                if self.numGarbage > 0:
                    yield None
                self.notify.info(self._report[i])
            self.notify.info('===== Garbage Report Done =====')
            self.printingEnd()
        yield Job.Done

    def finished(self):
        if False:
            while True:
                i = 10
        if self._args.doneCallback:
            self._args.doneCallback(self)
        if self._args.autoDestroy:
            self.destroy()

    def destroy(self):
        if False:
            return 10
        del self._args
        del self.garbage
        del self.referrersByReference
        del self.referrersByNumber
        del self.referentsByReference
        del self.referentsByNumber
        if hasattr(self, 'cycles'):
            del self.cycles
        del self._report
        if hasattr(self, '_reportStr'):
            del self._reportStr
        Job.destroy(self)

    def getNumCycles(self):
        if False:
            while True:
                i = 10
        return self.numCycles

    def getDesc2numDict(self):
        if False:
            while True:
                i = 10
        desc2num = {}
        for cycleBySyntax in self.cyclesBySyntax:
            desc2num.setdefault(cycleBySyntax, 0)
            desc2num[cycleBySyntax] += 1
        return desc2num

    def getGarbage(self):
        if False:
            for i in range(10):
                print('nop')
        return self.garbage

    def getReport(self):
        if False:
            print('Hello World!')
        if not hasattr(self, '_reportStr'):
            self._reportStr = ''
            for str in self._report:
                self._reportStr += '\n' + str
        return self._reportStr

    def _getReferrers(self, obj):
        if False:
            for i in range(10):
                print('nop')
        yield None
        byRef = gc.get_referrers(obj)
        yield None
        byNum = []
        for i in range(len(byRef)):
            if i % 20 == 0:
                yield None
            referrer = byRef[i]
            num = self._id2index.get(id(referrer), None)
            byNum.append(num)
        yield (byNum, byRef)

    def _getReferents(self, obj):
        if False:
            i = 10
            return i + 15
        yield None
        byRef = gc.get_referents(obj)
        yield None
        byNum = []
        for i in range(len(byRef)):
            if i % 20 == 0:
                yield None
            referent = byRef[i]
            num = self._id2index.get(id(referent), None)
            byNum.append(num)
        yield (byNum, byRef)

    def _getNormalizedCycle(self, cycle):
        if False:
            return 10
        if len(cycle) == 0:
            return cycle
        min = 1 << 30
        minIndex = None
        for i in range(len(cycle)):
            elem = cycle[i]
            if elem < min:
                min = elem
                minIndex = i
        return cycle[minIndex:] + cycle[:minIndex]

    def _getCycles(self, index, uniqueCycleSets=None):
        if False:
            for i in range(10):
                print('nop')
        assert self.notify.debugCall()
        cycles = []
        if uniqueCycleSets is None:
            uniqueCycleSets = set()
        stateStack = Stack()
        rootId = index
        objId = id(self.garbage[rootId])
        numDelInstances = int(objId in self.garbageInstanceIds)
        stateStack.push(([rootId], rootId, numDelInstances, 0))
        while True:
            yield None
            if len(stateStack) == 0:
                break
            (candidateCycle, curId, numDelInstances, resumeIndex) = stateStack.pop()
            if self.notify.getDebug():
                if self._args.delOnly:
                    print('restart: %s root=%s cur=%s numDelInstances=%s resume=%s' % (candidateCycle, rootId, curId, numDelInstances, resumeIndex))
                else:
                    print('restart: %s root=%s cur=%s resume=%s' % (candidateCycle, rootId, curId, resumeIndex))
            for index in range(resumeIndex, len(self.referentsByNumber[curId])):
                yield None
                refId = self.referentsByNumber[curId][index]
                if self.notify.getDebug():
                    print('       : %s -> %s' % (curId, refId))
                if refId == rootId:
                    normCandidateCycle = self._getNormalizedCycle(candidateCycle)
                    normCandidateCycleTuple = tuple(normCandidateCycle)
                    if not normCandidateCycleTuple in uniqueCycleSets:
                        if not self._args.delOnly or numDelInstances >= 1:
                            if self.notify.getDebug():
                                print('  FOUND: ', normCandidateCycle + [normCandidateCycle[0]])
                            cycles.append(normCandidateCycle + [normCandidateCycle[0]])
                            uniqueCycleSets.add(normCandidateCycleTuple)
                elif refId in candidateCycle:
                    pass
                elif refId is not None:
                    objId = id(self.garbage[refId])
                    numDelInstances += int(objId in self.garbageInstanceIds)
                    stateStack.push((list(candidateCycle), curId, numDelInstances, index + 1))
                    stateStack.push((list(candidateCycle) + [refId], refId, numDelInstances, 0))
                    break
        yield cycles

class GarbageLogger(GarbageReport):
    """If you just want to log the current garbage to the log file, make
    one of these. It automatically destroys itself after logging"""

    def __init__(self, name, *args, **kArgs):
        if False:
            print('Hello World!')
        kArgs['log'] = True
        kArgs['autoDestroy'] = True
        GarbageReport.__init__(self, name, *args, **kArgs)

class _CFGLGlobals:
    LastNumGarbage = 0
    LastNumCycles = 0

def checkForGarbageLeaks():
    if False:
        print('Hello World!')
    gc.collect()
    numGarbage = len(gc.garbage)
    if numGarbage > 0 and ConfigVariableBool('auto-garbage-logging', False):
        if numGarbage != _CFGLGlobals.LastNumGarbage:
            print('')
            gr = GarbageReport('found garbage', threaded=False, collect=False)
            print('')
            _CFGLGlobals.LastNumGarbage = numGarbage
            _CFGLGlobals.LastNumCycles = gr.getNumCycles()
            messenger.send(GarbageCycleCountAnnounceEvent, [gr.getDesc2numDict()])
            gr.destroy()
        notify = directNotify.newCategory('GarbageDetect')
        if ConfigVariableBool('allow-garbage-cycles', True):
            func = notify.warning
        else:
            func = notify.error
        func('%s garbage cycles found, see info above' % _CFGLGlobals.LastNumCycles)
    return numGarbage

def b_checkForGarbageLeaks(wantReply=False):
    if False:
        print('Hello World!')
    if not __dev__:
        return 0
    try:
        base.cr.timeManager
    except Exception:
        pass
    else:
        if base.cr.timeManager:
            base.cr.timeManager.d_checkForGarbageLeaks(wantReply=wantReply)
    return checkForGarbageLeaks()