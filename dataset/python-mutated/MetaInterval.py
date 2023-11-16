"""
This module defines the various "meta intervals", which execute other
intervals either in parallel or in a specified sequential order.
"""
__all__ = ['MetaInterval', 'Sequence', 'Parallel', 'ParallelEndTogether', 'Track']
from panda3d.core import PStatCollector, ostream
from panda3d.direct import CInterval, CMetaInterval
from direct.directnotify.DirectNotifyGlobal import directNotify
from .IntervalManager import ivalMgr
from . import Interval
from direct.task.Task import TaskManager
PREVIOUS_END = CMetaInterval.RSPreviousEnd
PREVIOUS_START = CMetaInterval.RSPreviousBegin
TRACK_START = CMetaInterval.RSLevelBegin

class MetaInterval(CMetaInterval):
    notify = directNotify.newCategory('MetaInterval')
    SequenceNum = 1

    def __init__(self, *ivals, **kw):
        if False:
            print('Hello World!')
        name = None
        if 'name' in kw:
            name = kw['name']
            del kw['name']
        autoPause = 0
        autoFinish = 0
        if 'autoPause' in kw:
            autoPause = kw['autoPause']
            del kw['autoPause']
        if 'autoFinish' in kw:
            autoFinish = kw['autoFinish']
            del kw['autoFinish']
        self.phonyDuration = -1
        if 'duration' in kw:
            self.phonyDuration = kw['duration']
            del kw['duration']
        if kw:
            self.notify.error('Unexpected keyword parameters: %s' % list(kw.keys()))
        self.ivals = ivals
        self.__ivalsDirty = 1
        if name is None:
            name = self.__class__.__name__ + '-%d'
        if '%' in name:
            name = name % self.SequenceNum
            MetaInterval.SequenceNum += 1
        CMetaInterval.__init__(self, name)
        self.__manager = ivalMgr
        self.setAutoPause(autoPause)
        self.setAutoFinish(autoFinish)
        self.pstats = None
        if __debug__ and TaskManager.taskTimerVerbose:
            self.pname = name.split('-', 1)[0]
            self.pstats = PStatCollector('App:Tasks:ivalLoop:%s' % self.pname)
        self.pythonIvals = []
        assert self.validateComponents(self.ivals)

    def append(self, ival):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(self.ivals, tuple):
            self.ivals = list(self.ivals)
        self.ivals.append(ival)
        self.__ivalsDirty = 1
        assert self.validateComponent(ival)

    def extend(self, ivals):
        if False:
            print('Hello World!')
        self += ivals

    def count(self, ival):
        if False:
            while True:
                i = 10
        return self.ivals.count(ival)

    def index(self, ival):
        if False:
            for i in range(10):
                print('nop')
        return self.ivals.index(ival)

    def insert(self, index, ival):
        if False:
            print('Hello World!')
        if isinstance(self.ivals, tuple):
            self.ivals = list(self.ivals)
        self.ivals.insert(index, ival)
        self.__ivalsDirty = 1
        assert self.validateComponent(ival)

    def pop(self, index=None):
        if False:
            return 10
        if isinstance(self.ivals, tuple):
            self.ivals = list(self.ivals)
        self.__ivalsDirty = 1
        if index is None:
            return self.ivals.pop()
        else:
            return self.ivals.pop(index)

    def remove(self, ival):
        if False:
            return 10
        if isinstance(self.ivals, tuple):
            self.ivals = list(self.ivals)
        self.ivals.remove(ival)
        self.__ivalsDirty = 1

    def reverse(self):
        if False:
            return 10
        if isinstance(self.ivals, tuple):
            self.ivals = list(self.ivals)
        self.ivals.reverse()
        self.__ivalsDirty = 1

    def sort(self, cmpfunc=None):
        if False:
            while True:
                i = 10
        if isinstance(self.ivals, tuple):
            self.ivals = list(self.ivals)
        self.__ivalsDirty = 1
        if cmpfunc is None:
            self.ivals.sort()
        else:
            self.ivals.sort(cmpfunc)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.ivals)

    def __getitem__(self, index):
        if False:
            while True:
                i = 10
        return self.ivals[index]

    def __setitem__(self, index, value):
        if False:
            print('Hello World!')
        if isinstance(self.ivals, tuple):
            self.ivals = list(self.ivals)
        self.ivals[index] = value
        self.__ivalsDirty = 1
        assert self.validateComponent(value)

    def __delitem__(self, index):
        if False:
            print('Hello World!')
        if isinstance(self.ivals, tuple):
            self.ivals = list(self.ivals)
        del self.ivals[index]
        self.__ivalsDirty = 1

    def __getslice__(self, i, j):
        if False:
            return 10
        if isinstance(self.ivals, tuple):
            self.ivals = list(self.ivals)
        return self.__class__(self.ivals[i:j])

    def __setslice__(self, i, j, s):
        if False:
            return 10
        if isinstance(self.ivals, tuple):
            self.ivals = list(self.ivals)
        self.ivals[i:j] = s
        self.__ivalsDirty = 1
        assert self.validateComponents(s)

    def __delslice__(self, i, j):
        if False:
            return 10
        if isinstance(self.ivals, tuple):
            self.ivals = list(self.ivals)
        del self.ivals[i:j]
        self.__ivalsDirty = 1

    def __iadd__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(self.ivals, tuple):
            self.ivals = list(self.ivals)
        if isinstance(other, MetaInterval):
            assert self.__class__ == other.__class__
            ivals = other.ivals
        else:
            ivals = list(other)
        self.ivals += ivals
        self.__ivalsDirty = 1
        assert self.validateComponents(ivals)
        return self

    def __add__(self, other):
        if False:
            print('Hello World!')
        copy = self[:]
        copy += other
        return copy

    def addSequence(self, list, name, relTime, relTo, duration):
        if False:
            print('Hello World!')
        self.pushLevel(name, relTime, relTo)
        for ival in list:
            self.addInterval(ival, 0.0, PREVIOUS_END)
        self.popLevel(duration)

    def addParallel(self, list, name, relTime, relTo, duration):
        if False:
            i = 10
            return i + 15
        self.pushLevel(name, relTime, relTo)
        for ival in list:
            self.addInterval(ival, 0.0, TRACK_START)
        self.popLevel(duration)

    def addParallelEndTogether(self, list, name, relTime, relTo, duration):
        if False:
            i = 10
            return i + 15
        maxDuration = 0
        for ival in list:
            maxDuration = max(maxDuration, ival.getDuration())
        self.pushLevel(name, relTime, relTo)
        for ival in list:
            self.addInterval(ival, maxDuration - ival.getDuration(), TRACK_START)
        self.popLevel(duration)

    def addTrack(self, trackList, name, relTime, relTo, duration):
        if False:
            i = 10
            return i + 15
        self.pushLevel(name, relTime, relTo)
        for tupleObj in trackList:
            if isinstance(tupleObj, tuple) or isinstance(tupleObj, list):
                relTime = tupleObj[0]
                ival = tupleObj[1]
                if len(tupleObj) >= 3:
                    relTo = tupleObj[2]
                else:
                    relTo = TRACK_START
                self.addInterval(ival, relTime, relTo)
            else:
                self.notify.error('Not a tuple in Track: %s' % (tupleObj,))
        self.popLevel(duration)

    def addInterval(self, ival, relTime, relTo):
        if False:
            while True:
                i = 10
        if isinstance(ival, CInterval):
            if getattr(ival, 'inPython', 0):
                index = len(self.pythonIvals)
                self.pythonIvals.append(ival)
                self.addExtIndex(index, ival.getName(), ival.getDuration(), ival.getOpenEnded(), relTime, relTo)
            elif isinstance(ival, MetaInterval):
                ival.applyIvals(self, relTime, relTo)
            else:
                self.addCInterval(ival, relTime, relTo)
        elif isinstance(ival, Interval.Interval):
            index = len(self.pythonIvals)
            self.pythonIvals.append(ival)
            if self.pstats:
                ival.pstats = PStatCollector(self.pstats, ival.pname)
            self.addExtIndex(index, ival.getName(), ival.getDuration(), ival.getOpenEnded(), relTime, relTo)
        else:
            self.notify.error('Not an Interval: %s' % (ival,))

    def setManager(self, manager):
        if False:
            i = 10
            return i + 15
        self.__manager = manager
        CMetaInterval.setManager(self, manager)

    def getManager(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__manager
    manager = property(getManager, setManager)

    def setT(self, t):
        if False:
            while True:
                i = 10
        self.__updateIvals()
        CMetaInterval.setT(self, t)
    t = property(CMetaInterval.getT, setT)

    def start(self, startT=0.0, endT=-1.0, playRate=1.0):
        if False:
            print('Hello World!')
        self.__updateIvals()
        self.setupPlay(startT, endT, playRate, 0)
        self.__manager.addInterval(self)

    def loop(self, startT=0.0, endT=-1.0, playRate=1.0):
        if False:
            i = 10
            return i + 15
        self.__updateIvals()
        self.setupPlay(startT, endT, playRate, 1)
        self.__manager.addInterval(self)

    def pause(self):
        if False:
            print('Hello World!')
        if self.getState() == CInterval.SStarted:
            self.privInterrupt()
        self.__manager.removeInterval(self)
        self.privPostEvent()
        return self.getT()

    def resume(self, startT=None):
        if False:
            while True:
                i = 10
        self.__updateIvals()
        if startT is not None:
            self.setT(startT)
        self.setupResume()
        self.__manager.addInterval(self)

    def resumeUntil(self, endT):
        if False:
            print('Hello World!')
        self.__updateIvals()
        self.setupResumeUntil(endT)
        self.__manager.addInterval(self)

    def finish(self):
        if False:
            return 10
        self.__updateIvals()
        state = self.getState()
        if state == CInterval.SInitial:
            self.privInstant()
        elif state != CInterval.SFinal:
            self.privFinalize()
        self.__manager.removeInterval(self)
        self.privPostEvent()

    def clearToInitial(self):
        if False:
            while True:
                i = 10
        self.pause()
        CMetaInterval.clearToInitial(self)

    def validateComponent(self, component):
        if False:
            i = 10
            return i + 15
        return isinstance(component, CInterval) or isinstance(component, Interval.Interval)

    def validateComponents(self, components):
        if False:
            print('Hello World!')
        for component in components:
            if not self.validateComponent(component):
                return 0
        return 1

    def __updateIvals(self):
        if False:
            for i in range(10):
                print('nop')
        if self.__ivalsDirty:
            self.clearIntervals()
            self.applyIvals(self, 0, TRACK_START)
            self.__ivalsDirty = 0

    def clearIntervals(self):
        if False:
            return 10
        CMetaInterval.clearIntervals(self)
        self.inPython = 0

    def applyIvals(self, meta, relTime, relTo):
        if False:
            return 10
        if len(self.ivals) == 0:
            pass
        elif len(self.ivals) == 1:
            meta.addInterval(self.ivals[0], relTime, relTo)
        else:
            self.notify.error('Cannot build list from MetaInterval directly.')

    def setPlayRate(self, playRate):
        if False:
            for i in range(10):
                print('nop')
        ' Changes the play rate of the interval.  If the interval is\n        already started, this changes its speed on-the-fly.  Note that\n        since playRate is a parameter to start() and loop(), the next\n        call to start() or loop() will reset this parameter. '
        if self.isPlaying():
            self.pause()
            CMetaInterval.setPlayRate(self, playRate)
            self.resume()
        else:
            CMetaInterval.setPlayRate(self, playRate)
    play_rate = property(CMetaInterval.getPlayRate, setPlayRate)

    def __doPythonCallbacks(self):
        if False:
            while True:
                i = 10
        ival = None
        try:
            while self.isEventReady():
                index = self.getEventIndex()
                t = self.getEventT()
                eventType = self.getEventType()
                self.popEvent()
                ival = self.pythonIvals[index]
                ival.privDoEvent(t, eventType)
                ival.privPostEvent()
                ival = None
        except:
            if ival is not None:
                print('Exception occurred while processing %s of %s:' % (ival.getName(), self.getName()))
            else:
                print('Exception occurred while processing %s:' % self.getName())
            print(self)
            raise

    def privDoEvent(self, t, event):
        if False:
            return 10
        if self.pstats:
            self.pstats.start()
        self.__updateIvals()
        CMetaInterval.privDoEvent(self, t, event)
        if self.pstats:
            self.pstats.stop()

    def privPostEvent(self):
        if False:
            while True:
                i = 10
        if self.pstats:
            self.pstats.start()
        self.__doPythonCallbacks()
        CMetaInterval.privPostEvent(self)
        if self.pstats:
            self.pstats.stop()

    def setIntervalStartTime(self, *args, **kw):
        if False:
            while True:
                i = 10
        self.__updateIvals()
        self.inPython = 1
        return CMetaInterval.setIntervalStartTime(self, *args, **kw)

    def getIntervalStartTime(self, *args, **kw):
        if False:
            i = 10
            return i + 15
        self.__updateIvals()
        return CMetaInterval.getIntervalStartTime(self, *args, **kw)

    def getDuration(self):
        if False:
            while True:
                i = 10
        self.__updateIvals()
        return CMetaInterval.getDuration(self)
    duration = property(getDuration)

    def __repr__(self, *args, **kw):
        if False:
            return 10
        self.__updateIvals()
        return CMetaInterval.__repr__(self, *args, **kw)

    def __str__(self, *args, **kw):
        if False:
            return 10
        self.__updateIvals()
        return CMetaInterval.__str__(self, *args, **kw)

    def timeline(self, out=None):
        if False:
            i = 10
            return i + 15
        self.__updateIvals()
        if out is None:
            out = ostream
        CMetaInterval.timeline(self, out)
    add_sequence = addSequence
    add_parallel = addParallel
    add_parallel_end_together = addParallelEndTogether
    add_track = addTrack
    add_interval = addInterval
    set_manager = setManager
    get_manager = getManager
    set_t = setT
    resume_until = resumeUntil
    clear_to_initial = clearToInitial
    clear_intervals = clearIntervals
    set_play_rate = setPlayRate
    priv_do_event = privDoEvent
    priv_post_event = privPostEvent
    set_interval_start_time = setIntervalStartTime
    get_interval_start_time = getIntervalStartTime
    get_duration = getDuration

class Sequence(MetaInterval):

    def applyIvals(self, meta, relTime, relTo):
        if False:
            while True:
                i = 10
        meta.addSequence(self.ivals, self.getName(), relTime, relTo, self.phonyDuration)

class Parallel(MetaInterval):

    def applyIvals(self, meta, relTime, relTo):
        if False:
            for i in range(10):
                print('nop')
        meta.addParallel(self.ivals, self.getName(), relTime, relTo, self.phonyDuration)

class ParallelEndTogether(MetaInterval):

    def applyIvals(self, meta, relTime, relTo):
        if False:
            return 10
        meta.addParallelEndTogether(self.ivals, self.getName(), relTime, relTo, self.phonyDuration)

class Track(MetaInterval):

    def applyIvals(self, meta, relTime, relTo):
        if False:
            print('Hello World!')
        meta.addTrack(self.ivals, self.getName(), relTime, relTo, self.phonyDuration)

    def validateComponent(self, tupleObj):
        if False:
            return 10
        if not (isinstance(tupleObj, tuple) or isinstance(tupleObj, list)):
            return 0
        relTime = tupleObj[0]
        ival = tupleObj[1]
        if len(tupleObj) >= 3:
            relTo = tupleObj[2]
        else:
            relTo = TRACK_START
        if not (isinstance(relTime, float) or isinstance(relTime, int)):
            return 0
        if not MetaInterval.validateComponent(self, ival):
            return 0
        if relTo != PREVIOUS_END and relTo != PREVIOUS_START and (relTo != TRACK_START):
            return 0
        return 1