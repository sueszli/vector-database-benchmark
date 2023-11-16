"""Contains objects that report different types of leaks to the
ContainerLeakDetector.
"""
from panda3d.core import ConfigVariableBool, MemoryUsage
from direct.showbase.DirectObject import DirectObject
from direct.showbase.PythonUtil import safeTypeName, typeName, uniqueName, serialNum
from direct.showbase.Job import Job
from direct.showbase.JobManagerGlobal import jobMgr
from direct.showbase.MessengerGlobal import messenger
from direct.task.TaskManagerGlobal import taskMgr
import gc
import builtins

class LeakDetector:

    def __init__(self):
        if False:
            while True:
                i = 10
        if not hasattr(builtins, 'leakDetectors'):
            builtins.leakDetectors = {}
        self._leakDetectorsKey = self.getLeakDetectorKey()
        if __dev__:
            assert self._leakDetectorsKey not in builtins.leakDetectors
        builtins.leakDetectors[self._leakDetectorsKey] = self

    def destroy(self):
        if False:
            while True:
                i = 10
        del builtins.leakDetectors[self._leakDetectorsKey]

    def getLeakDetectorKey(self):
        if False:
            i = 10
            return i + 15
        return '%s-%s' % (self.__class__.__name__, id(self))

class ObjectTypeLeakDetector(LeakDetector):

    def __init__(self, otld, objType, generation):
        if False:
            print('Hello World!')
        self._otld = otld
        self._objType = objType
        self._generation = generation
        LeakDetector.__init__(self)

    def destroy(self):
        if False:
            return 10
        self._otld = None
        LeakDetector.destroy(self)

    def getLeakDetectorKey(self):
        if False:
            return 10
        return '%s-%s' % (self._objType, self.__class__.__name__)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        num = self._otld._getNumObjsOfType(self._objType, self._generation)
        self._generation = self._otld._getGeneration()
        return num

class ObjectTypesLeakDetector(LeakDetector):

    def __init__(self):
        if False:
            return 10
        LeakDetector.__init__(self)
        self._type2ld = {}
        self._type2count = {}
        self._generation = 0
        self._thisLdGen = 0

    def destroy(self):
        if False:
            print('Hello World!')
        for ld in self._type2ld.values():
            ld.destroy()
        LeakDetector.destroy(self)

    def _recalc(self):
        if False:
            for i in range(10):
                print('nop')
        objs = gc.get_objects()
        self._type2count = {}
        for obj in objs:
            objType = safeTypeName(obj)
            if objType not in self._type2ld:
                self._type2ld[objType] = ObjectTypeLeakDetector(self, objType, self._generation)
            self._type2count.setdefault(objType, 0)
            self._type2count[objType] += 1
        self._generation += 1

    def _getGeneration(self):
        if False:
            i = 10
            return i + 15
        return self._generation

    def _getNumObjsOfType(self, objType, otherGen):
        if False:
            for i in range(10):
                print('nop')
        if self._generation == otherGen:
            self._recalc()
        return self._type2count.get(objType, 0)

    def __len__(self):
        if False:
            while True:
                i = 10
        if self._generation == self._thisLdGen:
            self._recalc()
        self._thisLdGen = self._generation
        return len(self._type2count)

class GarbageLeakDetector(LeakDetector):

    def __len__(self):
        if False:
            i = 10
            return i + 15
        oldFlags = gc.get_debug()
        gc.set_debug(0)
        gc.collect()
        numGarbage = len(gc.garbage)
        del gc.garbage[:]
        gc.set_debug(oldFlags)
        return numGarbage

class SceneGraphLeakDetector(LeakDetector):

    def __init__(self, render):
        if False:
            print('Hello World!')
        LeakDetector.__init__(self)
        self._render = render
        if ConfigVariableBool('leak-scene-graph', False):
            self._leakTaskName = 'leakNodes-%s' % serialNum()
            self._leakNode()

    def destroy(self):
        if False:
            return 10
        if hasattr(self, '_leakTaskName'):
            taskMgr.remove(self._leakTaskName)
        del self._render
        LeakDetector.destroy(self)

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return self._render.countNumDescendants()

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'SceneGraphLeakDetector(%s)' % self._render

    def _leakNode(self, task=None):
        if False:
            for i in range(10):
                print('nop')
        self._render.attachNewNode('leakNode-%s' % serialNum())
        taskMgr.doMethodLater(10, self._leakNode, self._leakTaskName)

class CppMemoryUsage(LeakDetector):

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return MemoryUsage.getCurrentCppSize()

class TaskLeakDetectorBase:

    def _getTaskNamePattern(self, taskName):
        if False:
            print('Hello World!')
        for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9):
            taskName = taskName.replace('%s' % i, '')
        return taskName

class _TaskNamePatternLeakDetector(LeakDetector, TaskLeakDetectorBase):

    def __init__(self, taskNamePattern):
        if False:
            print('Hello World!')
        self._taskNamePattern = taskNamePattern
        LeakDetector.__init__(self)

    def __len__(self):
        if False:
            i = 10
            return i + 15
        numTasks = 0
        for task in taskMgr.getTasks():
            if self._getTaskNamePattern(task.name) == self._taskNamePattern:
                numTasks += 1
        for task in taskMgr.getDoLaters():
            if self._getTaskNamePattern(task.name) == self._taskNamePattern:
                numTasks += 1
        return numTasks

    def getLeakDetectorKey(self):
        if False:
            i = 10
            return i + 15
        return '%s-%s' % (self._taskNamePattern, self.__class__.__name__)

class TaskLeakDetector(LeakDetector, TaskLeakDetectorBase):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        LeakDetector.__init__(self)
        self._taskName2collector = {}

    def destroy(self):
        if False:
            for i in range(10):
                print('nop')
        for (taskName, collector) in self._taskName2collector.items():
            collector.destroy()
        del self._taskName2collector
        LeakDetector.destroy(self)

    def _processTaskName(self, taskName):
        if False:
            return 10
        namePattern = self._getTaskNamePattern(taskName)
        if namePattern not in self._taskName2collector:
            self._taskName2collector[namePattern] = _TaskNamePatternLeakDetector(namePattern)

    def __len__(self):
        if False:
            while True:
                i = 10
        self._taskName2collector = {}
        for task in taskMgr.getTasks():
            self._processTaskName(task.name)
        for task in taskMgr.getDoLaters():
            self._processTaskName(task.name)
        return len(self._taskName2collector)

class MessageLeakDetectorBase:

    def _getMessageNamePattern(self, msgName):
        if False:
            while True:
                i = 10
        for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9):
            msgName = msgName.replace('%s' % i, '')
        return msgName

class _MessageTypeLeakDetector(LeakDetector, MessageLeakDetectorBase):

    def __init__(self, msgNamePattern):
        if False:
            while True:
                i = 10
        self._msgNamePattern = msgNamePattern
        self._msgNames = set()
        LeakDetector.__init__(self)

    def addMsgName(self, msgName):
        if False:
            i = 10
            return i + 15
        self._msgNames.add(msgName)

    def __len__(self):
        if False:
            i = 10
            return i + 15
        toRemove = set()
        num = 0
        for msgName in self._msgNames:
            n = messenger._getNumListeners(msgName)
            if n == 0:
                toRemove.add(msgName)
            else:
                num += n
        self._msgNames.difference_update(toRemove)
        return num

    def getLeakDetectorKey(self):
        if False:
            print('Hello World!')
        return '%s-%s' % (self._msgNamePattern, self.__class__.__name__)

class _MessageTypeLeakDetectorCreator(Job):

    def __init__(self, creator):
        if False:
            while True:
                i = 10
        Job.__init__(self, uniqueName(typeName(self)))
        self._creator = creator

    def destroy(self):
        if False:
            for i in range(10):
                print('nop')
        self._creator = None
        Job.destroy(self)

    def finished(self):
        if False:
            print('Hello World!')
        Job.finished(self)

    def run(self):
        if False:
            while True:
                i = 10
        for msgName in messenger._getEvents():
            yield None
            namePattern = self._creator._getMessageNamePattern(msgName)
            if namePattern not in self._creator._msgName2detector:
                self._creator._msgName2detector[namePattern] = _MessageTypeLeakDetector(namePattern)
            self._creator._msgName2detector[namePattern].addMsgName(msgName)
        yield Job.Done

class MessageTypesLeakDetector(LeakDetector, MessageLeakDetectorBase):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        LeakDetector.__init__(self)
        self._msgName2detector = {}
        self._createJob = None
        if ConfigVariableBool('leak-message-types', False):
            self._leakers = []
            self._leakTaskName = uniqueName('leak-message-types')
            taskMgr.add(self._leak, self._leakTaskName)

    def _leak(self, task):
        if False:
            return 10
        self._leakers.append(DirectObject())
        self._leakers[-1].accept('leak-msg', self._leak)
        return task.cont

    def destroy(self):
        if False:
            i = 10
            return i + 15
        if hasattr(self, '_leakTaskName'):
            taskMgr.remove(self._leakTaskName)
            for leaker in self._leakers:
                leaker.ignoreAll()
            self._leakers = None
        if self._createJob:
            self._createJob.destroy()
        self._createJob = None
        for (msgName, detector) in self._msgName2detector.items():
            detector.destroy()
        del self._msgName2detector
        LeakDetector.destroy(self)

    def __len__(self):
        if False:
            return 10
        if self._createJob:
            if self._createJob.isFinished():
                self._createJob.destroy()
                self._createJob = None
        self._createJob = _MessageTypeLeakDetectorCreator(self)
        jobMgr.add(self._createJob)
        return len(self._msgName2detector)

class _MessageListenerTypeLeakDetector(LeakDetector):

    def __init__(self, typeName):
        if False:
            for i in range(10):
                print('nop')
        self._typeName = typeName
        LeakDetector.__init__(self)

    def __len__(self):
        if False:
            print('Hello World!')
        numObjs = 0
        for obj in messenger._getObjects():
            if typeName(obj) == self._typeName:
                numObjs += 1
        return numObjs

    def getLeakDetectorKey(self):
        if False:
            print('Hello World!')
        return '%s-%s' % (self._typeName, self.__class__.__name__)

class _MessageListenerTypeLeakDetectorCreator(Job):

    def __init__(self, creator):
        if False:
            for i in range(10):
                print('nop')
        Job.__init__(self, uniqueName(typeName(self)))
        self._creator = creator

    def destroy(self):
        if False:
            for i in range(10):
                print('nop')
        self._creator = None
        Job.destroy(self)

    def finished(self):
        if False:
            print('Hello World!')
        Job.finished(self)

    def run(self):
        if False:
            i = 10
            return i + 15
        for obj in messenger._getObjects():
            yield None
            tName = typeName(obj)
            if tName not in self._creator._typeName2detector:
                self._creator._typeName2detector[tName] = _MessageListenerTypeLeakDetector(tName)
        yield Job.Done

class MessageListenerTypesLeakDetector(LeakDetector):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        LeakDetector.__init__(self)
        self._typeName2detector = {}
        self._createJob = None
        if ConfigVariableBool('leak-message-listeners', False):
            self._leakers = []
            self._leakTaskName = uniqueName('leak-message-listeners')
            taskMgr.add(self._leak, self._leakTaskName)

    def _leak(self, task):
        if False:
            for i in range(10):
                print('nop')
        self._leakers.append(DirectObject())
        self._leakers[-1].accept(uniqueName('leak-msg-listeners'), self._leak)
        return task.cont

    def destroy(self):
        if False:
            while True:
                i = 10
        if hasattr(self, '_leakTaskName'):
            taskMgr.remove(self._leakTaskName)
            for leaker in self._leakers:
                leaker.ignoreAll()
            self._leakers = None
        if self._createJob:
            self._createJob.destroy()
        self._createJob = None
        for (typeName, detector) in self._typeName2detector.items():
            detector.destroy()
        del self._typeName2detector
        LeakDetector.destroy(self)

    def __len__(self):
        if False:
            i = 10
            return i + 15
        if self._createJob:
            if self._createJob.isFinished():
                self._createJob.destroy()
                self._createJob = None
        self._createJob = _MessageListenerTypeLeakDetectorCreator(self)
        jobMgr.add(self._createJob)
        return len(self._typeName2detector)