from __future__ import annotations
from direct.directnotify.DirectNotifyGlobal import directNotify
import direct.showbase.DConfig as config
from direct.showbase.PythonUtil import makeFlywheelGen, flywheel
from direct.showbase.PythonUtil import itype, serialNum, safeRepr, fastRepr
from direct.showbase.PythonUtil import getBase, uniqueName, ScratchPad, nullGen
from direct.showbase.Job import Job
from direct.showbase.JobManagerGlobal import jobMgr
from direct.showbase.MessengerGlobal import messenger
from direct.task.TaskManagerGlobal import taskMgr
import types
import weakref
import random
import builtins
deadEndTypes = frozenset((types.BuiltinFunctionType, types.BuiltinMethodType, types.CodeType, types.FunctionType, types.GeneratorType, types.CoroutineType, types.AsyncGeneratorType, bool, complex, float, int, type, bytes, str, list, tuple, type(None), type(NotImplemented)))

def _createContainerLeak():
    if False:
        return 10

    def leakContainer(task=None):
        if False:
            return 10
        base = getBase()
        if not hasattr(base, 'leakContainer'):
            base.leakContainer = {}

        class LeakKey:
            pass
        base.leakContainer[LeakKey(),] = {}
        if random.random() < 0.01:
            key = random.choice(list(base.leakContainer.keys()))
            ContainerLeakDetector.notify.debug('removing reference to leakContainer key %s so it will be garbage-collected' % safeRepr(key))
            del base.leakContainer[key]
        taskMgr.doMethodLater(10, leakContainer, 'leakContainer-%s' % serialNum())
        if task:
            return task.done
    leakContainer()

def _createTaskLeak():
    if False:
        for i in range(10):
            print('nop')
    leakTaskName = uniqueName('leakedTask')
    leakDoLaterName = uniqueName('leakedDoLater')

    def nullTask(task=None):
        if False:
            return 10
        return task.cont

    def nullDoLater(task=None):
        if False:
            i = 10
            return i + 15
        return task.done

    def leakTask(task=None, leakTaskName=leakTaskName):
        if False:
            while True:
                i = 10
        base = getBase()
        taskMgr.add(nullTask, uniqueName(leakTaskName))
        taskMgr.doMethodLater(1 << 31, nullDoLater, uniqueName(leakDoLaterName))
        taskMgr.doMethodLater(10, leakTask, 'doLeakTask-%s' % serialNum())
        if task:
            return task.done
    leakTask()

class NoDictKey:
    pass

class Indirection:
    """
    Represents the indirection that brings you from a container to an element of the container.
    Stored as a string to be used as part of an eval, or as a key to be looked up in a dict.
    Each dictionary dereference is individually eval'd since the dict key might have been
    garbage-collected
    TODO: store string components that are duplicates of strings in the actual system so that
    Python will keep one copy and reduce memory usage
    """

    def __init__(self, evalStr=None, dictKey=NoDictKey):
        if False:
            while True:
                i = 10
        self.evalStr = evalStr
        self.dictKey = NoDictKey
        self._isWeakRef = False
        self._refCount = 0
        if dictKey is not NoDictKey:
            keyRepr = safeRepr(dictKey)
            useEval = False
            try:
                keyEval = eval(keyRepr)
                useEval = True
            except Exception:
                pass
            if useEval:
                if hash(keyEval) != hash(dictKey):
                    useEval = False
            if useEval:
                self.evalStr = '[%s]' % keyRepr
            else:
                try:
                    self.dictKey = weakref.ref(dictKey)
                    self._isWeakRef = True
                except TypeError as e:
                    ContainerLeakDetector.notify.debug('could not weakref dict key %s' % keyRepr)
                    self.dictKey = dictKey
                    self._isWeakRef = False

    def destroy(self):
        if False:
            print('Hello World!')
        self.dictKey = NoDictKey

    def acquire(self):
        if False:
            i = 10
            return i + 15
        self._refCount += 1

    def release(self):
        if False:
            while True:
                i = 10
        self._refCount -= 1
        if self._refCount == 0:
            self.destroy()

    def isDictKey(self):
        if False:
            for i in range(10):
                print('nop')
        return self.dictKey is not NoDictKey

    def _getNonWeakDictKey(self):
        if False:
            return 10
        if not self._isWeakRef:
            return self.dictKey
        else:
            key = self.dictKey()
            if key is None:
                return '<garbage-collected dict key>'
            return key

    def dereferenceDictKey(self, parentDict):
        if False:
            for i in range(10):
                print('nop')
        key = self._getNonWeakDictKey()
        if parentDict is None:
            return key
        return parentDict[key]

    def getString(self, prevIndirection=None, nextIndirection=None):
        if False:
            return 10
        instanceDictStr = '.__dict__'
        if self.evalStr is not None:
            if nextIndirection is not None and self.evalStr[-len(instanceDictStr):] == instanceDictStr:
                return self.evalStr[:-len(instanceDictStr)]
            if prevIndirection is not None and prevIndirection.evalStr is not None:
                if prevIndirection.evalStr[-len(instanceDictStr):] == instanceDictStr:
                    return '.%s' % self.evalStr[2:-2]
            return self.evalStr
        keyRepr = safeRepr(self._getNonWeakDictKey())
        if prevIndirection is not None and prevIndirection.evalStr is not None:
            if prevIndirection.evalStr[-len(instanceDictStr):] == instanceDictStr:
                return '.%s' % keyRepr
        return '[%s]' % keyRepr

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return self.getString()

class ObjectRef:
    """
    stores a reference to a container in a way that does not prevent garbage
    collection of the container if possible
    stored as a series of 'indirections' (obj.foo -> '.foo', dict[key] -> '[key]', etc.)
    """
    notify = directNotify.newCategory('ObjectRef')

    class FailedEval(Exception):
        pass

    def __init__(self, indirection, objId, other=None):
        if False:
            return 10
        self._indirections = []
        if other is not None:
            for ind in other._indirections:
                self._indirections.append(ind)
        assert type(objId) is int
        assert not self.goesThrough(objId=objId)
        self._indirections.append(indirection)
        for ind in self._indirections:
            ind.acquire()
        self.notify.debug(repr(self))

    def destroy(self):
        if False:
            return 10
        for indirection in self._indirections:
            indirection.release()
        del self._indirections

    def getNumIndirections(self):
        if False:
            i = 10
            return i + 15
        return len(self._indirections)

    def goesThroughGen(self, obj=None, objId=None):
        if False:
            while True:
                i = 10
        if obj is None:
            assert type(objId) is int
        else:
            objId = id(obj)
        o = None
        evalStr = ''
        curObj = None
        indirections = self._indirections
        for indirection in indirections:
            yield None
            indirection.acquire()
        for indirection in indirections:
            yield None
            if not indirection.isDictKey():
                evalStr += indirection.getString()
            else:
                curObj = self._getContainerByEval(evalStr, curObj=curObj)
                if curObj is None:
                    raise FailedEval(evalStr)
                curObj = indirection.dereferenceDictKey(curObj)
                evalStr = ''
            yield None
            o = self._getContainerByEval(evalStr, curObj=curObj)
            if id(o) == objId:
                break
        for indirection in indirections:
            yield None
            indirection.release()
        yield (id(o) == objId)

    def goesThrough(self, obj=None, objId=None):
        if False:
            for i in range(10):
                print('nop')
        for goesThrough in self.goesThroughGen(obj=obj, objId=objId):
            pass
        return goesThrough

    def _getContainerByEval(self, evalStr, curObj=None):
        if False:
            i = 10
            return i + 15
        if curObj is not None:
            evalStr = 'curObj%s' % evalStr
        else:
            bis = 'builtins'
            if evalStr[:len(bis)] != bis:
                evalStr = '%s.%s' % (bis, evalStr)
        try:
            container = eval(evalStr)
        except NameError as ne:
            return None
        except AttributeError as ae:
            return None
        except KeyError as ke:
            return None
        return container

    def getContainerGen(self, getInstance=False):
        if False:
            i = 10
            return i + 15
        evalStr = ''
        curObj = None
        indirections = self._indirections
        for indirection in indirections:
            indirection.acquire()
        for indirection in indirections:
            yield None
            if not indirection.isDictKey():
                evalStr += indirection.getString()
            else:
                curObj = self._getContainerByEval(evalStr, curObj=curObj)
                if curObj is None:
                    raise FailedEval(evalStr)
                curObj = indirection.dereferenceDictKey(curObj)
                evalStr = ''
        for indirection in indirections:
            yield None
            indirection.release()
        if getInstance:
            lenDict = len('.__dict__')
            if evalStr[-lenDict:] == '.__dict__':
                evalStr = evalStr[:-lenDict]
        yield self._getContainerByEval(evalStr, curObj=curObj)

    def getEvalStrGen(self, getInstance=False):
        if False:
            print('Hello World!')
        str = ''
        prevIndirection = None
        curIndirection = None
        nextIndirection = None
        indirections = self._indirections
        for indirection in indirections:
            indirection.acquire()
        for i in range(len(indirections)):
            yield None
            if i > 0:
                prevIndirection = indirections[i - 1]
            else:
                prevIndirection = None
            curIndirection = indirections[i]
            if i < len(indirections) - 1:
                nextIndirection = indirections[i + 1]
            else:
                nextIndirection = None
            str += curIndirection.getString(prevIndirection=prevIndirection, nextIndirection=nextIndirection)
        if getInstance:
            lenDict = len('.__dict__')
            if str[-lenDict:] == '.__dict__':
                str = str[:-lenDict]
        for indirection in indirections:
            yield None
            indirection.release()
        yield str

    def getFinalIndirectionStr(self):
        if False:
            for i in range(10):
                print('nop')
        prevIndirection = None
        if len(self._indirections) > 1:
            prevIndirection = self._indirections[-2]
        return self._indirections[-1].getString(prevIndirection=prevIndirection)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        for result in self.getEvalStrGen():
            pass
        return result

class FindContainers(Job):
    """
    Explore the Python graph, looking for objects that support __len__()
    """

    def __init__(self, name, leakDetector):
        if False:
            return 10
        Job.__init__(self, name)
        self._leakDetector = leakDetector
        self._id2ref = self._leakDetector._id2ref
        self._id2baseStartRef = {}
        self._id2discoveredStartRef = {}
        self._baseStartRefWorkingList = ScratchPad(refGen=nullGen(), source=self._id2baseStartRef)
        self._discoveredStartRefWorkingList = ScratchPad(refGen=nullGen(), source=self._id2discoveredStartRef)
        self.notify = self._leakDetector.notify
        ContainerLeakDetector.addPrivateObj(self.__dict__)
        ref = ObjectRef(Indirection(evalStr='builtins.__dict__'), id(builtins.__dict__))
        self._id2baseStartRef[id(builtins.__dict__)] = ref
        if not hasattr(builtins, 'leakDetectors'):
            builtins.leakDetectors = {}
        ref = ObjectRef(Indirection(evalStr='leakDetectors'), id(builtins.leakDetectors))
        self._id2baseStartRef[id(builtins.leakDetectors)] = ref
        for i in self._addContainerGen(builtins.__dict__, ref):
            pass
        try:
            base
        except Exception:
            pass
        else:
            ref = ObjectRef(Indirection(evalStr='base.__dict__'), id(base.__dict__))
            self._id2baseStartRef[id(base.__dict__)] = ref
            for i in self._addContainerGen(base.__dict__, ref):
                pass
        try:
            simbase
        except Exception:
            pass
        else:
            ref = ObjectRef(Indirection(evalStr='simbase.__dict__'), id(simbase.__dict__))
            self._id2baseStartRef[id(simbase.__dict__)] = ref
            for i in self._addContainerGen(simbase.__dict__, ref):
                pass

    def destroy(self):
        if False:
            while True:
                i = 10
        ContainerLeakDetector.removePrivateObj(self.__dict__)
        Job.destroy(self)

    def getPriority(self):
        if False:
            return 10
        return Job.Priorities.Low

    @staticmethod
    def getStartObjAffinity(startObj):
        if False:
            print('Hello World!')
        try:
            return len(startObj)
        except Exception:
            return 1

    def _isDeadEnd(self, obj, objName=None):
        if False:
            while True:
                i = 10
        if type(obj) in deadEndTypes:
            return True
        if id(obj) in ContainerLeakDetector.PrivateIds:
            return True
        if type(objName) == str and objName in ('im_self', 'im_class'):
            return True
        try:
            className = obj.__class__.__name__
        except Exception:
            pass
        else:
            if className == 'method-wrapper':
                return True
        return False

    def _hasLength(self, obj):
        if False:
            i = 10
            return i + 15
        return hasattr(obj, '__len__')

    def _addContainerGen(self, cont, objRef):
        if False:
            while True:
                i = 10
        contId = id(cont)
        if contId in self._id2ref:
            for existingRepr in self._id2ref[contId].getEvalStrGen():
                yield None
            for newRepr in objRef.getEvalStrGen():
                yield None
        if contId not in self._id2ref or len(newRepr) < len(existingRepr):
            if contId in self._id2ref:
                self._leakDetector.removeContainerById(contId)
            self._id2ref[contId] = objRef

    def _addDiscoveredStartRef(self, obj, ref):
        if False:
            i = 10
            return i + 15
        objId = id(obj)
        if objId in self._id2discoveredStartRef:
            existingRef = self._id2discoveredStartRef[objId]
            if type(existingRef) is not int:
                if existingRef.getNumIndirections() >= ref.getNumIndirections():
                    return
        if objId in self._id2ref:
            if self._id2ref[objId].getNumIndirections() >= ref.getNumIndirections():
                return
        storedItem = ref
        if objId in self._id2ref:
            storedItem = objId
        self._id2discoveredStartRef[objId] = storedItem

    def run(self):
        if False:
            return 10
        try:
            workingListSelector = nullGen()
            curObjRef = None
            while True:
                yield None
                if curObjRef is None:
                    try:
                        startRefWorkingList = next(workingListSelector)
                    except StopIteration:
                        baseLen = len(self._baseStartRefWorkingList.source)
                        discLen = len(self._discoveredStartRefWorkingList.source)
                        minLen = float(max(1, min(baseLen, discLen)))
                        minLen *= 3.0
                        workingListSelector = flywheel([self._baseStartRefWorkingList, self._discoveredStartRefWorkingList], [baseLen / minLen, discLen / minLen])
                        yield None
                        continue
                    while True:
                        yield None
                        try:
                            curObjRef = next(startRefWorkingList.refGen)
                            break
                        except StopIteration:
                            if len(startRefWorkingList.source) == 0:
                                break
                            for fw in makeFlywheelGen(list(startRefWorkingList.source.values()), countFunc=lambda x: self.getStartObjAffinity(x), scale=0.05):
                                yield None
                            startRefWorkingList.refGen = fw
                    if curObjRef is None:
                        continue
                    if type(curObjRef) is int:
                        startId = curObjRef
                        curObjRef = None
                        try:
                            for containerRef in self._leakDetector.getContainerByIdGen(startId):
                                yield None
                        except Exception:
                            self.notify.debug('invalid startRef, stored as id %s' % startId)
                            self._leakDetector.removeContainerById(startId)
                            continue
                        curObjRef = containerRef
                try:
                    for curObj in curObjRef.getContainerGen():
                        yield None
                except Exception:
                    self.notify.debug('lost current container, ref.getContainerGen() failed')
                    curObjRef = None
                    continue
                self.notify.debug('--> %s' % curObjRef)
                parentObjRef = curObjRef
                curObjRef = None
                if type(curObj) is types.CellType:
                    child = curObj.cell_contents
                    hasLength = self._hasLength(child)
                    notDeadEnd = not self._isDeadEnd(child)
                    if hasLength or notDeadEnd:
                        objRef = ObjectRef(Indirection(evalStr='.cell_contents'), id(child), parentObjRef)
                        yield None
                        if hasLength:
                            for i in self._addContainerGen(child, objRef):
                                yield None
                        if notDeadEnd:
                            self._addDiscoveredStartRef(child, objRef)
                            curObjRef = objRef
                    continue
                if hasattr(curObj, '__dict__'):
                    child = curObj.__dict__
                    hasLength = self._hasLength(child)
                    notDeadEnd = not self._isDeadEnd(child)
                    if hasLength or notDeadEnd:
                        for goesThrough in parentObjRef.goesThroughGen(child):
                            pass
                        if not goesThrough:
                            objRef = ObjectRef(Indirection(evalStr='.__dict__'), id(child), parentObjRef)
                            yield None
                            if hasLength:
                                for i in self._addContainerGen(child, objRef):
                                    yield None
                            if notDeadEnd:
                                self._addDiscoveredStartRef(child, objRef)
                                curObjRef = objRef
                    continue
                if type(curObj) is dict:
                    key = None
                    attr = None
                    keys = list(curObj.keys())
                    numKeysLeft = len(keys) + 1
                    for key in keys:
                        yield None
                        numKeysLeft -= 1
                        try:
                            attr = curObj[key]
                        except KeyError as e:
                            self.notify.debug('could not index into %s with key %s' % (parentObjRef, safeRepr(key)))
                            continue
                        hasLength = self._hasLength(attr)
                        notDeadEnd = False
                        if curObjRef is None:
                            notDeadEnd = not self._isDeadEnd(attr, key)
                        if hasLength or notDeadEnd:
                            for goesThrough in parentObjRef.goesThroughGen(curObj[key]):
                                pass
                            if not goesThrough:
                                if curObj is builtins.__dict__:
                                    objRef = ObjectRef(Indirection(evalStr='%s' % key), id(curObj[key]))
                                else:
                                    objRef = ObjectRef(Indirection(dictKey=key), id(curObj[key]), parentObjRef)
                                yield None
                                if hasLength:
                                    for i in self._addContainerGen(attr, objRef):
                                        yield None
                                if notDeadEnd:
                                    self._addDiscoveredStartRef(attr, objRef)
                                    if curObjRef is None and random.randrange(numKeysLeft) == 0:
                                        curObjRef = objRef
                    del key
                    del attr
        except Exception as e:
            print('FindContainers job caught exception: %s' % e)
            if __dev__:
                raise
        yield Job.Done

class CheckContainers(Job):
    """
    Job to check container sizes and find potential leaks; sub-job of ContainerLeakDetector
    """
    ReprItems = 5

    def __init__(self, name, leakDetector, index):
        if False:
            while True:
                i = 10
        Job.__init__(self, name)
        self._leakDetector = leakDetector
        self.notify = self._leakDetector.notify
        self._index = index
        ContainerLeakDetector.addPrivateObj(self.__dict__)

    def destroy(self):
        if False:
            while True:
                i = 10
        ContainerLeakDetector.removePrivateObj(self.__dict__)
        Job.destroy(self)

    def getPriority(self):
        if False:
            i = 10
            return i + 15
        return Job.Priorities.Normal

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            self._leakDetector._index2containerId2len[self._index] = {}
            ids = self._leakDetector.getContainerIds()
            for objId in ids:
                yield None
                try:
                    for result in self._leakDetector.getContainerByIdGen(objId):
                        yield None
                    container = result
                except Exception as e:
                    if self.notify.getDebug():
                        for contName in self._leakDetector.getContainerNameByIdGen(objId):
                            yield None
                        self.notify.debug('%s no longer exists; caught exception in getContainerById (%s)' % (contName, e))
                    self._leakDetector.removeContainerById(objId)
                    continue
                if container is None:
                    if self.notify.getDebug():
                        for contName in self._leakDetector.getContainerNameByIdGen(objId):
                            yield None
                        self.notify.debug('%s no longer exists; getContainerById returned None' % contName)
                    self._leakDetector.removeContainerById(objId)
                    continue
                try:
                    cLen = len(container)
                except Exception as e:
                    if self.notify.getDebug():
                        for contName in self._leakDetector.getContainerNameByIdGen(objId):
                            yield None
                        self.notify.debug('%s is no longer a container, it is now %s (%s)' % (contName, safeRepr(container), e))
                    self._leakDetector.removeContainerById(objId)
                    continue
                self._leakDetector._index2containerId2len[self._index][objId] = cLen
            if self._index > 0:
                idx2id2len = self._leakDetector._index2containerId2len
                for objId in idx2id2len[self._index]:
                    yield None
                    if objId in idx2id2len[self._index - 1]:
                        diff = idx2id2len[self._index][objId] - idx2id2len[self._index - 1][objId]
                        "\n                        # this check is too spammy\n                        if diff > 20:\n                            if diff > idx2id2len[self._index-1][objId]:\n                                minutes = (self._leakDetector._index2delay[self._index] -\n                                           self._leakDetector._index2delay[self._index-1]) / 60.\n                                name = self._leakDetector.getContainerNameById(objId)\n                                if idx2id2len[self._index-1][objId] != 0:\n                                    percent = 100. * (float(diff) / float(idx2id2len[self._index-1][objId]))\n                                    try:\n                                        for container in self._leakDetector.getContainerByIdGen(objId):\n                                            yield None\n                                    except Exception:\n                                        # TODO\n                                        self.notify.debug('caught exception in getContainerByIdGen (1)')\n                                    else:\n                                        self.notify.warning(\n                                            '%s (%s) grew %.2f%% in %.2f minutes (%s items at last measurement, current contents: %s)' % (\n                                            name, itype(container), percent, minutes, idx2id2len[self._index][objId],\n                                            fastRepr(container, maxLen=CheckContainers.ReprItems)))\n                                    yield None\n                                    "
                        if self._index > 2 and objId in idx2id2len[self._index - 2] and (objId in idx2id2len[self._index - 3]):
                            diff2 = idx2id2len[self._index - 1][objId] - idx2id2len[self._index - 2][objId]
                            diff3 = idx2id2len[self._index - 2][objId] - idx2id2len[self._index - 3][objId]
                            if self._index <= 4:
                                if diff > 0 and diff2 > 0 and (diff3 > 0):
                                    name = self._leakDetector.getContainerNameById(objId)
                                    try:
                                        for container in self._leakDetector.getContainerByIdGen(objId):
                                            yield None
                                    except Exception:
                                        self.notify.debug('caught exception in getContainerByIdGen (2)')
                                    else:
                                        msg = '%s (%s) consistently increased in size over the last 3 periods (%s items at last measurement, current contents: %s)' % (name, itype(container), idx2id2len[self._index][objId], fastRepr(container, maxLen=CheckContainers.ReprItems))
                                        self.notify.warning(msg)
                                    yield None
                            elif objId in idx2id2len[self._index - 4] and objId in idx2id2len[self._index - 5]:
                                diff4 = idx2id2len[self._index - 3][objId] - idx2id2len[self._index - 4][objId]
                                diff5 = idx2id2len[self._index - 4][objId] - idx2id2len[self._index - 5][objId]
                                if diff > 0 and diff2 > 0 and (diff3 > 0) and (diff4 > 0) and (diff5 > 0):
                                    name = self._leakDetector.getContainerNameById(objId)
                                    try:
                                        for container in self._leakDetector.getContainerByIdGen(objId):
                                            yield None
                                    except Exception:
                                        self.notify.debug('caught exception in getContainerByIdGen (3)')
                                    else:
                                        msg = 'leak detected: %s (%s) consistently increased in size over the last 5 periods (%s items at last measurement, current contents: %s)' % (name, itype(container), idx2id2len[self._index][objId], fastRepr(container, maxLen=CheckContainers.ReprItems))
                                        self.notify.warning(msg)
                                        yield None
                                        messenger.send(self._leakDetector.getLeakEvent(), [container, name])
                                        if config.GetBool('pdb-on-leak-detect', 0):
                                            import pdb
                                            pdb.set_trace()
                                            pass
        except Exception as e:
            print('CheckContainers job caught exception: %s' % e)
            if __dev__:
                raise
        yield Job.Done

class FPTObjsOfType(Job):

    def __init__(self, name, leakDetector, otn, doneCallback=None):
        if False:
            return 10
        Job.__init__(self, name)
        self._leakDetector = leakDetector
        self.notify = self._leakDetector.notify
        self._otn = otn
        self._doneCallback = doneCallback
        self._ldde = self._leakDetector._getDestroyEvent()
        self.accept(self._ldde, self._handleLDDestroy)
        ContainerLeakDetector.addPrivateObj(self.__dict__)

    def destroy(self):
        if False:
            return 10
        self.ignore(self._ldde)
        self._leakDetector = None
        self._doneCallback = None
        ContainerLeakDetector.removePrivateObj(self.__dict__)
        Job.destroy(self)

    def _handleLDDestroy(self):
        if False:
            return 10
        self.destroy()

    def getPriority(self):
        if False:
            print('Hello World!')
        return Job.Priorities.High

    def run(self):
        if False:
            return 10
        ids = self._leakDetector.getContainerIds()
        try:
            for id in ids:
                getInstance = self._otn.lower() not in 'dict'
                yield None
                try:
                    for container in self._leakDetector.getContainerByIdGen(id, getInstance=getInstance):
                        yield None
                except Exception:
                    pass
                else:
                    if hasattr(container, '__class__'):
                        cName = container.__class__.__name__
                    else:
                        cName = container.__name__
                    if self._otn.lower() in cName.lower():
                        try:
                            for ptc in self._leakDetector.getContainerNameByIdGen(id, getInstance=getInstance):
                                yield None
                        except Exception:
                            pass
                        else:
                            print('GPTC(' + self._otn + '):' + self.getJobName() + ': ' + ptc)
        except Exception as e:
            print('FPTObjsOfType job caught exception: %s' % e)
            if __dev__:
                raise
        yield Job.Done

    def finished(self):
        if False:
            print('Hello World!')
        if self._doneCallback:
            self._doneCallback(self)

class FPTObjsNamed(Job):

    def __init__(self, name, leakDetector, on, doneCallback=None):
        if False:
            return 10
        Job.__init__(self, name)
        self._leakDetector = leakDetector
        self.notify = self._leakDetector.notify
        self._on = on
        self._doneCallback = doneCallback
        self._ldde = self._leakDetector._getDestroyEvent()
        self.accept(self._ldde, self._handleLDDestroy)
        ContainerLeakDetector.addPrivateObj(self.__dict__)

    def destroy(self):
        if False:
            i = 10
            return i + 15
        self.ignore(self._ldde)
        self._leakDetector = None
        self._doneCallback = None
        ContainerLeakDetector.removePrivateObj(self.__dict__)
        Job.destroy(self)

    def _handleLDDestroy(self):
        if False:
            for i in range(10):
                print('nop')
        self.destroy()

    def getPriority(self):
        if False:
            while True:
                i = 10
        return Job.Priorities.High

    def run(self):
        if False:
            while True:
                i = 10
        ids = self._leakDetector.getContainerIds()
        try:
            for id in ids:
                yield None
                try:
                    for container in self._leakDetector.getContainerByIdGen(id):
                        yield None
                except Exception:
                    pass
                else:
                    name = self._leakDetector._id2ref[id].getFinalIndirectionStr()
                    if self._on.lower() in name.lower():
                        try:
                            for ptc in self._leakDetector.getContainerNameByIdGen(id):
                                yield None
                        except Exception:
                            pass
                        else:
                            print('GPTCN(' + self._on + '):' + self.getJobName() + ': ' + ptc)
        except Exception as e:
            print('FPTObjsNamed job caught exception: %s' % e)
            if __dev__:
                raise
        yield Job.Done

    def finished(self):
        if False:
            print('Hello World!')
        if self._doneCallback:
            self._doneCallback(self)

class PruneObjectRefs(Job):
    """
    Job to destroy any container refs that are no longer valid.
    Checks validity by asking for each container
    """

    def __init__(self, name, leakDetector):
        if False:
            for i in range(10):
                print('nop')
        Job.__init__(self, name)
        self._leakDetector = leakDetector
        self.notify = self._leakDetector.notify
        ContainerLeakDetector.addPrivateObj(self.__dict__)

    def destroy(self):
        if False:
            print('Hello World!')
        ContainerLeakDetector.removePrivateObj(self.__dict__)
        Job.destroy(self)

    def getPriority(self):
        if False:
            print('Hello World!')
        return Job.Priorities.Normal

    def run(self):
        if False:
            while True:
                i = 10
        try:
            ids = self._leakDetector.getContainerIds()
            for id in ids:
                yield None
                try:
                    for container in self._leakDetector.getContainerByIdGen(id):
                        yield None
                except Exception:
                    self._leakDetector.removeContainerById(id)
            _id2baseStartRef = self._leakDetector._findContainersJob._id2baseStartRef
            ids = list(_id2baseStartRef.keys())
            for id in ids:
                yield None
                try:
                    for container in _id2baseStartRef[id].getContainerGen():
                        yield None
                except Exception:
                    del _id2baseStartRef[id]
            _id2discoveredStartRef = self._leakDetector._findContainersJob._id2discoveredStartRef
            ids = list(_id2discoveredStartRef.keys())
            for id in ids:
                yield None
                try:
                    for container in _id2discoveredStartRef[id].getContainerGen():
                        yield None
                except Exception:
                    del _id2discoveredStartRef[id]
        except Exception as e:
            print('PruneObjectRefs job caught exception: %s' % e)
            if __dev__:
                raise
        yield Job.Done

class ContainerLeakDetector(Job):
    """
    Low-priority Python object-graph walker that looks for leaking containers.
    To reduce memory usage, this does a random walk of the Python objects to
    discover containers rather than keep a set of all visited objects; it may
    visit the same object many times but eventually it will discover every object.
    Checks container sizes at ever-increasing intervals.
    """
    notify = directNotify.newCategory('ContainerLeakDetector')
    PrivateIds: set[int] = set()

    def __init__(self, name, firstCheckDelay=None):
        if False:
            i = 10
            return i + 15
        Job.__init__(self, name)
        self._serialNum = serialNum()
        self._findContainersJob = None
        self._checkContainersJob = None
        self._pruneContainersJob = None
        if firstCheckDelay is None:
            firstCheckDelay = 60.0 * 15.0
        self._nextCheckDelay = firstCheckDelay / 2.0
        self._checkDelayScale = config.GetFloat('leak-detector-check-delay-scale', 1.5)
        self._pruneTaskPeriod = config.GetFloat('leak-detector-prune-period', 60.0 * 30.0)
        self._id2ref = {}
        self._index2containerId2len = {}
        self._index2delay = {}
        if config.GetBool('leak-container', 0):
            _createContainerLeak()
        if config.GetBool('leak-tasks', 0):
            _createTaskLeak()
        ContainerLeakDetector.addPrivateObj(ContainerLeakDetector.PrivateIds)
        ContainerLeakDetector.addPrivateObj(self.__dict__)
        self.setPriority(Job.Priorities.Min)
        jobMgr.add(self)

    def destroy(self):
        if False:
            print('Hello World!')
        messenger.send(self._getDestroyEvent())
        self.ignoreAll()
        if self._pruneContainersJob is not None:
            jobMgr.remove(self._pruneContainersJob)
            self._pruneContainersJob = None
        if self._checkContainersJob is not None:
            jobMgr.remove(self._checkContainersJob)
            self._checkContainersJob = None
        jobMgr.remove(self._findContainersJob)
        self._findContainersJob = None
        del self._id2ref
        del self._index2containerId2len
        del self._index2delay

    def _getDestroyEvent(self):
        if False:
            i = 10
            return i + 15
        return 'cldDestroy-%s' % self._serialNum

    def getLeakEvent(self):
        if False:
            print('Hello World!')
        return 'containerLeakDetected-%s' % self._serialNum

    @classmethod
    def addPrivateObj(cls, obj):
        if False:
            i = 10
            return i + 15
        cls.PrivateIds.add(id(obj))

    @classmethod
    def removePrivateObj(cls, obj):
        if False:
            while True:
                i = 10
        cls.PrivateIds.remove(id(obj))

    def _getCheckTaskName(self):
        if False:
            return 10
        return 'checkForLeakingContainers-%s' % self._serialNum

    def _getPruneTaskName(self):
        if False:
            print('Hello World!')
        return 'pruneLeakingContainerRefs-%s' % self._serialNum

    def getContainerIds(self):
        if False:
            for i in range(10):
                print('nop')
        return list(self._id2ref.keys())

    def getContainerByIdGen(self, id, **kwArgs):
        if False:
            for i in range(10):
                print('nop')
        return self._id2ref[id].getContainerGen(**kwArgs)

    def getContainerById(self, id):
        if False:
            return 10
        for result in self._id2ref[id].getContainerGen():
            pass
        return result

    def getContainerNameByIdGen(self, id, **kwArgs):
        if False:
            while True:
                i = 10
        return self._id2ref[id].getEvalStrGen(**kwArgs)

    def getContainerNameById(self, id):
        if False:
            i = 10
            return i + 15
        if id in self._id2ref:
            return repr(self._id2ref[id])
        return '<unknown container>'

    def removeContainerById(self, id):
        if False:
            i = 10
            return i + 15
        if id in self._id2ref:
            self._id2ref[id].destroy()
            del self._id2ref[id]

    def run(self):
        if False:
            while True:
                i = 10
        self._findContainersJob = FindContainers('%s-findContainers' % self.getJobName(), self)
        jobMgr.add(self._findContainersJob)
        self._scheduleNextLeakCheck()
        self._scheduleNextPruning()
        while True:
            yield Job.Sleep

    def getPathsToContainers(self, name, ot, doneCallback=None):
        if False:
            print('Hello World!')
        j = FPTObjsOfType(name, self, ot, doneCallback)
        jobMgr.add(j)
        return j

    def getPathsToContainersNamed(self, name, on, doneCallback=None):
        if False:
            i = 10
            return i + 15
        j = FPTObjsNamed(name, self, on, doneCallback)
        jobMgr.add(j)
        return j

    def _scheduleNextLeakCheck(self):
        if False:
            print('Hello World!')
        taskMgr.doMethodLater(self._nextCheckDelay, self._checkForLeaks, self._getCheckTaskName())
        self._nextCheckDelay = self._nextCheckDelay * self._checkDelayScale

    def _checkForLeaks(self, task=None):
        if False:
            i = 10
            return i + 15
        self._index2delay[len(self._index2containerId2len)] = self._nextCheckDelay
        self._checkContainersJob = CheckContainers('%s-checkForLeaks' % self.getJobName(), self, len(self._index2containerId2len))
        self.acceptOnce(self._checkContainersJob.getFinishedEvent(), self._scheduleNextLeakCheck)
        jobMgr.add(self._checkContainersJob)
        return task.done

    def _scheduleNextPruning(self):
        if False:
            return 10
        taskMgr.doMethodLater(self._pruneTaskPeriod, self._pruneObjectRefs, self._getPruneTaskName())

    def _pruneObjectRefs(self, task=None):
        if False:
            return 10
        self._pruneContainersJob = PruneObjectRefs('%s-pruneObjectRefs' % self.getJobName(), self)
        self.acceptOnce(self._pruneContainersJob.getFinishedEvent(), self._scheduleNextPruning)
        jobMgr.add(self._pruneContainersJob)
        return task.done