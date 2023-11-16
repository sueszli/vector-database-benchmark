from direct.directnotify.DirectNotifyGlobal import directNotify
from direct.showbase.DirectObject import DirectObject
from direct.showbase.PythonUtil import itype, fastRepr
from direct.showbase.Job import Job
from direct.showbase.JobManagerGlobal import jobMgr
from direct.showbase.MessengerGlobal import messenger
import gc
import builtins

class MessengerLeakObject(DirectObject):

    def __init__(self):
        if False:
            print('Hello World!')
        self.accept('leakEvent', self._handleEvent)

    def _handleEvent(self):
        if False:
            while True:
                i = 10
        pass

def _leakMessengerObject():
    if False:
        while True:
            i = 10
    leakObject = MessengerLeakObject()

class MessengerLeakDetector(Job):
    notify = directNotify.newCategory('MessengerLeakDetector')

    def __init__(self, name):
        if False:
            return 10
        Job.__init__(self, name)
        self.setPriority(Job.Priorities.Normal * 2)
        jobMgr.add(self)

    def run(self):
        if False:
            return 10
        builtinIds = set()
        builtinIds.add(id(builtins.__dict__))
        try:
            builtinIds.add(id(base))
            builtinIds.add(id(base.cr))
            builtinIds.add(id(base.cr.doId2do))
        except Exception:
            pass
        try:
            builtinIds.add(id(simbase))
            builtinIds.add(id(simbase.air))
            builtinIds.add(id(simbase.air.doId2do))
        except Exception:
            pass
        try:
            builtinIds.add(id(uber))
            builtinIds.add(id(uber.air))
            builtinIds.add(id(uber.air.doId2do))
        except Exception:
            pass
        while True:
            yield None
            objects = list(messenger._Messenger__objectEvents.keys())
            assert self.notify.debug('%s objects in the messenger' % len(objects))
            for object in objects:
                yield None
                assert self.notify.debug('---> new object: %s' % itype(object))
                objList1 = []
                objList2 = []
                curObjList = objList1
                nextObjList = objList2
                visitedObjIds = set()
                visitedObjIds.add(id(object))
                visitedObjIds.add(id(messenger._Messenger__objectEvents))
                visitedObjIds.add(id(messenger._Messenger__callbacks))
                nextObjList.append(object)
                foundBuiltin = False
                while len(nextObjList) > 0:
                    if foundBuiltin:
                        break
                    curObjList = nextObjList
                    nextObjList = []
                    assert self.notify.debug('next search iteration, num objects: %s' % len(curObjList))
                    for curObj in curObjList:
                        if foundBuiltin:
                            break
                        yield None
                        referrers = gc.get_referrers(curObj)
                        assert self.notify.debug('curObj: %s @ %s, %s referrers, repr=%s' % (itype(curObj), hex(id(curObj)), len(referrers), fastRepr(curObj, maxLen=2)))
                        for referrer in referrers:
                            yield None
                            refId = id(referrer)
                            if refId in visitedObjIds:
                                continue
                            if referrer is curObjList or referrer is nextObjList:
                                continue
                            if refId in builtinIds:
                                foundBuiltin = True
                                break
                            else:
                                visitedObjIds.add(refId)
                                nextObjList.append(referrer)
                if not foundBuiltin:
                    self.notify.warning('%s is referenced only by the messenger' % itype(object))