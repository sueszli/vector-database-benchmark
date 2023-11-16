from direct.directnotify.DirectNotifyGlobal import directNotify
from direct.showbase.DirectObject import DirectObject
from direct.showbase.EventGroup import EventGroup
from direct.showbase.MessengerGlobal import messenger

class InterestWatcher(DirectObject):
    """Object that observes all interests adds/removes over a period of time,
    and sends out an event when all of those interests have closed"""
    notify = directNotify.newCategory('InterestWatcher')

    def __init__(self, interestMgr, name, doneEvent=None, recurse=True, start=True, mustCollect=False, doCollectionMgr=None):
        if False:
            for i in range(10):
                print('nop')
        DirectObject.__init__(self)
        self._interestMgr = interestMgr
        if doCollectionMgr is None:
            doCollectionMgr = interestMgr
        self._doCollectionMgr = doCollectionMgr
        self._eGroup = EventGroup(name, doneEvent=doneEvent)
        self._doneEvent = self._eGroup.getDoneEvent()
        self._gotEvent = False
        self._recurse = recurse
        if self._recurse:
            self.closingParent2zones = {}
        if start:
            self.startCollect(mustCollect)

    def startCollect(self, mustCollect=False):
        if False:
            i = 10
            return i + 15
        self._mustCollect = mustCollect
        self.accept(self._interestMgr._getAddInterestEvent(), self._handleInterestOpenEvent)
        self.accept(self._interestMgr._getRemoveInterestEvent(), self._handleInterestCloseEvent)

    def stopCollect(self):
        if False:
            while True:
                i = 10
        self.ignore(self._interestMgr._getAddInterestEvent())
        self.ignore(self._interestMgr._getRemoveInterestEvent())
        mustCollect = self._mustCollect
        del self._mustCollect
        if not self._gotEvent:
            if mustCollect:
                logFunc = self.notify.error
            else:
                logFunc = self.notify.warning
            logFunc('%s: empty interest-complete set' % self.getName())
            self.destroy()
            messenger.send(self.getDoneEvent())
        else:
            self.accept(self.getDoneEvent(), self.destroy)

    def destroy(self):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(self, '_eGroup'):
            self._eGroup.destroy()
            del self._eGroup
            del self._gotEvent
            del self._interestMgr
            self.ignoreAll()

    def getName(self):
        if False:
            return 10
        return self._eGroup.getName()

    def getDoneEvent(self):
        if False:
            while True:
                i = 10
        return self._doneEvent

    def _handleInterestOpenEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        self._gotEvent = True
        self._eGroup.addEvent(event)

    def _handleInterestCloseEvent(self, event, parentId, zoneIdList):
        if False:
            while True:
                i = 10
        self._gotEvent = True
        self._eGroup.addEvent(event)