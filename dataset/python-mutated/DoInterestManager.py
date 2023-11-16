"""
The DoInterestManager keeps track of which parent/zones that we currently
have interest in.  When you want to "look" into a zone you add an interest
to that zone.  When you want to get rid of, or ignore, the objects in that
zone, remove interest in that zone.

p.s. A great deal of this code is just code moved from ClientRepository.py.
"""
from __future__ import annotations
from panda3d.core import ConfigVariableBool
from .MsgTypes import CLIENT_ADD_INTEREST, CLIENT_ADD_INTEREST_MULTIPLE, CLIENT_REMOVE_INTEREST
from direct.showbase import DirectObject
from direct.showbase.MessengerGlobal import messenger
from .PyDatagram import PyDatagram
from direct.directnotify.DirectNotifyGlobal import directNotify
import types
from direct.showbase.PythonUtil import FrameDelayedCall, ScratchPad, SerialNumGen, report, serialNum, uniqueElements, uniqueName

class InterestState:
    StateActive = 'Active'
    StatePendingDel = 'PendingDel'

    def __init__(self, desc, state, context, event, parentId, zoneIdList, eventCounter, auto=False):
        if False:
            i = 10
            return i + 15
        self.desc = desc
        self.state = state
        self.context = context
        self.events = []
        self.eventCounter = eventCounter
        if event:
            self.addEvent(event)
        self.parentId = parentId
        self.zoneIdList = zoneIdList
        self.auto = auto

    def addEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        self.events.append(event)
        self.eventCounter.num += 1

    def getEvents(self):
        if False:
            return 10
        return list(self.events)

    def clearEvents(self):
        if False:
            while True:
                i = 10
        self.eventCounter.num -= len(self.events)
        assert self.eventCounter.num >= 0
        self.events = []

    def sendEvents(self):
        if False:
            while True:
                i = 10
        for event in self.events:
            messenger.send(event)
        self.clearEvents()

    def setDesc(self, desc):
        if False:
            i = 10
            return i + 15
        self.desc = desc

    def isPendingDelete(self):
        if False:
            for i in range(10):
                print('nop')
        return self.state == InterestState.StatePendingDel

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'InterestState(desc=%s, state=%s, context=%s, event=%s, parentId=%s, zoneIdList=%s)' % (self.desc, self.state, self.context, self.events, self.parentId, self.zoneIdList)

class InterestHandle:
    """This class helps to ensure that valid handles get passed in to DoInterestManager funcs"""

    def __init__(self, id):
        if False:
            for i in range(10):
                print('nop')
        self._id = id

    def asInt(self):
        if False:
            print('Hello World!')
        return self._id

    def __eq__(self, other):
        if False:
            print('Hello World!')
        if type(self) == type(other):
            return self._id == other._id
        return self._id == other

    def __repr__(self):
        if False:
            print('Hello World!')
        return '%s(%s)' % (self.__class__.__name__, self._id)
NO_CONTEXT = 0

class DoInterestManager(DirectObject.DirectObject):
    """
    Top level Interest Manager
    """
    notify = directNotify.newCategory('DoInterestManager')
    InterestDebug = ConfigVariableBool('interest-debug', False)
    _HandleSerialNum = 0
    _HandleMask = 32767
    _ContextIdSerialNum = 100
    _ContextIdMask = 1073741823
    _interests: dict[int, InterestState] = {}
    if __debug__:
        _debug_interestHistory: list[tuple] = []
        _debug_maxDescriptionLen = 40
    _SerialGen = SerialNumGen()
    _SerialNum = serialNum()

    def __init__(self):
        if False:
            print('Hello World!')
        assert DoInterestManager.notify.debugCall()
        DirectObject.DirectObject.__init__(self)
        self._addInterestEvent = uniqueName('DoInterestManager-Add')
        self._removeInterestEvent = uniqueName('DoInterestManager-Remove')
        self._noNewInterests = False
        self._completeDelayedCallback = None
        self._completeEventCount = ScratchPad(num=0)
        self._allInterestsCompleteCallbacks = []

    def __verbose(self):
        if False:
            i = 10
            return i + 15
        return self.InterestDebug.getValue() or self.getVerbose()

    def _getAnonymousEvent(self, desc):
        if False:
            for i in range(10):
                print('nop')
        return 'anonymous-%s-%s' % (desc, DoInterestManager._SerialGen.next())

    def setNoNewInterests(self, flag):
        if False:
            print('Hello World!')
        self._noNewInterests = flag

    def noNewInterests(self):
        if False:
            for i in range(10):
                print('nop')
        return self._noNewInterests

    def setAllInterestsCompleteCallback(self, callback):
        if False:
            while True:
                i = 10
        if self._completeEventCount.num == 0 and self._completeDelayedCallback is None:
            callback()
        else:
            self._allInterestsCompleteCallbacks.append(callback)

    def getAllInterestsCompleteEvent(self):
        if False:
            while True:
                i = 10
        return 'allInterestsComplete-%s' % DoInterestManager._SerialNum

    def resetInterestStateForConnectionLoss(self):
        if False:
            print('Hello World!')
        DoInterestManager._interests.clear()
        self._completeEventCount = ScratchPad(num=0)
        if __debug__:
            self._addDebugInterestHistory('RESET', '', 0, 0, 0, [])

    def isValidInterestHandle(self, handle):
        if False:
            while True:
                i = 10
        if not isinstance(handle, InterestHandle):
            return False
        return handle.asInt() in DoInterestManager._interests

    def updateInterestDescription(self, handle, desc):
        if False:
            for i in range(10):
                print('nop')
        iState = DoInterestManager._interests.get(handle.asInt())
        if iState:
            iState.setDesc(desc)

    def addInterest(self, parentId, zoneIdList, description, event=None):
        if False:
            i = 10
            return i + 15
        '\n        Look into a (set of) zone(s).\n        '
        assert DoInterestManager.notify.debugCall()
        handle = self._getNextHandle()
        if self._noNewInterests:
            DoInterestManager.notify.warning('addInterest: addingInterests on delete: %s' % handle)
            return
        if parentId not in (self.getGameDoId(),):
            parent = self.getDo(parentId)
            if not parent:
                DoInterestManager.notify.error('addInterest: attempting to add interest under unknown object %s' % parentId)
            elif not parent.hasParentingRules():
                DoInterestManager.notify.error('addInterest: no setParentingRules defined in the DC for object %s (%s)' % (parentId, parent.__class__.__name__))
        if event:
            contextId = self._getNextContextId()
        else:
            contextId = 0
        DoInterestManager._interests[handle] = InterestState(description, InterestState.StateActive, contextId, event, parentId, zoneIdList, self._completeEventCount)
        if self.__verbose():
            print('CR::INTEREST.addInterest(handle=%s, parentId=%s, zoneIdList=%s, description=%s, event=%s)' % (handle, parentId, zoneIdList, description, event))
        self._sendAddInterest(handle, contextId, parentId, zoneIdList, description)
        if event:
            messenger.send(self._getAddInterestEvent(), [event])
        assert self.printInterestsIfDebug()
        return InterestHandle(handle)

    def addAutoInterest(self, parentId, zoneIdList, description):
        if False:
            while True:
                i = 10
        '\n        Look into a (set of) zone(s).\n        '
        assert DoInterestManager.notify.debugCall()
        handle = self._getNextHandle()
        if self._noNewInterests:
            DoInterestManager.notify.warning('addInterest: addingInterests on delete: %s' % handle)
            return
        if parentId not in (self.getGameDoId(),):
            parent = self.getDo(parentId)
            if not parent:
                DoInterestManager.notify.error('addInterest: attempting to add interest under unknown object %s' % parentId)
            elif not parent.hasParentingRules():
                DoInterestManager.notify.error('addInterest: no setParentingRules defined in the DC for object %s (%s)' % (parentId, parent.__class__.__name__))
        DoInterestManager._interests[handle] = InterestState(description, InterestState.StateActive, 0, None, parentId, zoneIdList, self._completeEventCount, True)
        if self.__verbose():
            print('CR::INTEREST.addInterest(handle=%s, parentId=%s, zoneIdList=%s, description=%s)' % (handle, parentId, zoneIdList, description))
        assert self.printInterestsIfDebug()
        return InterestHandle(handle)

    def removeInterest(self, handle, event=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Stop looking in a (set of) zone(s)\n        '
        assert DoInterestManager.notify.debugCall()
        assert isinstance(handle, InterestHandle)
        existed = False
        if not event:
            event = self._getAnonymousEvent('removeInterest')
        handle = handle.asInt()
        if handle in DoInterestManager._interests:
            existed = True
            intState = DoInterestManager._interests[handle]
            if event:
                messenger.send(self._getRemoveInterestEvent(), [event, intState.parentId, intState.zoneIdList])
            if intState.isPendingDelete():
                self.notify.warning('removeInterest: interest %s already pending removal' % handle)
                if event is not None:
                    intState.addEvent(event)
            else:
                if len(intState.events) > 0:
                    assert self.notify.warning('removeInterest: abandoning events: %s' % intState.events)
                    intState.clearEvents()
                intState.state = InterestState.StatePendingDel
                contextId = self._getNextContextId()
                intState.context = contextId
                if event:
                    intState.addEvent(event)
                self._sendRemoveInterest(handle, contextId)
                if not event:
                    self._considerRemoveInterest(handle)
                if self.__verbose():
                    print('CR::INTEREST.removeInterest(handle=%s, event=%s)' % (handle, event))
        else:
            DoInterestManager.notify.warning('removeInterest: handle not found: %s' % handle)
        assert self.printInterestsIfDebug()
        return existed

    def removeAutoInterest(self, handle):
        if False:
            return 10
        '\n        Stop looking in a (set of) zone(s)\n        '
        assert DoInterestManager.notify.debugCall()
        assert isinstance(handle, InterestHandle)
        existed = False
        handle = handle.asInt()
        if handle in DoInterestManager._interests:
            existed = True
            intState = DoInterestManager._interests[handle]
            if intState.isPendingDelete():
                self.notify.warning('removeInterest: interest %s already pending removal' % handle)
            else:
                if len(intState.events) > 0:
                    self.notify.warning('removeInterest: abandoning events: %s' % intState.events)
                    intState.clearEvents()
                intState.state = InterestState.StatePendingDel
                self._considerRemoveInterest(handle)
                if self.__verbose():
                    print('CR::INTEREST.removeAutoInterest(handle=%s)' % handle)
        else:
            DoInterestManager.notify.warning('removeInterest: handle not found: %s' % handle)
        assert self.printInterestsIfDebug()
        return existed

    @report(types=['args'], dConfigParam='guildmgr')
    def removeAIInterest(self, handle):
        if False:
            return 10
        "\n        handle is NOT an InterestHandle.  It's just a bare integer representing an\n        AI opened interest. We're making the client close down this interest since\n        the AI has trouble removing interests(that its opened) when the avatar goes\n        offline.  See GuildManager(UD) for how it's being used.\n        "
        self._sendRemoveAIInterest(handle)

    def alterInterest(self, handle, parentId, zoneIdList, description=None, event=None):
        if False:
            while True:
                i = 10
        "\n        Removes old interests and adds new interests.\n\n        Note that when an interest is changed, only the most recent\n        change's event will be triggered. Previous events are abandoned.\n        If this is a problem, consider opening multiple interests.\n        "
        assert DoInterestManager.notify.debugCall()
        assert isinstance(handle, InterestHandle)
        handle = handle.asInt()
        if self._noNewInterests:
            DoInterestManager.notify.warning('alterInterest: addingInterests on delete: %s' % handle)
            return
        exists = False
        if event is None:
            event = self._getAnonymousEvent('alterInterest')
        if handle in DoInterestManager._interests:
            if description is not None:
                DoInterestManager._interests[handle].desc = description
            else:
                description = DoInterestManager._interests[handle].desc
            if DoInterestManager._interests[handle].context != NO_CONTEXT:
                DoInterestManager._interests[handle].clearEvents()
            contextId = self._getNextContextId()
            DoInterestManager._interests[handle].context = contextId
            DoInterestManager._interests[handle].parentId = parentId
            DoInterestManager._interests[handle].zoneIdList = zoneIdList
            DoInterestManager._interests[handle].addEvent(event)
            if self.__verbose():
                print('CR::INTEREST.alterInterest(handle=%s, parentId=%s, zoneIdList=%s, description=%s, event=%s)' % (handle, parentId, zoneIdList, description, event))
            self._sendAddInterest(handle, contextId, parentId, zoneIdList, description, action='modify')
            exists = True
            assert self.printInterestsIfDebug()
        else:
            DoInterestManager.notify.warning('alterInterest: handle not found: %s' % handle)
        return exists

    def openAutoInterests(self, obj):
        if False:
            print('Hello World!')
        if hasattr(obj, '_autoInterestHandle'):
            self.notify.debug('openAutoInterests(%s): interests already open' % obj.__class__.__name__)
            return
        autoInterests = obj.getAutoInterests()
        obj._autoInterestHandle = None
        if len(autoInterests) == 0:
            return
        obj._autoInterestHandle = self.addAutoInterest(obj.doId, autoInterests, '%s-autoInterest' % obj.__class__.__name__)

    def closeAutoInterests(self, obj):
        if False:
            print('Hello World!')
        if not hasattr(obj, '_autoInterestHandle'):
            self.notify.debug('closeAutoInterests(%s): interests already closed' % obj)
            return
        if obj._autoInterestHandle is not None:
            self.removeAutoInterest(obj._autoInterestHandle)
        del obj._autoInterestHandle

    def _getAddInterestEvent(self):
        if False:
            return 10
        return self._addInterestEvent

    def _getRemoveInterestEvent(self):
        if False:
            for i in range(10):
                print('nop')
        return self._removeInterestEvent

    def _getInterestState(self, handle):
        if False:
            return 10
        return DoInterestManager._interests[handle]

    def _getNextHandle(self):
        if False:
            while True:
                i = 10
        handle = DoInterestManager._HandleSerialNum
        while True:
            handle = handle + 1 & DoInterestManager._HandleMask
            if handle not in DoInterestManager._interests:
                break
            DoInterestManager.notify.warning('interest %s already in use' % handle)
        DoInterestManager._HandleSerialNum = handle
        return DoInterestManager._HandleSerialNum

    def _getNextContextId(self):
        if False:
            return 10
        contextId = DoInterestManager._ContextIdSerialNum
        while True:
            contextId = contextId + 1 & DoInterestManager._ContextIdMask
            if contextId != NO_CONTEXT:
                break
        DoInterestManager._ContextIdSerialNum = contextId
        return DoInterestManager._ContextIdSerialNum

    def _considerRemoveInterest(self, handle):
        if False:
            return 10
        '\n        Consider whether we should cull the interest set.\n        '
        assert DoInterestManager.notify.debugCall()
        if handle in DoInterestManager._interests:
            if DoInterestManager._interests[handle].isPendingDelete():
                if DoInterestManager._interests[handle].context == NO_CONTEXT:
                    assert len(DoInterestManager._interests[handle].events) == 0
                    del DoInterestManager._interests[handle]
    if __debug__:

        def printInterestsIfDebug(self):
            if False:
                print('Hello World!')
            if DoInterestManager.notify.getDebug():
                self.printInterests()
            return 1

        def _addDebugInterestHistory(self, action, description, handle, contextId, parentId, zoneIdList):
            if False:
                print('Hello World!')
            if description is None:
                description = ''
            DoInterestManager._debug_interestHistory.append((action, description, handle, contextId, parentId, zoneIdList))
            DoInterestManager._debug_maxDescriptionLen = max(DoInterestManager._debug_maxDescriptionLen, len(description))

        def printInterestHistory(self):
            if False:
                i = 10
                return i + 15
            print('***************** Interest History *************')
            format = '%9s %' + str(DoInterestManager._debug_maxDescriptionLen) + 's %6s %6s %9s %s'
            print(format % ('Action', 'Description', 'Handle', 'Context', 'ParentId', 'ZoneIdList'))
            for i in DoInterestManager._debug_interestHistory:
                print(format % tuple(i))
            print('Note: interests with a Context of 0 do not get done/finished notices.')

        def printInterestSets(self):
            if False:
                while True:
                    i = 10
            print('******************* Interest Sets **************')
            format = '%6s %' + str(DoInterestManager._debug_maxDescriptionLen) + 's %11s %11s %8s %8s %8s'
            print(format % ('Handle', 'Description', 'ParentId', 'ZoneIdList', 'State', 'Context', 'Event'))
            for (id, state) in DoInterestManager._interests.items():
                if len(state.events) == 0:
                    event = ''
                elif len(state.events) == 1:
                    event = state.events[0]
                else:
                    event = state.events
                print(format % (id, state.desc, state.parentId, state.zoneIdList, state.state, state.context, event))
            print('************************************************')

        def printInterests(self):
            if False:
                i = 10
                return i + 15
            self.printInterestHistory()
            self.printInterestSets()

    def _sendAddInterest(self, handle, contextId, parentId, zoneIdList, description, action=None):
        if False:
            return 10
        "\n        Part of the new otp-server code.\n\n        handle is a client-side created number that refers to\n                a set of interests.  The same handle number doesn't\n                necessarily have any relationship to the same handle\n                on another client.\n        "
        assert DoInterestManager.notify.debugCall()
        if __debug__:
            if isinstance(zoneIdList, list):
                zoneIdList.sort()
            if action is None:
                action = 'add'
            self._addDebugInterestHistory(action, description, handle, contextId, parentId, zoneIdList)
        if parentId == 0:
            DoInterestManager.notify.error('trying to set interest to invalid parent: %s' % parentId)
        datagram = PyDatagram()
        if isinstance(zoneIdList, list):
            vzl = sorted(zoneIdList)
            uniqueElements(vzl)
            datagram.addUint16(CLIENT_ADD_INTEREST_MULTIPLE)
            datagram.addUint32(contextId)
            datagram.addUint16(handle)
            datagram.addUint32(parentId)
            datagram.addUint16(len(vzl))
            for zone in vzl:
                datagram.addUint32(zone)
        else:
            datagram.addUint16(CLIENT_ADD_INTEREST)
            datagram.addUint32(contextId)
            datagram.addUint16(handle)
            datagram.addUint32(parentId)
            datagram.addUint32(zoneIdList)
        self.send(datagram)

    def _sendRemoveInterest(self, handle, contextId):
        if False:
            return 10
        "\n        handle is a client-side created number that refers to\n                a set of interests.  The same handle number doesn't\n                necessarily have any relationship to the same handle\n                on another client.\n        "
        assert DoInterestManager.notify.debugCall()
        assert handle in DoInterestManager._interests
        datagram = PyDatagram()
        datagram.addUint16(CLIENT_REMOVE_INTEREST)
        datagram.addUint32(contextId)
        datagram.addUint16(handle)
        self.send(datagram)
        if __debug__:
            state = DoInterestManager._interests[handle]
            self._addDebugInterestHistory('remove', state.desc, handle, contextId, state.parentId, state.zoneIdList)

    def _sendRemoveAIInterest(self, handle):
        if False:
            while True:
                i = 10
        '\n        handle is a bare int, NOT an InterestHandle.  Use this to\n        close an AI opened interest.\n        '
        datagram = PyDatagram()
        datagram.addUint16(CLIENT_REMOVE_INTEREST)
        datagram.addUint16((1 << 15) + handle)
        self.send(datagram)

    def cleanupWaitAllInterestsComplete(self):
        if False:
            for i in range(10):
                print('nop')
        if self._completeDelayedCallback is not None:
            self._completeDelayedCallback.destroy()
            self._completeDelayedCallback = None

    def queueAllInterestsCompleteEvent(self, frames=5):
        if False:
            for i in range(10):
                print('nop')

        def checkMoreInterests():
            if False:
                for i in range(10):
                    print('nop')
            return self._completeEventCount.num > 0

        def sendEvent():
            if False:
                while True:
                    i = 10
            messenger.send(self.getAllInterestsCompleteEvent())
            for callback in self._allInterestsCompleteCallbacks:
                callback()
            self._allInterestsCompleteCallbacks = []
        self.cleanupWaitAllInterestsComplete()
        self._completeDelayedCallback = FrameDelayedCall('waitForAllInterestCompletes', callback=sendEvent, frames=frames, cancelFunc=checkMoreInterests)
        checkMoreInterests = None
        sendEvent = None

    def handleInterestDoneMessage(self, di):
        if False:
            return 10
        '\n        This handles the interest done messages and may dispatch an event\n        '
        assert DoInterestManager.notify.debugCall()
        contextId = di.getUint32()
        handle = di.getUint16()
        if self.__verbose():
            print('CR::INTEREST.interestDone(handle=%s)' % handle)
        DoInterestManager.notify.debug('handleInterestDoneMessage--> Received handle %s, context %s' % (handle, contextId))
        if handle in DoInterestManager._interests:
            eventsToSend = []
            if contextId == DoInterestManager._interests[handle].context:
                DoInterestManager._interests[handle].context = NO_CONTEXT
                eventsToSend = list(DoInterestManager._interests[handle].getEvents())
                DoInterestManager._interests[handle].clearEvents()
            else:
                DoInterestManager.notify.debug('handleInterestDoneMessage--> handle: %s: Expecting context %s, got %s' % (handle, DoInterestManager._interests[handle].context, contextId))
            if __debug__:
                state = DoInterestManager._interests[handle]
                self._addDebugInterestHistory('finished', state.desc, handle, contextId, state.parentId, state.zoneIdList)
            self._considerRemoveInterest(handle)
            for event in eventsToSend:
                messenger.send(event)
        else:
            DoInterestManager.notify.warning('handleInterestDoneMessage: handle not found: %s' % handle)
        if self._completeEventCount.num == 0:
            self.queueAllInterestsCompleteEvent()
        assert self.printInterestsIfDebug()
if __debug__:
    import unittest
    import time

    class AsyncTestCase(unittest.TestCase):

        def setCompleted(self):
            if False:
                while True:
                    i = 10
            self._async_completed = True

        def isCompleted(self):
            if False:
                print('Hello World!')
            return getattr(self, '_async_completed', False)

    class AsyncTestSuite(unittest.TestSuite):
        pass

    class AsyncTestLoader(unittest.TestLoader):
        suiteClass = AsyncTestSuite

    class AsyncTextTestRunner(unittest.TextTestRunner):

        def run(self, test):
            if False:
                while True:
                    i = 10
            result = self._makeResult()
            startTime = time.time()
            test(result)
            stopTime = time.time()
            timeTaken = stopTime - startTime
            result.printErrors()
            self.stream.writeln(result.separator2)
            run = result.testsRun
            self.stream.writeln('Ran %d test%s in %.3fs' % (run, run != 1 and 's' or '', timeTaken))
            self.stream.writeln()
            if not result.wasSuccessful():
                self.stream.write('FAILED (')
                (failed, errored) = map(len, (result.failures, result.errors))
                if failed:
                    self.stream.write('failures=%d' % failed)
                if errored:
                    if failed:
                        self.stream.write(', ')
                    self.stream.write('errors=%d' % errored)
                self.stream.writeln(')')
            else:
                self.stream.writeln('OK')
            return result

    class TestInterestAddRemove(AsyncTestCase, DirectObject.DirectObject):

        def testInterestAdd(self):
            if False:
                print('Hello World!')
            event = uniqueName('InterestAdd')
            self.acceptOnce(event, self.gotInterestAddResponse)
            self.handle = base.cr.addInterest(base.cr.GameGlobalsId, 100, 'TestInterest', event=event)

        def gotInterestAddResponse(self):
            if False:
                for i in range(10):
                    print('nop')
            event = uniqueName('InterestRemove')
            self.acceptOnce(event, self.gotInterestRemoveResponse)
            base.cr.removeInterest(self.handle, event=event)

        def gotInterestRemoveResponse(self):
            if False:
                i = 10
                return i + 15
            self.setCompleted()

    def runTests():
        if False:
            print('Hello World!')
        suite = unittest.makeSuite(TestInterestAddRemove)
        unittest.AsyncTextTestRunner(verbosity=2).run(suite)