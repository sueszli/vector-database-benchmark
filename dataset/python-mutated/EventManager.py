"""Contains the EventManager class.  See :mod:`.EventManagerGlobal` for the
global eventMgr instance."""
__all__ = ['EventManager']
from direct.directnotify.DirectNotifyGlobal import directNotify
from direct.task.TaskManagerGlobal import taskMgr
from direct.showbase.MessengerGlobal import messenger
from panda3d.core import PStatCollector, EventQueue, EventHandler
from panda3d.core import ConfigVariableBool

class EventManager:
    notify = None

    def __init__(self, eventQueue=None):
        if False:
            print('Hello World!')
        '\n        Create a C++ event queue and handler\n        '
        if EventManager.notify is None:
            EventManager.notify = directNotify.newCategory('EventManager')
        self.eventQueue = eventQueue
        self.eventHandler = None
        self._wantPstats = ConfigVariableBool('pstats-eventmanager', False)

    def doEvents(self):
        if False:
            while True:
                i = 10
        '\n        Process all the events on the C++ event queue\n        '
        if self._wantPstats:
            processFunc = self.processEventPstats
        else:
            processFunc = self.processEvent
        isEmptyFunc = self.eventQueue.isQueueEmpty
        dequeueFunc = self.eventQueue.dequeueEvent
        while not isEmptyFunc():
            processFunc(dequeueFunc())

    def eventLoopTask(self, task):
        if False:
            print('Hello World!')
        '\n        Process all the events on the C++ event queue\n        '
        self.doEvents()
        messenger.send('event-loop-done')
        return task.cont

    def parseEventParameter(self, eventParameter):
        if False:
            return 10
        '\n        Extract the actual data from the eventParameter\n        '
        if eventParameter.isInt():
            return eventParameter.getIntValue()
        elif eventParameter.isDouble():
            return eventParameter.getDoubleValue()
        elif eventParameter.isString():
            return eventParameter.getStringValue()
        elif eventParameter.isWstring():
            return eventParameter.getWstringValue()
        elif eventParameter.isTypedRefCount():
            return eventParameter.getTypedRefCountValue()
        elif eventParameter.isEmpty():
            return None
        else:
            return eventParameter.getPtr()

    def processEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        '\n        Process a C++ event\n        Duplicate any changes in processEventPstats\n        '
        eventName = event.name
        if eventName:
            paramList = []
            for eventParameter in event.parameters:
                eventParameterData = self.parseEventParameter(eventParameter)
                paramList.append(eventParameterData)
            if EventManager.notify.getDebug() and eventName != 'NewFrame':
                EventManager.notify.debug('received C++ event named: ' + eventName + ' parameters: ' + repr(paramList))
            messenger.send(eventName, paramList)
            handler = self.eventHandler
            if handler:
                handler.dispatchEvent(event)
        else:
            EventManager.notify.warning('unnamed event in processEvent')

    def processEventPstats(self, event):
        if False:
            for i in range(10):
                print('nop')
        '\n        Process a C++ event with pstats tracking\n        Duplicate any changes in processEvent\n        '
        eventName = event.name
        if eventName:
            paramList = []
            for eventParameter in event.parameters:
                eventParameterData = self.parseEventParameter(eventParameter)
                paramList.append(eventParameterData)
            if EventManager.notify.getDebug() and eventName != 'NewFrame':
                EventManager.notify.debug('received C++ event named: ' + eventName + ' parameters: ' + repr(paramList))
            name = eventName
            hyphen = name.find('-')
            if hyphen >= 0:
                name = name[0:hyphen]
            pstatCollector = PStatCollector('App:Tasks:eventManager:' + name)
            pstatCollector.start()
            if self.eventHandler:
                cppPstatCollector = PStatCollector('App:Tasks:eventManager:' + name + ':C++')
            messenger.send(eventName, paramList)
            handler = self.eventHandler
            if handler:
                cppPstatCollector.start()
                handler.dispatchEvent(event)
                cppPstatCollector.stop()
            pstatCollector.stop()
        else:
            EventManager.notify.warning('unnamed event in processEvent')

    def restart(self):
        if False:
            i = 10
            return i + 15
        if self.eventQueue is None:
            self.eventQueue = EventQueue.getGlobalEventQueue()
        if self.eventHandler is None:
            if self.eventQueue == EventQueue.getGlobalEventQueue():
                self.eventHandler = EventHandler.getGlobalEventHandler()
            else:
                self.eventHandler = EventHandler(self.eventQueue)
        taskMgr.add(self.eventLoopTask, 'eventManager')

    def shutdown(self):
        if False:
            i = 10
            return i + 15
        taskMgr.remove('eventManager')
        if self.eventQueue is not None:
            self.eventQueue.clear()
    do_events = doEvents
    process_event = processEvent