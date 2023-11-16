"""This module defines the EventGroup class."""
__all__ = ['EventGroup']
from direct.showbase import DirectObject
from direct.showbase.PythonUtil import SerialNumGen, Functor
from direct.showbase.MessengerGlobal import messenger

class EventGroup(DirectObject.DirectObject):
    """This class allows you to group together multiple events and treat
    them as a single event. The EventGroup will not send out its event until
    all of its sub-events have occured."""
    _SerialNumGen = SerialNumGen()

    def __init__(self, name, subEvents=None, doneEvent=None):
        if False:
            i = 10
            return i + 15
        "\n        Provide a meaningful name to aid debugging.\n\n        doneEvent is optional. If not provided, a unique done event will be\n        generated and is available as EventGroup.getDoneEvent().\n\n        Examples:\n\n        # waits for gotRed and gotBlue, then sends out 'gotColors'\n        EventGroup('getRedAndBlue', ('gotRed', 'gotBlue'), doneEvent='gotColors')\n\n        # waits for two interests to close, then calls self._handleBothInterestsClosed()\n        # uses EventGroup.getDoneEvent() and EventGroup.newEvent() to generate unique,\n        # disposable event names\n        eGroup = EventGroup('closeInterests')\n        self.acceptOnce(eGroup.getDoneEvent(), self._handleBothInterestsClosed)\n        base.cr.closeInterest(interest1, event=eGroup.newEvent('closeInterest1'))\n        base.cr.closeInterest(interest2, event=eGroup.newEvent('closeInterest2'))\n        "
        self._name = name
        self._subEvents = set()
        self._completedEvents = set()
        if doneEvent is None:
            doneEvent = 'EventGroup-%s-%s-Done' % (EventGroup._SerialNumGen.next(), self._name)
        self._doneEvent = doneEvent
        self._completed = False
        if subEvents is not None:
            for event in subEvents:
                self.addEvent(event)

    def destroy(self):
        if False:
            while True:
                i = 10
        if hasattr(self, '_name'):
            del self._name
            del self._subEvents
            del self._completedEvents
            self.ignoreAll()

    def getName(self):
        if False:
            for i in range(10):
                print('nop')
        return self._name

    def getDoneEvent(self):
        if False:
            i = 10
            return i + 15
        return self._doneEvent

    def isCompleted(self):
        if False:
            for i in range(10):
                print('nop')
        return self._completed

    def addEvent(self, eventName):
        if False:
            while True:
                i = 10
        " Adds a new event to the list of sub-events that we're waiting on.\n        Returns the name of the event. "
        if self._completed:
            self.notify.error("addEvent('%s') called on completed EventGroup '%s'" % (eventName, self.getName()))
        if eventName in self._subEvents:
            self.notify.error("addEvent('%s'): event already in EventGroup '%s'" % (eventName, self.getName()))
        self._subEvents.add(eventName)
        self.acceptOnce(eventName, Functor(self._subEventComplete, eventName))
        return eventName

    def newEvent(self, name):
        if False:
            for i in range(10):
                print('nop')
        " Pass in an event name and it will be unique-ified for you and added\n        to this EventGroup. TIP: there's no need to repeat information in this event\n        name that is already in the name of the EventGroup object.\n        Returns the new event name. "
        return self.addEvent('%s-SubEvent-%s-%s' % (self._name, EventGroup._SerialNumGen.next(), name))

    def _subEventComplete(self, subEventName, *args, **kwArgs):
        if False:
            return 10
        if subEventName in self._completedEvents:
            self.notify.warning("_subEventComplete: '%s' already received" % subEventName)
        else:
            self._completedEvents.add(subEventName)
            if self._completedEvents == self._subEvents:
                self._signalComplete()

    def _signalComplete(self):
        if False:
            return 10
        self._completed = True
        messenger.send(self._doneEvent)
        self.destroy()

    def __repr__(self):
        if False:
            while True:
                i = 10
        return "%s('%s', %s, doneEvent='%s') # completed=%s" % (self.__class__.__name__, self._name, tuple(self._subEvents), self._doneEvent, tuple(self._completedEvents))