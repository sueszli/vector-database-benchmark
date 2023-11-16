"""StateData module: contains StateData class"""
__all__ = ['StateData']
from direct.directnotify.DirectNotifyGlobal import directNotify
from direct.showbase.DirectObject import DirectObject

class StateData(DirectObject):
    """
    A StateData is a base class for a single state within a Finite
    State Machine (ClassicFSM).
    """
    notify = directNotify.newCategory('StateData')

    def __init__(self, doneEvent):
        if False:
            i = 10
            return i + 15
        self.doneEvent = doneEvent
        self.doneStatus = None
        self.isLoaded = 0
        self.isEntered = 0

    def enter(self):
        if False:
            print('Hello World!')
        '\n        Enters the StateData.  This makes it active in whatever sense\n        this applies.  Returns true if this is a change (i.e. it was\n        not previously entered), or false if this is the same (i.e. it\n        was already entered).\n        '
        if self.isEntered:
            return 0
        if not self.isLoaded:
            self.notify.warning('entered StateData before it was loaded')
            self.load()
        self.isEntered = 1
        StateData.notify.debug('enter()')
        return 1

    def exit(self):
        if False:
            while True:
                i = 10
        '\n        Exits the StateData.  Returns true if this is a change\n        (i.e. it was previously entered), or false if this is the same\n        (i.e. it was already exited).\n        '
        if not self.isEntered:
            return 0
        self.isEntered = 0
        StateData.notify.debug('exit()')
        return 1

    def load(self):
        if False:
            i = 10
            return i + 15
        '\n        Loads the StateData.  This loads whatever assets are needed\n        from disk, and otherwise prepares the StateData for being\n        entered, without actually entering it.  Returns true if this\n        is a change (i.e. it was not already loaded), or false if this\n        is the same (i.e. it was previously loaded).\n        '
        if self.isLoaded:
            return 0
        self.isLoaded = 1
        StateData.notify.debug('load()')
        return 1

    def unload(self):
        if False:
            return 10
        '\n        Unloads the StateData.  This frees whatever assets were loaded\n        by load(), and generally makes the memory usage for this thing\n        be as small as possible.  Some StateData-derived classes can\n        load and unload repeatedly; others are useless once they have\n        been unloaded.\n        '
        if not self.isLoaded:
            return 0
        if self.isEntered:
            self.notify.warning('unloaded StateData before it was exited')
            self.exit()
        self.isLoaded = 0
        StateData.notify.debug('unload()')
        return 1

    def getDoneStatus(self):
        if False:
            print('Hello World!')
        '\n        The done status of a state data may be anything.  It is common\n        practice to return a Python dictionary or a string; the default\n        value is None.\n        '
        return self.doneStatus