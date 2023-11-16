"""Contains the FourState class."""
__all__ = ['FourState']
from direct.directnotify import DirectNotifyGlobal
from . import ClassicFSM
from . import State

class FourState:
    """
    Generic four state ClassicFSM base class.

    This is a mix-in class that expects that your derived class
    is a DistributedObject.

    Inherit from FourStateFSM and pass in your states.  Two of
    the states should be oposites of each other and the other
    two should be the transition states between the first two.
    E.g::

                    +--------+
                 -->| closed | --
                |   +--------+   |
                |                |
                |                v
          +---------+       +---------+
          | closing |<----->| opening |
          +---------+       +---------+
                ^                |
                |                |
                |    +------+    |
                 ----| open |<---
                     +------+

    There is a fifth off state, but that is an implementation
    detail (and that's why it's not called a five state ClassicFSM).

    I found that this pattern repeated in several things I was
    working on, so this base class was created.
    """
    notify = DirectNotifyGlobal.directNotify.newCategory('FourState')

    def __init__(self, names, durations=[0, 1, None, 1, 1]):
        if False:
            print('Hello World!')
        "\n        Names is a list of state names.  Some examples are::\n\n            ['off', 'opening', 'open', 'closing', 'closed',]\n\n            ['off', 'locking', 'locked', 'unlocking', 'unlocked',]\n\n            ['off', 'deactivating', 'deactive', 'activating', 'activated',]\n\n        durations is a list of time values (floats) or None values.\n\n        Each list must have five entries.\n\n        .. rubric:: More Details\n\n        Here is a diagram showing the where the names from the list\n        are used::\n\n            +---------+\n            | 0 (off) |----> (any other state and vice versa).\n            +---------+\n\n                       +--------+\n                    -->| 4 (on) |---\n                   |   +--------+   |\n                   |                |\n                   |                v\n             +---------+       +---------+\n             | 3 (off) |<----->| 1 (off) |\n             +---------+       +---------+\n                   ^                |\n                   |                |\n                   |  +---------+   |\n                    --| 2 (off) |<--\n                      +---------+\n\n        Each states also has an associated on or off value.  The only\n        state that is 'on' is state 4.  So, the transition states\n        between off and on (states 1 and 3) are also considered\n        off (and so is state 2 which is oposite of 4 and therefore\n        oposite of 'on').\n        "
        self.stateIndex = 0
        assert self.__debugPrint('FourState(names=%s)' % names)
        self.track = None
        self.stateTime = 0.0
        self.names = names
        self.durations = durations
        self.states = {0: State.State(names[0], self.enterState0, self.exitState0, [names[1], names[2], names[3], names[4]]), 1: State.State(names[1], self.enterState1, self.exitState1, [names[2], names[3]]), 2: State.State(names[2], self.enterState2, self.exitState2, [names[3]]), 3: State.State(names[3], self.enterState3, self.exitState3, [names[4], names[1]]), 4: State.State(names[4], self.enterState4, self.exitState4, [names[1]])}
        self.fsm = ClassicFSM.ClassicFSM('FourState', list(self.states.values()), names[0], names[0])
        self.fsm.enterInitialState()

    def setTrack(self, track):
        if False:
            while True:
                i = 10
        assert self.__debugPrint('setTrack(track=%s)' % (track,))
        if self.track is not None:
            self.track.pause()
            self.track = None
        if track is not None:
            track.start(self.stateTime)
            self.track = track

    def enterStateN(self, stateIndex):
        if False:
            i = 10
            return i + 15
        self.stateIndex = stateIndex
        self.duration = self.durations[stateIndex] or 0.0

    def isOn(self):
        if False:
            print('Hello World!')
        assert self.__debugPrint('isOn() returning %s (stateIndex=%s)' % (self.stateIndex == 4, self.stateIndex))
        return self.stateIndex == 4

    def changedOnState(self, isOn):
        if False:
            for i in range(10):
                print('nop')
        '\n        Allow derived classes to overide this.\n        '
        assert self.__debugPrint('changedOnState(isOn=%s)' % (isOn,))

    def enterState0(self):
        if False:
            while True:
                i = 10
        assert self.__debugPrint('enter0()')
        self.enterStateN(0)

    def exitState0(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.__debugPrint('exit0()')
        self.changedOnState(0)

    def enterState1(self):
        if False:
            print('Hello World!')
        assert self.__debugPrint('enterState1()')
        self.enterStateN(1)

    def exitState1(self):
        if False:
            return 10
        assert self.__debugPrint('exitState1()')

    def enterState2(self):
        if False:
            return 10
        assert self.__debugPrint('enterState2()')
        self.enterStateN(2)

    def exitState2(self):
        if False:
            while True:
                i = 10
        assert self.__debugPrint('exitState2()')

    def enterState3(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.__debugPrint('enterState3()')
        self.enterStateN(3)

    def exitState3(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.__debugPrint('exitState3()')

    def enterState4(self):
        if False:
            print('Hello World!')
        assert self.__debugPrint('enterState4()')
        self.enterStateN(4)
        self.changedOnState(1)

    def exitState4(self):
        if False:
            print('Hello World!')
        assert self.__debugPrint('exitState4()')
        self.changedOnState(0)
    if __debug__:

        def __debugPrint(self, message):
            if False:
                while True:
                    i = 10
            'for debugging'
            return self.notify.debug('%d (%d) %s' % (id(self), self.stateIndex == 4, message))