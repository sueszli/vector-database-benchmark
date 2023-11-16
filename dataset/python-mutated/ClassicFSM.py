"""Finite State Machine module: contains the ClassicFSM class.

Note:
    This module and class exist only for backward compatibility with
    existing code.  New code should use the :mod:`.FSM` module instead.
"""
from __future__ import annotations
__all__ = ['ClassicFSM']
from direct.directnotify.DirectNotifyGlobal import directNotify
from direct.showbase.DirectObject import DirectObject
from direct.showbase.MessengerGlobal import messenger
import weakref
if __debug__:
    _debugFsms: dict[str, weakref.ref] = {}

    def printDebugFsmList():
        if False:
            i = 10
            return i + 15
        for k in sorted(_debugFsms.keys()):
            print('%s %s' % (k, _debugFsms[k]()))
    __builtins__['debugFsmList'] = printDebugFsmList

class ClassicFSM(DirectObject):
    """
    Finite State Machine class.

    This module and class exist only for backward compatibility with
    existing code.  New code should use the FSM class instead.
    """
    notify = directNotify.newCategory('ClassicFSM')
    ALLOW = 0
    DISALLOW = 1
    DISALLOW_VERBOSE = 2
    ERROR = 3

    def __init__(self, name, states=[], initialStateName=None, finalStateName=None, onUndefTransition=DISALLOW_VERBOSE):
        if False:
            for i in range(10):
                print('nop')
        "__init__(self, string, State[], string, string, int)\n\n        ClassicFSM constructor: takes name, list of states, initial state and\n        final state as::\n\n            fsm = ClassicFSM.ClassicFSM('stopLight',\n              [State.State('red', enterRed, exitRed, ['green']),\n                State.State('yellow', enterYellow, exitYellow, ['red']),\n                State.State('green', enterGreen, exitGreen, ['yellow'])],\n              'red',\n              'red')\n\n        each state's last argument, a list of allowed state transitions,\n        is optional; if left out (or explicitly specified to be\n        State.State.Any) then any transition from the state is 'defined'\n        and allowed\n\n        'onUndefTransition' flag determines behavior when undefined\n        transition is requested; see flag definitions above\n        "
        self.setName(name)
        self.setStates(states)
        self.setInitialState(initialStateName)
        self.setFinalState(finalStateName)
        self.onUndefTransition = onUndefTransition
        self.inspecting = 0
        self.__currentState = None
        self.__internalStateInFlux = 0
        if __debug__:
            _debugFsms[name] = weakref.ref(self)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return self.__str__()

    def __str__(self):
        if False:
            i = 10
            return i + 15
        '\n        Print out something useful about the fsm\n        '
        name = self.getName()
        currentState = self.getCurrentState()
        if currentState:
            str = f'ClassicFSM {name} in state "{currentState.getName()}"'
        else:
            str = f'ClassicFSM {name} not in any state'
        return str

    def enterInitialState(self, argList=[]):
        if False:
            print('Hello World!')
        assert not self.__internalStateInFlux
        if self.__currentState == self.__initialState:
            return
        assert self.__currentState is None
        self.__internalStateInFlux = 1
        self.__enter(self.__initialState, argList)
        assert not self.__internalStateInFlux

    def getName(self):
        if False:
            print('Hello World!')
        return self.__name

    def setName(self, name):
        if False:
            print('Hello World!')
        self.__name = name

    def getStates(self):
        if False:
            for i in range(10):
                print('nop')
        return list(self.__states.values())

    def setStates(self, states):
        if False:
            i = 10
            return i + 15
        'setStates(self, State[])'
        self.__states = {}
        for state in states:
            self.__states[state.getName()] = state

    def addState(self, state):
        if False:
            return 10
        self.__states[state.getName()] = state

    def getInitialState(self):
        if False:
            i = 10
            return i + 15
        return self.__initialState

    def setInitialState(self, initialStateName):
        if False:
            for i in range(10):
                print('nop')
        self.__initialState = self.getStateNamed(initialStateName)

    def getFinalState(self):
        if False:
            i = 10
            return i + 15
        return self.__finalState

    def setFinalState(self, finalStateName):
        if False:
            print('Hello World!')
        self.__finalState = self.getStateNamed(finalStateName)

    def requestFinalState(self):
        if False:
            while True:
                i = 10
        self.request(self.getFinalState().getName())

    def getCurrentState(self):
        if False:
            return 10
        return self.__currentState

    def getStateNamed(self, stateName):
        if False:
            print('Hello World!')
        '\n        Return the state with given name if found, issue warning otherwise\n        '
        state = self.__states.get(stateName)
        if state:
            return state
        else:
            ClassicFSM.notify.warning('[%s]: getStateNamed: %s, no such state' % (self.__name, stateName))

    def hasStateNamed(self, stateName):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return True if stateName is a valid state, False otherwise.\n        '
        result = False
        state = self.__states.get(stateName)
        if state:
            result = True
        return result

    def __exitCurrent(self, argList):
        if False:
            for i in range(10):
                print('nop')
        '\n        Exit the current state\n        '
        assert self.__internalStateInFlux
        assert ClassicFSM.notify.debug('[%s]: exiting %s' % (self.__name, self.__currentState.getName()))
        self.__currentState.exit(argList)
        if self.inspecting:
            messenger.send(self.getName() + '_' + self.__currentState.getName() + '_exited')
        self.__currentState = None

    def __enter(self, aState, argList=[]):
        if False:
            return 10
        '\n        Enter a given state, if it exists\n        '
        assert self.__internalStateInFlux
        stateName = aState.getName()
        if stateName in self.__states:
            assert ClassicFSM.notify.debug('[%s]: entering %s' % (self.__name, stateName))
            self.__currentState = aState
            if self.inspecting:
                messenger.send(self.getName() + '_' + stateName + '_entered')
            self.__internalStateInFlux = 0
            aState.enter(argList)
        else:
            self.__internalStateInFlux = 0
            ClassicFSM.notify.error('[%s]: enter: no such state' % self.__name)

    def __transition(self, aState, enterArgList=[], exitArgList=[]):
        if False:
            print('Hello World!')
        '\n        Exit currentState and enter given one\n        '
        assert not self.__internalStateInFlux
        self.__internalStateInFlux = 1
        self.__exitCurrent(exitArgList)
        self.__enter(aState, enterArgList)
        assert not self.__internalStateInFlux

    def request(self, aStateName, enterArgList=[], exitArgList=[], force=0):
        if False:
            for i in range(10):
                print('nop')
        '\n        Attempt transition from currentState to given one.\n        Return true is transition exists to given state,\n        false otherwise.\n        '
        assert not self.__internalStateInFlux
        if not self.__currentState:
            ClassicFSM.notify.warning('[%s]: request: never entered initial state' % self.__name)
            self.__currentState = self.__initialState
        if isinstance(aStateName, str):
            aState = self.getStateNamed(aStateName)
        else:
            aState = aStateName
            aStateName = aState.getName()
        if aState is None:
            ClassicFSM.notify.error('[%s]: request: %s, no such state' % (self.__name, aStateName))
        transitionDefined = self.__currentState.isTransitionDefined(aStateName)
        transitionAllowed = transitionDefined
        if self.onUndefTransition == ClassicFSM.ALLOW:
            transitionAllowed = 1
            if not transitionDefined:
                ClassicFSM.notify.warning('[%s]: performing undefined transition from %s to %s' % (self.__name, self.__currentState.getName(), aStateName))
        if transitionAllowed or force:
            self.__transition(aState, enterArgList, exitArgList)
            return 1
        elif aStateName == self.__finalState.getName():
            if self.__currentState == self.__finalState:
                assert ClassicFSM.notify.debug('[%s]: already in final state: %s' % (self.__name, aStateName))
                return 1
            else:
                assert ClassicFSM.notify.debug('[%s]: implicit transition to final state: %s' % (self.__name, aStateName))
                self.__transition(aState, enterArgList, exitArgList)
                return 1
        elif aStateName == self.__currentState.getName():
            assert ClassicFSM.notify.debug('[%s]: already in state %s and no self transition' % (self.__name, aStateName))
            return 0
        else:
            msg = '[%s]: no transition exists from %s to %s' % (self.__name, self.__currentState.getName(), aStateName)
            if self.onUndefTransition == ClassicFSM.ERROR:
                ClassicFSM.notify.error(msg)
            elif self.onUndefTransition == ClassicFSM.DISALLOW_VERBOSE:
                ClassicFSM.notify.warning(msg)
            return 0

    def forceTransition(self, aStateName, enterArgList=[], exitArgList=[]):
        if False:
            while True:
                i = 10
        '\n        force a transition -- for debugging ONLY\n        '
        self.request(aStateName, enterArgList, exitArgList, force=1)

    def conditional_request(self, aStateName, enterArgList=[], exitArgList=[]):
        if False:
            print('Hello World!')
        "\n        'if this transition is defined, do it'\n        Attempt transition from currentState to given one, if it exists.\n        Return true if transition exists to given state, false otherwise.\n        It is NOT an error/warning to attempt a cond_request if the\n        transition doesn't exist.  This lets people be sloppy about\n        ClassicFSM transitions, letting the same fn be used for different\n        states that may not have the same out transitions.\n        "
        assert not self.__internalStateInFlux
        if not self.__currentState:
            ClassicFSM.notify.warning('[%s]: request: never entered initial state' % self.__name)
            self.__currentState = self.__initialState
        if isinstance(aStateName, str):
            aState = self.getStateNamed(aStateName)
        else:
            aState = aStateName
            aStateName = aState.getName()
        if aState is None:
            ClassicFSM.notify.error('[%s]: request: %s, no such state' % (self.__name, aStateName))
        transitionDefined = self.__currentState.isTransitionDefined(aStateName) or aStateName in [self.__currentState.getName(), self.__finalState.getName()]
        if transitionDefined:
            return self.request(aStateName, enterArgList, exitArgList)
        else:
            assert ClassicFSM.notify.debug('[%s]: condition_request: %s, transition doesnt exist' % (self.__name, aStateName))
            return 0

    def view(self):
        if False:
            print('Hello World!')
        import importlib
        FSMInspector = importlib.import_module('direct.tkpanels.FSMInspector')
        FSMInspector.FSMInspector(self)

    def isInternalStateInFlux(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__internalStateInFlux