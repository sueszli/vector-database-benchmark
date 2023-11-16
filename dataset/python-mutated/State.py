"""State module: contains State class"""
from __future__ import annotations
__all__ = ['State']
from direct.directnotify.DirectNotifyGlobal import directNotify
from direct.showbase.DirectObject import DirectObject

class State(DirectObject):
    notify = directNotify.newCategory('State')
    Any = 'ANY'
    if __debug__:
        import weakref
        States: weakref.WeakKeyDictionary[State, int] = weakref.WeakKeyDictionary()

        @classmethod
        def replaceMethod(cls, oldFunction, newFunction):
            if False:
                for i in range(10):
                    print('nop')
            import types
            count = 0
            for state in cls.States:
                enterFunc = state.getEnterFunc()
                exitFunc = state.getExitFunc()
                if isinstance(enterFunc, types.MethodType):
                    if enterFunc.__func__ == oldFunction:
                        state.setEnterFunc(types.MethodType(newFunction, enterFunc.__self__))
                        count += 1
                if isinstance(exitFunc, types.MethodType):
                    if exitFunc.__func__ == oldFunction:
                        state.setExitFunc(types.MethodType(newFunction, exitFunc.__self__))
                        count += 1
            return count

    def __init__(self, name, enterFunc=None, exitFunc=None, transitions=Any, inspectorPos=[]):
        if False:
            while True:
                i = 10
        '__init__(self, string, func, func, string[], inspectorPos = [])\n        State constructor: takes name, enter func, exit func, and\n        a list of states it can transition to (or State.Any).'
        self.__name = name
        self.__enterFunc = enterFunc
        self.__exitFunc = exitFunc
        self.__transitions = transitions
        self.__FSMList = []
        if __debug__:
            self.setInspectorPos(inspectorPos)
            self.States[self] = 1

    def getName(self):
        if False:
            return 10
        return self.__name

    def setName(self, stateName):
        if False:
            return 10
        self.__name = stateName

    def getEnterFunc(self):
        if False:
            print('Hello World!')
        return self.__enterFunc

    def setEnterFunc(self, stateEnterFunc):
        if False:
            i = 10
            return i + 15
        self.__enterFunc = stateEnterFunc

    def getExitFunc(self):
        if False:
            return 10
        return self.__exitFunc

    def setExitFunc(self, stateExitFunc):
        if False:
            for i in range(10):
                print('nop')
        self.__exitFunc = stateExitFunc

    def transitionsToAny(self):
        if False:
            print('Hello World!')
        ' returns true if State defines transitions to any other state '
        return self.__transitions is State.Any

    def getTransitions(self):
        if False:
            print('Hello World!')
        '\n        warning -- if the state transitions to any other state,\n        returns an empty list (falsely implying that the state\n        has no transitions)\n        see State.transitionsToAny()\n        '
        if self.transitionsToAny():
            return []
        return self.__transitions

    def isTransitionDefined(self, otherState):
        if False:
            return 10
        if self.transitionsToAny():
            return 1
        if not isinstance(otherState, str):
            otherState = otherState.getName()
        return otherState in self.__transitions

    def setTransitions(self, stateTransitions):
        if False:
            for i in range(10):
                print('nop')
        'setTransitions(self, string[])'
        self.__transitions = stateTransitions

    def addTransition(self, transition):
        if False:
            for i in range(10):
                print('nop')
        'addTransitions(self, string)'
        if not self.transitionsToAny():
            self.__transitions.append(transition)
        else:
            State.notify.warning('attempted to add transition %s to state that transitions to any state')
    if __debug__:

        def getInspectorPos(self):
            if False:
                for i in range(10):
                    print('nop')
            'getInspectorPos(self)'
            return self.__inspectorPos

        def setInspectorPos(self, inspectorPos):
            if False:
                i = 10
                return i + 15
            'setInspectorPos(self, [x, y])'
            self.__inspectorPos = inspectorPos

    def getChildren(self):
        if False:
            print('Hello World!')
        '\n        Return the list of child FSMs\n        '
        return self.__FSMList

    def setChildren(self, FSMList):
        if False:
            while True:
                i = 10
        'setChildren(self, ClassicFSM[])\n        Set the children to given list of FSMs\n        '
        self.__FSMList = FSMList

    def addChild(self, ClassicFSM):
        if False:
            return 10
        '\n        Add the given ClassicFSM to list of child FSMs\n        '
        self.__FSMList.append(ClassicFSM)

    def removeChild(self, ClassicFSM):
        if False:
            print('Hello World!')
        '\n        Remove the given ClassicFSM from list of child FSMs\n        '
        if ClassicFSM in self.__FSMList:
            self.__FSMList.remove(ClassicFSM)

    def hasChildren(self):
        if False:
            i = 10
            return i + 15
        '\n        Return true if state has child FSMs\n        '
        return len(self.__FSMList) > 0

    def __enterChildren(self, argList):
        if False:
            i = 10
            return i + 15
        '\n        Enter all child FSMs\n        '
        for fsm in self.__FSMList:
            if fsm.getCurrentState():
                fsm.conditional_request(fsm.getInitialState().getName())
            else:
                fsm.enterInitialState()

    def __exitChildren(self, argList):
        if False:
            return 10
        '\n        Exit all child FSMs\n        '
        for fsm in self.__FSMList:
            fsm.request(fsm.getFinalState().getName())

    def enter(self, argList=[]):
        if False:
            return 10
        '\n        Call the enter function for this state\n        '
        self.__enterChildren(argList)
        if self.__enterFunc is not None:
            self.__enterFunc(*argList)

    def exit(self, argList=[]):
        if False:
            while True:
                i = 10
        '\n        Call the exit function for this state\n        '
        self.__exitChildren(argList)
        if self.__exitFunc is not None:
            self.__exitFunc(*argList)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'State: name = %s, enter = %s, exit = %s, trans = %s, children = %s' % (self.__name, self.__enterFunc, self.__exitFunc, self.__transitions, self.__FSMList)