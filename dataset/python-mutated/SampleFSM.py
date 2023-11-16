"""Undocumented Module"""
__all__ = ['ClassicStyle', 'NewStyle', 'ToonEyes']
from . import FSM
from direct.task import Task
from direct.task.TaskManagerGlobal import taskMgr

class ClassicStyle(FSM.FSM):

    def __init__(self, name):
        if False:
            for i in range(10):
                print('nop')
        FSM.FSM.__init__(self, name)
        self.defaultTransitions = {'Red': ['Green'], 'Yellow': ['Red'], 'Green': ['Yellow']}

    def enterRed(self):
        if False:
            for i in range(10):
                print('nop')
        print("enterRed(self, '%s', '%s')" % (self.oldState, self.newState))

    def exitRed(self):
        if False:
            while True:
                i = 10
        print("exitRed(self, '%s', '%s')" % (self.oldState, self.newState))

    def enterYellow(self):
        if False:
            return 10
        print("enterYellow(self, '%s', '%s')" % (self.oldState, self.newState))

    def exitYellow(self):
        if False:
            for i in range(10):
                print('nop')
        print("exitYellow(self, '%s', '%s')" % (self.oldState, self.newState))

    def enterGreen(self):
        if False:
            return 10
        print("enterGreen(self, '%s', '%s')" % (self.oldState, self.newState))

    def exitGreen(self):
        if False:
            while True:
                i = 10
        print("exitGreen(self, '%s', '%s')" % (self.oldState, self.newState))

class NewStyle(FSM.FSM):

    def enterRed(self):
        if False:
            for i in range(10):
                print('nop')
        print("enterRed(self, '%s', '%s')" % (self.oldState, self.newState))

    def filterRed(self, request, args):
        if False:
            print('Hello World!')
        print("filterRed(self, '%s', %s)" % (request, args))
        if request == 'advance':
            return 'Green'
        return self.defaultFilter(request, args)

    def exitRed(self):
        if False:
            for i in range(10):
                print('nop')
        print("exitRed(self, '%s', '%s')" % (self.oldState, self.newState))

    def enterYellow(self):
        if False:
            return 10
        print("enterYellow(self, '%s', '%s')" % (self.oldState, self.newState))

    def filterYellow(self, request, args):
        if False:
            i = 10
            return i + 15
        print("filterYellow(self, '%s', %s)" % (request, args))
        if request == 'advance':
            return 'Red'
        return self.defaultFilter(request, args)

    def exitYellow(self):
        if False:
            while True:
                i = 10
        print("exitYellow(self, '%s', '%s')" % (self.oldState, self.newState))

    def enterGreen(self):
        if False:
            return 10
        print("enterGreen(self, '%s', '%s')" % (self.oldState, self.newState))

    def filterGreen(self, request, args):
        if False:
            while True:
                i = 10
        print("filterGreen(self, '%s', %s)" % (request, args))
        if request == 'advance':
            return 'Yellow'
        return self.defaultFilter(request, args)

    def exitGreen(self):
        if False:
            while True:
                i = 10
        print("exitGreen(self, '%s', '%s')" % (self.oldState, self.newState))

class ToonEyes(FSM.FSM):

    def __init__(self):
        if False:
            print('Hello World!')
        FSM.FSM.__init__(self, 'eyes')
        self.__unblinkName = 'unblink'
        self.request('Open')

    def defaultFilter(self, request, args):
        if False:
            while True:
                i = 10
        if request[0].isupper():
            return request
        return None

    def enterOpen(self):
        if False:
            return 10
        print('swap in eyes open model')

    def filterOpen(self, request, args):
        if False:
            i = 10
            return i + 15
        if request == 'blink':
            taskMgr.remove(self.__unblinkName)
            taskMgr.doMethodLater(0.125, self.__unblink, self.__unblinkName)
            return 'Closed'
        return self.defaultFilter(request, args)

    def __unblink(self, task):
        if False:
            i = 10
            return i + 15
        self.request('unblink')
        return Task.done

    def enterClosed(self):
        if False:
            i = 10
            return i + 15
        print('swap in eyes closed model')

    def filterClosed(self, request, args):
        if False:
            for i in range(10):
                print('nop')
        if request == 'unblink':
            return 'Open'
        return self.defaultFilter(request, args)

    def enterSurprised(self):
        if False:
            return 10
        print('swap in eyes surprised model')

    def enterOff(self):
        if False:
            return 10
        taskMgr.remove(self.__unblinkName)