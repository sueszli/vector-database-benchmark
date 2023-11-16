"""Defines the DirectObject class, a convenient class to inherit from if the
object needs to be able to respond to events."""
__all__ = ['DirectObject']
from direct.directnotify.DirectNotifyGlobal import directNotify
from direct.task.TaskManagerGlobal import taskMgr
from .MessengerGlobal import messenger

class DirectObject:
    """
    This is the class that all Direct/SAL classes should inherit from
    """

    def accept(self, event, method, extraArgs=[]):
        if False:
            return 10
        return messenger.accept(event, self, method, extraArgs, 1)

    def acceptOnce(self, event, method, extraArgs=[]):
        if False:
            for i in range(10):
                print('nop')
        return messenger.accept(event, self, method, extraArgs, 0)

    def ignore(self, event):
        if False:
            print('Hello World!')
        return messenger.ignore(event, self)

    def ignoreAll(self):
        if False:
            i = 10
            return i + 15
        return messenger.ignoreAll(self)

    def isAccepting(self, event):
        if False:
            return 10
        return messenger.isAccepting(event, self)

    def getAllAccepting(self):
        if False:
            return 10
        return messenger.getAllAccepting(self)

    def isIgnoring(self, event):
        if False:
            print('Hello World!')
        return messenger.isIgnoring(event, self)

    def addTask(self, *args, **kwargs):
        if False:
            return 10
        if not hasattr(self, '_taskList'):
            self._taskList = {}
        kwargs['owner'] = self
        task = taskMgr.add(*args, **kwargs)
        return task

    def doMethodLater(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if not hasattr(self, '_taskList'):
            self._taskList = {}
        kwargs['owner'] = self
        task = taskMgr.doMethodLater(*args, **kwargs)
        return task

    def removeTask(self, taskOrName):
        if False:
            print('Hello World!')
        if isinstance(taskOrName, str):
            if hasattr(self, '_taskList'):
                taskListValues = list(self._taskList.values())
                for task in taskListValues:
                    if task.name == taskOrName:
                        task.remove()
        else:
            taskOrName.remove()

    def removeAllTasks(self):
        if False:
            print('Hello World!')
        if hasattr(self, '_taskList'):
            for task in list(self._taskList.values()):
                task.remove()

    def _addTask(self, task):
        if False:
            return 10
        self._taskList[task.id] = task

    def _clearTask(self, task):
        if False:
            for i in range(10):
                print('nop')
        del self._taskList[task.id]

    def detectLeaks(self):
        if False:
            return 10
        if not __dev__:
            return
        events = messenger.getAllAccepting(self)
        tasks = []
        if hasattr(self, '_taskList'):
            tasks = [task.name for task in self._taskList.values()]
        if len(events) != 0 or len(tasks) != 0:
            from direct.showbase.PythonUtil import getRepository
            estr = 'listening to events: %s' % events if len(events) != 0 else ''
            andStr = ' and ' if len(events) != 0 and len(tasks) != 0 else ''
            tstr = '%srunning tasks: %s' % (andStr, tasks) if len(tasks) != 0 else ''
            notify = directNotify.newCategory('LeakDetect')
            crash = getattr(getRepository(), '_crashOnProactiveLeakDetect', False)
            func = self.notify.error if crash else self.notify.warning
            func('destroyed %s instance is still %s%s' % (self.__class__.__name__, estr, tstr))
    add_task = addTask
    do_method_later = doMethodLater
    detect_leaks = detectLeaks
    accept_once = acceptOnce
    ignore_all = ignoreAll
    get_all_accepting = getAllAccepting
    is_ignoring = isIgnoring
    remove_all_tasks = removeAllTasks
    remove_task = removeTask
    is_accepting = isAccepting