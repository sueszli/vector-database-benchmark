"""This module implements a minimum task manager.  It is similar in
principle to the full-featured task manager implemented in Task.py,
but it has a sharply reduced feature set--completely bare-bones, in
fact--and it is designed to be a pure-python implementation that does
not require any C++ Panda support, so that it can be loaded before
Panda has been fully downloaded. """
__all__ = ['MiniTask', 'MiniTaskManager']

class MiniTask:
    done = 0
    cont = 1
    name: str

    def __init__(self, callback):
        if False:
            print('Hello World!')
        self.__call__ = callback

class MiniTaskManager:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.taskList = []
        self.running = 0

    def add(self, task, name):
        if False:
            return 10
        assert isinstance(task, MiniTask)
        task.name = name
        self.taskList.append(task)

    def remove(self, task):
        if False:
            i = 10
            return i + 15
        try:
            self.taskList.remove(task)
        except ValueError:
            pass

    def step(self):
        if False:
            for i in range(10):
                print('nop')
        i = 0
        while i < len(self.taskList):
            task = self.taskList[i]
            ret = task(task)
            if ret == task.cont:
                pass
            else:
                try:
                    self.taskList.remove(task)
                except ValueError:
                    pass
                continue
            i += 1

    def run(self):
        if False:
            print('Hello World!')
        self.running = 1
        while self.running:
            self.step()

    def stop(self):
        if False:
            i = 10
            return i + 15
        self.running = 0