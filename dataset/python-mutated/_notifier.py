from twisted.internet.defer import Deferred

class Notifier:

    def __init__(self):
        if False:
            print('Hello World!')
        self._waiters = []

    def wait(self):
        if False:
            print('Hello World!')
        d = Deferred()
        self._waiters.append(d)
        return d

    def notify(self, result):
        if False:
            print('Hello World!')
        if self._waiters:
            (waiters, self._waiters) = (self._waiters, [])
            for waiter in waiters:
                waiter.callback(result)

    def __bool__(self):
        if False:
            print('Hello World!')
        return bool(self._waiters)