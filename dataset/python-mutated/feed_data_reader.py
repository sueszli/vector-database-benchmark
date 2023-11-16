from paddle import base
from paddle.base.framework import Variable

def cyclic_reader(reader):
    if False:
        while True:
            i = 10

    def __reader__():
        if False:
            print('Hello World!')
        while True:
            yield from reader()
    return __reader__

class FeedDataReader:

    def __init__(self, feed_list, reader):
        if False:
            return 10
        self._feed_list = []
        for var in feed_list:
            if isinstance(var, Variable):
                self._feed_list.append(var.name)
            else:
                self._feed_list.append(var)
        self._reader = cyclic_reader(reader)
        self._iter = self._reader()

    def _feed_executor(self):
        if False:
            i = 10
            return i + 15
        next_data = next(self._iter)
        feed_data = {}
        assert len(self._feed_list) == len(next_data)
        for (key, value) in zip(self._feed_list, next_data):
            feed_data[key] = value
        return feed_data

    def get_next(self, exe, program):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(exe, base.Executor), 'exe must be Executor'
        return self._feed_executor()