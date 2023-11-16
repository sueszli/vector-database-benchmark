import time
from abc import ABCMeta, abstractmethod

class Event(metaclass=ABCMeta):

    def __init__(self, content):
        if False:
            while True:
                i = 10
        self.delay = content['delay']
        self.event_type = content['event_type']
        self.message = content['message']
        self.action = content['action']
        self.addon = content.get('addon')

    def __str__(self):
        if False:
            while True:
                i = 10
        if self.addon:
            return '[%d, %s, %s, %s, %s]' % (self.delay, self.event_type, self.message, self.action, str(self.addon))
        return '[%d, %s, %s, %s]' % (self.delay, self.event_type, self.message, self.action)

    def summarystr(self):
        if False:
            for i in range(10):
                print('nop')
        if self.event_type == 'EK':
            return 'key {0} {1} after {2}ms'.format(self.action[1], self.message[4:], self.delay)
        else:
            return '{0} after {1}ms'.format(self.message, self.delay)

    def sleep(self, thd=None):
        if False:
            while True:
                i = 10
        if thd:
            thd.sleep(self.delay)
        else:
            time.sleep(self.delay / 1000.0)

    @abstractmethod
    def execute(self, thd=None):
        if False:
            i = 10
            return i + 15
        pass