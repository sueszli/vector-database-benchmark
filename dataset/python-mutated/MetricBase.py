from abc import ABC, abstractmethod

class MetricBase(ABC):

    def __init__(self, name):
        if False:
            i = 10
            return i + 15
        self.name = name
        self.start = None
        self.end = None

    @abstractmethod
    def record_start(self):
        if False:
            for i in range(10):
                print('nop')
        return

    @abstractmethod
    def record_end(self):
        if False:
            i = 10
            return i + 15
        return

    @abstractmethod
    def elapsed_time(self):
        if False:
            return 10
        return

    def get_name(self):
        if False:
            return 10
        return self.name

    def get_end(self):
        if False:
            for i in range(10):
                print('nop')
        return self.end