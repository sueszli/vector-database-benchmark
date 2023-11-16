from abc import ABC, abstractmethod

class _StreamBase(ABC):
    """Base stream class abstraction for multi backends Stream to herit from"""

    @abstractmethod
    def wait_event(self, event):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    @abstractmethod
    def wait_stream(self, stream):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    @abstractmethod
    def record_event(self, event=None):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    @abstractmethod
    def query(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    @abstractmethod
    def synchronize(self):
        if False:
            return 10
        raise NotImplementedError()

    @abstractmethod
    def __eq__(self, stream):
        if False:
            return 10
        raise NotImplementedError()

class _EventBase(ABC):
    """Base Event class abstraction for multi backends Event to herit from"""

    @abstractmethod
    def wait(self, stream=None):
        if False:
            print('Hello World!')
        raise NotImplementedError()

    @abstractmethod
    def query(self):
        if False:
            return 10
        raise NotImplementedError()

    @abstractmethod
    def synchronize(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError()