from abc import ABC, abstractmethod

class SharedDict(ABC):

    @abstractmethod
    def get_int(self, key):
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def set_int(self, key, value):
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def get_str(self, key):
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def set_str(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        pass