import multiprocessing
from coalib.bears.Bear import Bear
from coalib.settings.Section import Section

class TestBear(Bear):

    def __init__(self):
        if False:
            return 10
        Bear.__init__(self, Section('settings'), multiprocessing.Queue())

    @staticmethod
    def kind():
        if False:
            i = 10
            return i + 15
        return 'kind'

    def origin(self):
        if False:
            while True:
                i = 10
        return __file__

class NoKind:

    def __init__(self):
        if False:
            while True:
                i = 10
        pass

    @staticmethod
    def kind():
        if False:
            print('Hello World!')
        raise NotImplementedError