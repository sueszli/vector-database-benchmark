from abc import ABCMeta

class Class:

    def __init_subclass__(self, default_name, **kwargs):
        if False:
            return 10
        ...

    @classmethod
    def badAllowed(self, x, /, other):
        if False:
            i = 10
            return i + 15
        ...

    @classmethod
    def stillBad(self, x, /, other):
        if False:
            for i in range(10):
                print('nop')
        ...

class MetaClass(ABCMeta):

    def badAllowed(self):
        if False:
            while True:
                i = 10
        pass

    def stillBad(self):
        if False:
            print('Hello World!')
        pass