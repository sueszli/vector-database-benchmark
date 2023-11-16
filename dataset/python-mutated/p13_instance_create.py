"""
Topic: 通过元类控制实例的创建
Desc : 
"""

class NoInstances(type):

    def __call__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        raise TypeError("Can't instantiate directly")

class Spam(metaclass=NoInstances):

    @staticmethod
    def grok(x):
        if False:
            return 10
        print('Spam.grok')

class Singleton(type):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.__instance = None
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if self.__instance is None:
            self.__instance = super().__call__(*args, **kwargs)
            return self.__instance
        else:
            return self.__instance

class Spam1(metaclass=Singleton):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        print('Creating Spam')
import weakref

class Cached(type):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        self.__cache = weakref.WeakValueDictionary()

    def __call__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        if args in self.__cache:
            return self.__cache[args]
        else:
            obj = super().__call__(*args)
            self.__cache[args] = obj
            return obj

class Spam2(metaclass=Cached):

    def __init__(self, name):
        if False:
            while True:
                i = 10
        print('Creating Spam({!r})'.format(name))
        self.name = name