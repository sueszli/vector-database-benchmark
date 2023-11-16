from abc import abstractmethod

class Base:

    def meth(self):
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def abstractmeth(self):
        if False:
            while True:
                i = 10
        pass

    @staticmethod
    @abstractmethod
    def staticmeth():
        if False:
            return 10
        pass

    @classmethod
    @abstractmethod
    def classmeth(cls):
        if False:
            return 10
        pass

    @property
    @abstractmethod
    def prop(self):
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    async def coroutinemeth(self):
        pass