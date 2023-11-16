import abc
import time

class ModelBase(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def dict_repr(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns dictionary representation of the current instance'
        raise NotImplementedError

class BasicModel(ModelBase):
    TYPE: str

    def __init__(self, type_str_repr, cliid, sessid):
        if False:
            for i in range(10):
                print('nop')
        self.type = type_str_repr
        self.timestamp = time.time()
        self.cliid = cliid
        self.sessid = sessid

    def dict_repr(self):
        if False:
            i = 10
            return i + 15
        return self.__dict__