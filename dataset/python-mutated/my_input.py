from abc import abstractmethod, ABCMeta
import six
from six.moves import input as _my_input

@six.add_metaclass(ABCMeta)
class Input:

    @abstractmethod
    def read_input(self, prompt):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

class RealInput(Input):

    def read_input(self, prompt):
        if False:
            print('Hello World!')
        return _my_input(prompt)

class HardCodedInput(Input):

    def __init__(self, reply=None):
        if False:
            return 10
        (self.reply, self.exception) = self._reply(reply)

    def set_reply(self, reply):
        if False:
            return 10
        (self.reply, self.exception) = self._reply(reply)

    def _reply(self, reply):
        if False:
            for i in range(10):
                print('nop')
        if reply is None:
            return (None, ValueError('No reply set'))
        else:
            return (reply, None)

    def raise_exception(self, exception):
        if False:
            while True:
                i = 10
        self.exception = exception

    def read_input(self, prompt):
        if False:
            i = 10
            return i + 15
        self.used_prompt = prompt
        if self.exception:
            raise self.exception
        return self.reply

    def last_prompt(self):
        if False:
            print('Hello World!')
        return self.used_prompt