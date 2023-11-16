from robot.model import ItemList, Message
from robot.utils import setter

class ExecutionErrors:
    """Represents errors occurred during the execution of tests.

    An error might be, for example, that importing a library has failed.
    """
    id = 'errors'

    def __init__(self, messages=None):
        if False:
            for i in range(10):
                print('nop')
        self.messages = messages

    @setter
    def messages(self, messages):
        if False:
            print('Hello World!')
        return ItemList(Message, {'parent': self}, items=messages)

    def add(self, other):
        if False:
            for i in range(10):
                print('nop')
        self.messages.extend(other.messages)

    def visit(self, visitor):
        if False:
            return 10
        visitor.visit_errors(self)

    def __iter__(self):
        if False:
            return 10
        return iter(self.messages)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.messages)

    def __getitem__(self, index):
        if False:
            print('Hello World!')
        return self.messages[index]

    def __str__(self):
        if False:
            while True:
                i = 10
        if not self:
            return 'No execution errors'
        if len(self) == 1:
            return f'Execution error: {self[0]}'
        return '\n'.join(['Execution errors:'] + ['- ' + str(m) for m in self])