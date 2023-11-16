from typing import List
from trashcli.put.core.int_generator import IntGenerator

class FakeRandomInt(IntGenerator):

    def __init__(self, values):
        if False:
            print('Hello World!')
        self.values = values

    def new_int(self, _a, _b):
        if False:
            print('Hello World!')
        return self.values.pop(0)

    def set_reply(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.values = [value]