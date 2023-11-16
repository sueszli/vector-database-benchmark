import sys

class MyClass(object):

    def __init__(self, name, identifier):
        if False:
            return 10
        self.name = name
        self.identifier = identifier

class MyClassSlot(object):
    __slot__ = ['name', 'identifier']

    def __init__(self, name, identifier):
        if False:
            print('Hello World!')
        self.name = name
        self.identifier = identifier