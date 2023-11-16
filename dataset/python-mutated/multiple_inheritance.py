"""
This example proves the following experience (under multiple inheritance circumstances)

1. The `Base.initialize` method is only called once with `super().initialize()`
"""

class Base:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        print('base init')

    def initialize(self):
        if False:
            while True:
                i = 10
        print('base initialize')

    def run(self):
        if False:
            return 10
        print('base run')

class Widget:

    def __init__(self, parent):
        if False:
            print('Hello World!')
        print('widget init')

class A(Base):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        print('A init')

    def initialize(self):
        if False:
            for i in range(10):
                print('nop')
        super().initialize()
        print('a initialize')

    def run(self):
        if False:
            while True:
                i = 10
        super().run()
        print('a run')

class B(Base, Widget):

    def __init__(self):
        if False:
            print('Hello World!')
        Base.__init__(self)
        Widget.__init__(self, self)
        print('B init')

    def initialize(self):
        if False:
            for i in range(10):
                print('nop')
        super().initialize()
        print('b initialize')

    def run(self):
        if False:
            i = 10
            return i + 15
        super().run()
        print('b run')

class M(A, B):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()

    def initialize(self):
        if False:
            i = 10
            return i + 15
        super().initialize()
        print('mixed initialize')

    def run(self):
        if False:
            while True:
                i = 10
        super().run()
        print('m run')
m = M()
m.initialize()
m.run()