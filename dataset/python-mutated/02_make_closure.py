"""This program is self-checking!"""
a = 5

class MakeClosureTest:

    def __init__(self, dev: str, b: bool):
        if False:
            return 10
        super().__init__()
        self.dev = dev
        self.b = b
        self.a = a
x = MakeClosureTest('dev', True)
assert x.dev == 'dev'
assert x.b == True
assert x.a == 5