import sys
import os

class WritelnDecorator:
    """Used to decorate file-like objects with a handy 'writeln' method"""

    def __init__(self, stream):
        if False:
            while True:
                i = 10
        self.stream = stream

    def __getattr__(self, attr):
        if False:
            print('Hello World!')
        return getattr(self.stream, attr)

    def writeln(self, arg=None):
        if False:
            for i in range(10):
                print('nop')
        if arg:
            self.write(arg)
        self.write('\n')