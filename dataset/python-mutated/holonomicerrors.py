""" Common Exceptions for `holonomic` module. """

class BaseHolonomicError(Exception):

    def new(self, *args):
        if False:
            while True:
                i = 10
        raise NotImplementedError('abstract base class')

class NotPowerSeriesError(BaseHolonomicError):

    def __init__(self, holonomic, x0):
        if False:
            print('Hello World!')
        self.holonomic = holonomic
        self.x0 = x0

    def __str__(self):
        if False:
            print('Hello World!')
        s = 'A Power Series does not exists for '
        s += str(self.holonomic)
        s += ' about %s.' % self.x0
        return s

class NotHolonomicError(BaseHolonomicError):

    def __init__(self, m):
        if False:
            print('Hello World!')
        self.m = m

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.m

class SingularityError(BaseHolonomicError):

    def __init__(self, holonomic, x0):
        if False:
            for i in range(10):
                print('nop')
        self.holonomic = holonomic
        self.x0 = x0

    def __str__(self):
        if False:
            while True:
                i = 10
        s = str(self.holonomic)
        s += ' has a singularity at %s.' % self.x0
        return s

class NotHyperSeriesError(BaseHolonomicError):

    def __init__(self, holonomic, x0):
        if False:
            while True:
                i = 10
        self.holonomic = holonomic
        self.x0 = x0

    def __str__(self):
        if False:
            return 10
        s = 'Power series expansion of '
        s += str(self.holonomic)
        s += ' about %s is not hypergeometric' % self.x0
        return s