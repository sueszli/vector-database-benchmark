"""
Testing helpers related to the module system.
"""
__all__ = ['NoReactor', 'AlternateReactor']
import sys
import twisted.internet
from twisted.test.test_twisted import SetAsideModule

class NoReactor(SetAsideModule):
    """
    Context manager that uninstalls the reactor, if any, and then restores it
    afterwards.
    """

    def __init__(self):
        if False:
            return 10
        SetAsideModule.__init__(self, 'twisted.internet.reactor')

    def __enter__(self):
        if False:
            print('Hello World!')
        SetAsideModule.__enter__(self)
        if 'twisted.internet.reactor' in self.modules:
            del twisted.internet.reactor

    def __exit__(self, excType, excValue, traceback):
        if False:
            while True:
                i = 10
        SetAsideModule.__exit__(self, excType, excValue, traceback)
        reactor = self.modules.get('twisted.internet.reactor', None)
        if reactor is not None:
            twisted.internet.reactor = reactor
        else:
            try:
                del twisted.internet.reactor
            except AttributeError:
                pass

class AlternateReactor(NoReactor):
    """
    A context manager which temporarily installs a different object as the
    global reactor.
    """

    def __init__(self, reactor):
        if False:
            i = 10
            return i + 15
        '\n        @param reactor: Any object to install as the global reactor.\n        '
        NoReactor.__init__(self)
        self.alternate = reactor

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        NoReactor.__enter__(self)
        twisted.internet.reactor = self.alternate
        sys.modules['twisted.internet.reactor'] = self.alternate