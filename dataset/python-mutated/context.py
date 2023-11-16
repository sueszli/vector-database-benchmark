"""
Provides some utility context guards.
"""

class DummyGuard:
    """ Context guard that does nothing. """

    @staticmethod
    def __enter__():
        if False:
            while True:
                i = 10
        pass

    @staticmethod
    def __exit__(exc_type, exc_value, traceback):
        if False:
            i = 10
            return i + 15
        pass