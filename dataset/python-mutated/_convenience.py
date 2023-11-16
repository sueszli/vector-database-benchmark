"""
Common functionality used within the implementation of various workers.
"""
from ._ithreads import AlreadyQuit

class Quit:
    """
    A flag representing whether a worker has been quit.

    @ivar isSet: Whether this flag is set.
    @type isSet: L{bool}
    """

    def __init__(self):
        if False:
            return 10
        '\n        Create a L{Quit} un-set.\n        '
        self.isSet = False

    def set(self):
        if False:
            i = 10
            return i + 15
        '\n        Set the flag if it has not been set.\n\n        @raise AlreadyQuit: If it has been set.\n        '
        self.check()
        self.isSet = True

    def check(self):
        if False:
            return 10
        '\n        Check if the flag has been set.\n\n        @raise AlreadyQuit: If it has been set.\n        '
        if self.isSet:
            raise AlreadyQuit()