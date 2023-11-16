"""Record counting support for showing progress of revision fetch."""
from __future__ import absolute_import

class RecordCounter(object):
    """Container for maintains estimates of work requires for fetch.

    Instance of this class is used along with a progress bar to provide
    the user an estimate of the amount of work pending for a fetch (push,
    pull, branch, checkout) operation.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self.initialized = False
        self.current = 0
        self.key_count = 0
        self.max = 0
        self.STEP = 7

    def is_initialized(self):
        if False:
            return 10
        return self.initialized

    def _estimate_max(self, key_count):
        if False:
            for i in range(10):
                print('nop')
        "Estimate the maximum amount of 'inserting stream' work.\n\n        This is just an estimate.\n        "
        return int(key_count * 10.3)

    def setup(self, key_count, current=0):
        if False:
            i = 10
            return i + 15
        'Setup RecordCounter with basic estimate of work pending.\n\n        Setup self.max and self.current to reflect the amount of work\n        pending for a fetch.\n        '
        self.current = current
        self.key_count = key_count
        self.max = self._estimate_max(key_count)
        self.initialized = True

    def increment(self, count):
        if False:
            for i in range(10):
                print('nop')
        'Increment self.current by count.\n\n        Apart from incrementing self.current by count, also ensure\n        that self.max > self.current.\n        '
        self.current += count
        if self.current > self.max:
            self.max += self.key_count