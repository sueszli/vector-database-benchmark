"""A utility class for reporting processing progress."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import datetime

class Progress(object):
    """A utility class for reporting processing progress."""

    def __init__(self, target_size):
        if False:
            for i in range(10):
                print('nop')
        self.target_size = target_size
        self.current_size = 0
        self.start_time = datetime.datetime.now()

    def Update(self, current_size):
        if False:
            return 10
        'Replaces internal current_size with current_size.'
        self.current_size = current_size

    def Add(self, size):
        if False:
            while True:
                i = 10
        'Increments internal current_size by size.'
        self.current_size += size

    def __str__(self):
        if False:
            print('Hello World!')
        processed = 1e-05 + self.current_size / float(self.target_size)
        current_time = datetime.datetime.now()
        elapsed = current_time - self.start_time
        eta = datetime.timedelta(seconds=elapsed.total_seconds() / processed - elapsed.total_seconds())
        return '%d / %d (elapsed %s eta %s)' % (self.current_size, self.target_size, str(elapsed).split('.')[0], str(eta).split('.')[0])