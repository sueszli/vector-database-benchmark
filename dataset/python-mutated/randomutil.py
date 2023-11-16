from __future__ import absolute_import
import random
from six.moves import range

def get_random_numbers(count):
    if False:
        return 10
    return [random.randrange(0, 1000) for idx in range(0, count)]