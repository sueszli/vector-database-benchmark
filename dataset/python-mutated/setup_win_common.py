"""A module for reading the information common to all Windows setups.

Exports read and get_definitions.
"""
import os
PATH = os.path.join('buildconfig', 'Setup_Win_Common.in')

class Definition:

    def __init__(self, name, value):
        if False:
            while True:
                i = 10
        self.name = name
        self.value = value

def read():
    if False:
        for i in range(10):
            print('nop')
    'Return the contents of the Windows Common Setup as a string'
    with open(PATH) as setup_in:
        return setup_in.read()

def get_definitions():
    if False:
        return 10
    "Return a list of definitions in the Windows Common Setup\n\n    Each macro definition object has a 'name' and 'value' attribute.\n    "
    import re
    deps = []
    match = re.compile('([a-zA-Z0-9_]+) += +(.+)$').match
    with open(PATH) as setup_in:
        for line in setup_in:
            m = match(line)
            if m is not None:
                deps.append(Definition(m.group(1), m.group(2)))
    return deps
__all__ = ['read', 'get_definitions']