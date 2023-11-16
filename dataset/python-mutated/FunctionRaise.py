from __future__ import print_function
import itertools
module_var = None

def raisy():
    if False:
        while True:
            i = 10
    raise TypeError

def calledRepeatedly(raisy):
    if False:
        print('Hello World!')
    module_var
    try:
        raisy()
    except TypeError:
        pass
    pass
for x in itertools.repeat(None, 50000):
    calledRepeatedly(raisy)
print('OK.')