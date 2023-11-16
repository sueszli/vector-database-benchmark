from __future__ import print_function
import itertools

def calledRepeatedly():
    if False:
        print('Hello World!')
    return (lambda x: x)(7)
    return 7
for x in itertools.repeat(None, 50000):
    calledRepeatedly()
print('OK.')