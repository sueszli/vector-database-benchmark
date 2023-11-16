""" Cover list contractions and a few special things in them.

"""
from __future__ import print_function

def displayDict(d):
    if False:
        return 10
    result = '{'
    first = True
    for (key, value) in sorted(d.items()):
        if not first:
            result += ','
        result += '%s: %s' % (repr(key), repr(value))
        first = False
    result += '}'
    return result
print('List contraction on the module level:')
x = [u if u % 2 == 0 else 0 for u in range(10)]
print(x)
print('List contraction on the function level:')

def someFunction():
    if False:
        while True:
            i = 10
    x = [u if u % 2 == 0 else 0 for u in range(10)]
    print(x)
someFunction()
print('List contractions with no, 1 one 2 conditions:')

def otherFunction():
    if False:
        print('Hello World!')
    print([x for x in range(8)])
    print([x for x in range(8) if x % 2 == 1])
    print([x for x in range(8) if x % 2 == 1 if x > 4])
otherFunction()
print('Complex list contractions with more than one for:')

def complexContractions():
    if False:
        for i in range(10):
            print('nop')
    print([(x, y) for x in range(3) for y in range(5)])
    seq = range(3)
    res = [(i, j, k) for i in iter(seq) for j in iter(seq) for k in iter(seq)]
    print(res)
complexContractions()
print('Contraction for 2 for statements and one final if referring to first for:')

def trickyContraction():
    if False:
        i = 10
        return i + 15

    class Range:

        def __init__(self, value):
            if False:
                while True:
                    i = 10
            self.value = value

        def __iter__(self):
            if False:
                while True:
                    i = 10
            print('Giving range iter to', self.value)
            return iter(range(self.value))

    def Cond(y):
        if False:
            print('Hello World!')
        print('Checking against', y)
        return y == 1
    r = [(x, z, y) for x in Range(3) for z in Range(2) for y in Range(4) if Cond(y)]
    print('result is', r)
trickyContraction()

def lambdaWithcontraction(x):
    if False:
        return 10
    l = lambda x: [z for z in range(x)]
    r = l(x)
    print('Lambda contraction locals:', displayDict(locals()))
lambdaWithcontraction(3)
print("Contraction that gets a 'del' on the iterator variable:", end=' ')

def allowedDelOnIteratorVariable(z):
    if False:
        while True:
            i = 10
    x = 2
    del x
    return [x * z for x in range(z)]
print(allowedDelOnIteratorVariable(3))