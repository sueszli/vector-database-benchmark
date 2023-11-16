from __future__ import print_function

def tryWhileContinueFinallyTest():
    if False:
        i = 10
        return i + 15
    print('Check if finally is executed in a continue using for loop:')
    x = 0
    while x < 10:
        x += 1
        try:
            if x % 2 == 1:
                continue
        finally:
            print(x, end=' ')
        print('-', end=' ')
    print()

def tryForContinueFinallyTest():
    if False:
        while True:
            i = 10
    print('Check if finally is executed in a continue using for loop:')
    for x in range(10):
        try:
            if x % 2 == 1:
                continue
        finally:
            print(x, end=' ')
        print('-', end=' ')
    print()

def tryWhileBreakFinallyTest():
    if False:
        print('Hello World!')
    print('Check if finally is executed in a break using while loop:')
    x = 0
    while x < 10:
        x += 1
        try:
            if x == 5:
                break
        finally:
            print(x, end=' ')
        print('-', end=' ')
    print()

def tryForBreakFinallyTest():
    if False:
        i = 10
        return i + 15
    print('Check if finally is executed in a break using for loop:')
    for x in range(10):
        try:
            if x == 5:
                break
        finally:
            print(x, end=' ')
        print('-', end=' ')
    print()
tryWhileContinueFinallyTest()
tryWhileBreakFinallyTest()
tryForContinueFinallyTest()
tryForBreakFinallyTest()