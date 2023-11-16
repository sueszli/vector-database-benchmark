from __future__ import print_function

def tryWhileExceptContinueTest():
    if False:
        i = 10
        return i + 15
    print('Check if continue is executed in a except handler using for loop:')
    global undefined
    x = 0
    while x < 10:
        x += 1
        try:
            if x % 2 == 1:
                undefined
        except:
            print(x, end=' ')
            continue
        print('-', end=' ')
    print()

def tryForExceptContinueTest():
    if False:
        return 10
    print('Check if continue is executed in a except handler using for loop:')
    for x in range(10):
        try:
            if x % 2 == 1:
                undefined
        except:
            print(x, end=' ')
            continue
        print('-', end=' ')
    print()

def tryWhileExceptBreakTest():
    if False:
        return 10
    print('Check if break is executed in a except handler using while loop:')
    x = 0
    while x < 10:
        x += 1
        try:
            if x == 5:
                undefined
        except:
            print(x, end=' ')
            break
        print('-', end=' ')
    print()

def tryForExceptBreakTest():
    if False:
        while True:
            i = 10
    print('Check if break is executed in a except handler using for loop:')
    for x in range(10):
        try:
            if x == 5:
                undefined
        except:
            print(x, end=' ')
            break
        print('-', end=' ')
    print()
tryWhileExceptContinueTest()
tryWhileExceptBreakTest()
tryForExceptContinueTest()
tryForExceptBreakTest()