def tryContinueFinallyTest():
    if False:
        print('Hello World!')
    for x in range(10):
        try:
            if x % 2 == 1:
                continue
        finally:
            yield x
        yield '-'

def tryBreakFinallyTest():
    if False:
        return 10
    for x in range(10):
        try:
            if x == 5:
                break
        finally:
            yield x
        yield '-'

def tryFinallyAfterYield():
    if False:
        return 10
    try:
        yield 3
    finally:
        print('Executing finally')

def tryReturnFinallyYield():
    if False:
        print('Hello World!')
    try:
        return
    finally:
        yield 1

def tryReturnExceptYield():
    if False:
        return 10
    try:
        return
    except StopIteration:
        print('Caught StopIteration')
        yield 2
    except:
        yield 1
    else:
        print('No exception')

def tryStopIterationExceptYield():
    if False:
        for i in range(10):
            print('nop')
    try:
        raise StopIteration
    except StopIteration:
        print('Caught StopIteration')
        yield 2
    except:
        yield 1
    else:
        print('No exception')
print('Check if finally is executed in a continue using for loop:')
print(tuple(tryContinueFinallyTest()))
print('Check if finally is executed in a break using for loop:')
print(tuple(tryBreakFinallyTest()))
print('Check what try yield finally something does:')
print(tuple(tryFinallyAfterYield()))
print('Check if yield is executed in finally after return:')
print(tuple(tryReturnFinallyYield()))
print('Check if yield is executed in except after return:')
print(tuple(tryReturnExceptYield()))
print('Check if yield is executed in except after StopIteration:')
print(tuple(tryReturnExceptYield()))