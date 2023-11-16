print('Yep, I got imported.')
try:
    x = const(1)
except NameError:
    const = lambda x: x
_CNT01 = 'CONST01'
_CNT02 = const(123)
A123 = const(123)
a123 = const(123)

def dummy():
    if False:
        i = 10
        return i + 15
    return False

def saysomething():
    if False:
        i = 10
        return i + 15
    print('There, I said it.')

def neverexecuted():
    if False:
        print('Hello World!')
    print('Never got here!')
print('Yep, got here')