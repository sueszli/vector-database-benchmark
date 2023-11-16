"""Check for else branches on loops with break and return only."""

def test_return_for():
    if False:
        i = 10
        return i + 15
    'else + return is not acceptable.'
    for i in range(10):
        if i % 2:
            return i
    else:
        print('math is broken')
    return None

def test_return_while():
    if False:
        print('Hello World!')
    'else + return is not acceptable.'
    while True:
        return 1
    else:
        print('math is broken')
    return None
while True:

    def short_fun():
        if False:
            return 10
        'A function with a loop.'
        for _ in range(10):
            break
else:
    print('or else!')
while True:
    while False:
        break
else:
    print('or else!')
for j in range(10):
    pass
else:
    print('fat chance')
    for j in range(10):
        break

def test_return_for2():
    if False:
        print('Hello World!')
    'no false positive for break in else\n\n    https://bitbucket.org/logilab/pylint/issue/117/useless-else-on-loop-false-positives\n    '
    for i in range(10):
        for _ in range(i):
            if i % 2:
                break
        else:
            break
    else:
        print('great math')

def test_break_in_orelse_deep():
    if False:
        i = 10
        return i + 15
    'no false positive for break in else deeply nested'
    for _ in range(10):
        if 1 < 2:
            for _ in range(3):
                if 3 < 2:
                    break
            else:
                break
    else:
        return True
    return False

def test_break_in_orelse_deep2():
    if False:
        while True:
            i = 10
    'should raise a useless-else-on-loop message, as the break statement is only\n    for the inner for loop\n    '
    for _ in range(10):
        if 1 < 2:
            for _ in range(3):
                if 3 < 2:
                    break
            else:
                print('all right')
    else:
        return True
    return False

def test_break_in_orelse_deep3():
    if False:
        return 10
    'no false positive for break deeply nested in else'
    for _ in range(10):
        for _ in range(3):
            pass
        else:
            if 1 < 2:
                break
    else:
        return True
    return False

def test_break_in_if_orelse():
    if False:
        while True:
            i = 10
    'should raise a useless-else-on-loop message due to break in else'
    for _ in range(10):
        if 1 < 2:
            pass
        else:
            break
    else:
        return True
    return False

def test_break_in_with():
    if False:
        print('Hello World!')
    'no false positive for break in with'
    for name in ['demo']:
        with open(__file__) as f:
            if name in f.read():
                break
    else:
        return True
    return False

def test_break_in_match():
    if False:
        for i in range(10):
            print('nop')
    'no false positive for break in match'
    for name in ['demo']:
        match name:
            case 'demo':
                break
    else:
        return True
    return False