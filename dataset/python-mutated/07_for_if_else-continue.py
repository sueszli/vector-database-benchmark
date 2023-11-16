"""This program is self-checking!"""

def test1(a, r=[]):
    if False:
        while True:
            i = 10
    for b in a:
        if b:
            r.append(3)
        else:
            r.append(5)
            continue
        if r == []:
            pass
    return r

def test2(a, r=None):
    if False:
        return 10
    for b in a:
        if b:
            continue
            raise AssertionError('CONTINUE not followed')
        else:
            continue
            raise AssertionError('CONTINUE not followed')
        if b:
            r = b
        raise AssertionError('CONTINUE not followed')
    return r
assert test1([], []) == [], 'For loop not taken'
assert test1([False], []) == [5], "if 'else' should have been taken"
assert test1([True], []) == [3], "if 'then' should have been taken"
assert test1([True, True], []) == [3, 3], 'if should have been taken'
assert test1([True, False], []) == [3, 5], "if and then 'else' should have been taken"
assert test1([False, True], []) == [5, 3], "if else and then 'then' should have been taken"
assert test1([False, False], []) == [5, 5], 'if else should have been taken twice'
assert test1([True, True], []) == [3, 3], "if 'then' should have been taken twice"
assert test2([True]) is None, 'Incorrect flow'
assert test2([False]) is None, 'Incorrect flow'