def testit(b):
    if False:
        return 10
    if b == 1:
        a = 1
    elif b == 2:
        a = 2
    else:
        a = 4
    return a
for x in (1, 2, 4):
    x = testit(x)
    assert x is not None, 'Should have returned a value, not None'
    assert x == x