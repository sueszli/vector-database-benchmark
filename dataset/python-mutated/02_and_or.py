def are_instructions_equal(a, b, c, d):
    if False:
        for i in range(10):
            print('nop')
    return a and (b or c) and d
for (a, b, c, d, expect) in ((True, True, False, True, True), (True, False, True, True, True), (False, False, True, True, False), (True, False, True, False, False)):
    assert are_instructions_equal(a, b, c, d) == expect

def n_alias(a, b, c, d=3):
    if False:
        while True:
            i = 10
    if a and b or c:
        d = 1
    else:
        d = 2
    return d
for (a, b, c, expect) in ((True, True, False, 1), (True, False, True, 1), (False, False, False, 2)):
    assert n_alias(a, b, c) == expect, f'{a}, {b}, {c}, {expect}'