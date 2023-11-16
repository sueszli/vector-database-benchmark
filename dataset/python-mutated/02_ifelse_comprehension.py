def extend(stmt, a, c, c1, c2, c3):
    if False:
        i = 10
        return i + 15
    return c(([(5 if c1 else c2) if a else c3] for i in enumerate(stmt)))

def foo(gen):
    if False:
        for i in range(10):
            print('nop')
    return list(gen)
assert extend([0], 0, foo, True, 'c2', 'c3') == [['c3']]
assert extend([0, 1], 1, foo, False, 'c2', 'c3') == [['c2'], ['c2']]
assert extend([0, 1], False, foo, False, 'c2', 'c3') == [['c3'], ['c3']]