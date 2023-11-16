def _study1(i, n, ch):
    if False:
        for i in range(10):
            print('nop')
    while i == 3:
        i = 4
        if ch:
            i = 10
            assert i < 5
            continue
        if n:
            return n
assert _study1(3, 4, False) == 4