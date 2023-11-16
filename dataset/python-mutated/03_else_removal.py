def _cmp(b, c):
    if False:
        for i in range(10):
            print('nop')
    if b:
        if c:
            return 0
        else:
            return 1
    else:
        assert False, 'never get here'