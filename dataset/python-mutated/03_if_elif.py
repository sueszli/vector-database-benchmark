if __name__:
    if __file__ and __name__:
        pass
    elif not __name__:
        assert False
if __name__:
    pass
elif __file__:
    assert __name__ or __file__
else:
    pass

def __floordiv__(a, b):
    if False:
        return 10
    if a:
        b += 1
    elif not b:
        return a
    b += 5
    return b
assert __floordiv__(1, 1) == 7
assert __floordiv__(1, 0) == 6
assert __floordiv__(0, 3) == 8
assert __floordiv__(0, 0) == 0