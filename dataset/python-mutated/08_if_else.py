def init(modules=None):
    if False:
        i = 10
        return i + 15
    mods = set() if not modules else set(modules)
    return mods
assert init() == set()
assert init([1, 2, 3]) == set([1, 2, 3])

def _escape(a, b, c, d, e):
    if False:
        for i in range(10):
            print('nop')
    if a:
        if b:
            if c:
                if d:
                    raise
                return
        if e:
            if d:
                raise
            return
        raise
assert _escape(False, True, True, True, True) is None
assert _escape(True, True, True, False, True) is None
assert _escape(True, True, False, False, True) is None
for args in ((True, True, True, False, True), (True, False, True, True, True), (True, False, True, True, False)):
    try:
        _escape(*args)
        assert False, args
    except:
        pass