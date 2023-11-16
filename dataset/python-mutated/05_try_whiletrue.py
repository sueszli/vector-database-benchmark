def iter(self):
    if False:
        while True:
            i = 10
    i = 0
    try:
        while True:
            v = self[i]
            yield v
            i += 1
    except IndexError:
        return
A = [10, 20, 30]
assert list(iter(A)) == A