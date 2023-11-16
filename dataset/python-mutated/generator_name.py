def Fun():
    if False:
        i = 10
        return i + 15
    yield

class A:

    def Fun(self):
        if False:
            i = 10
            return i + 15
        yield
try:
    print(Fun.__name__)
    print(A.Fun.__name__)
    print(A().Fun.__name__)
except AttributeError:
    print('SKIP')
    raise SystemExit