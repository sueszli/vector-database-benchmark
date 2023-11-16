def Fun():
    if False:
        return 10
    pass

class A:

    def __init__(self):
        if False:
            return 10
        pass

    def Fun(self):
        if False:
            for i in range(10):
                print('nop')
        pass
try:
    print(Fun.__name__)
    print(A.__init__.__name__)
    print(A.Fun.__name__)
    print(A().Fun.__name__)
except AttributeError:
    print('SKIP')
    raise SystemExit
try:
    str(1 .to_bytes.__name__)
except AttributeError:
    pass

def outer():
    if False:
        while True:
            i = 10
    x = 1

    def inner():
        if False:
            return 10
        return x
    return inner
print(outer.__name__)
print(outer().__name__)