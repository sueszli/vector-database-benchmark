class CtxMgr:

    def __init__(self, id):
        if False:
            i = 10
            return i + 15
        self.id = id

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        print('__enter__', self.id)
        if 10 <= self.id < 20:
            raise Exception('enter', self.id)
        return self

    def __exit__(self, a, b, c):
        if False:
            while True:
                i = 10
        print('__exit__', self.id, repr(a), repr(b))
        if 15 <= self.id < 25:
            raise Exception('exit', self.id)
try:
    with CtxMgr(1):
        pass
except Exception as e:
    print(e)
try:
    with CtxMgr(10):
        pass
except Exception as e:
    print(e)
try:
    with CtxMgr(15):
        pass
except Exception as e:
    print(e)
try:
    with CtxMgr(20):
        pass
except Exception as e:
    print(e)