class CtxMgr:

    def __enter__(self):
        if False:
            while True:
                i = 10
        print('__enter__')
        return self

    def __exit__(self, a, b, c):
        if False:
            i = 10
            return i + 15
        print('__exit__', repr(a), repr(b))
for i in range(5):
    print(i)
    with CtxMgr():
        if i == 3:
            continue