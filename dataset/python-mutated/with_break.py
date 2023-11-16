class CtxMgr:

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        print('__enter__')
        return self

    def __exit__(self, a, b, c):
        if False:
            return 10
        print('__exit__', repr(a), repr(b))
for i in range(5):
    print(i)
    with CtxMgr():
        if i == 3:
            break