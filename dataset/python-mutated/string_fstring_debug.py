def f():
    if False:
        print('Hello World!')
    return 4

def g(_):
    if False:
        i = 10
        return i + 15
    return 5

def h():
    if False:
        print('Hello World!')
    return 6
(x, y) = (1, 2)
print(f'x={x!r}')
print(f'x={x:08x}')
print(f'a x={x!r} b {y} c')
print(f'a x={x:08x} b {y} c')
print(f"""a f() + g("foo") + h()={f() + g('foo') + h()!r} b""")
print(f"""a f() + g("foo") + h()={f() + g('foo') + h():08x} b""")
print(f'a 1,={(1,)!r} b')
print(f'a x,y,={(x, y)!r} b')
print(f'a x,1={(x, 1)!r} b')