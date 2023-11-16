def f():
    if False:
        i = 10
        return i + 15
    return 4

def g(_):
    if False:
        i = 10
        return i + 15
    return 5

def h():
    if False:
        while True:
            i = 10
    return 6
print(f'no interpolation')
print(f'no interpolation')
print(f'no interpolation')
(x, y) = (1, 2)
print(f'{x}')
print(f'{x:08x}')
print(f'a {x} b {y} c')
print(f'a {x:08x} b {y} c')
print(f"a {'hello'} b")
print(f"a {f() + g('foo') + h()} b")

def foo(a, b):
    if False:
        for i in range(10):
            print('nop')
    return f'{x}{y}{a}{b}'
print(foo(7, 8))
print(f'a{[0, 1, 2][0:2]}')
print(f'a{[0, 15, 2][0:2][-1]:04x}')
print(f'a{ {0, 1, 2}}')
print(f'\\')
print(f'#')
try:
    eval("f'{\\}'")
except SyntaxError:
    print('SyntaxError')
try:
    eval("f'{#}'")
except SyntaxError:
    print('SyntaxError')
print(f'{{}}')
print(f'{{{4 * 10}}}', '{40}')
try:
    eval("f'{{}'")
except (ValueError, SyntaxError):
    print('SyntaxError')
print(f'a {(1,)} b')
print(f'a {(x, y)} b')
print(f'a {(x, 1)} b')
a = '123'
print(f'{a!r}')
print(f'{a!s}')
try:
    eval('print(f"{a!x}")')
except (ValueError, SyntaxError):
    print('ValueError')
print(f'{a!r:8s}')
print(f'{a!s:8s}')
print(f"{('1' if a != '456' else '0')!r:8s}")
print(f"{('1' if a != '456' else '0')!s:8s}")