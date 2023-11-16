"""This program is self-checking!"""
f"{f'3.1415={3.1415:.1f}':*^20}" == '*****3.1415=3.1*****'
y = 2

def f(x, width):
    if False:
        print('Hello World!')
    return f'x={x * y:{width}}'
assert f('foo', 10) == 'x=foofoo    '
x = 'bar'
assert f(10, 10), 'x=        20'
x = 'A string'
f'x={x!r}' == 'x=' + repr(x)
pi = 'π'
assert f'alpha α pi={pi!r} ω omega', "alpha α pi='π' ω omega"
x = 20
assert f'{x:=10}' == '        20'
assert f'{(x := 10)}' == '10'
assert x == 10