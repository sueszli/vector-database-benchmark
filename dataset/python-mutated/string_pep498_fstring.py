assert f'no interpolation' == 'no interpolation'
assert f'no interpolation' == 'no interpolation'
assert f'\\' == '\\'
assert f'#' == '#'
try:
    eval("f'{\\}'")
except SyntaxError:
    pass
else:
    raise AssertionError('f-string with backslash in expression did not raise SyntaxError')
try:
    eval("f'{#}'")
except SyntaxError:
    pass
else:
    raise AssertionError("f-string with '#' in expression did not raise SyntaxError")
assert f'{{}}' == '{}'
try:
    eval("f'{{}'")
except ValueError:
    pass
else:
    raise RuntimeError('Expected ValueError for invalid f-string literal bracing')
x = 1
assert f'{x}' == '1'

def foo():
    if False:
        return 10
    return 20
assert f'result={foo()}' == 'result=20'
assert f'result={foo()}' == 'result={}'.format(foo())
assert f'result={foo()}' == 'result={result}'.format(result=foo())
x = 10
y = 'hi'
assert f'{x:^4}' == ' 10 '
assert f'{{{4 * 10}}}' == '{40}'