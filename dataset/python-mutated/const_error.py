from micropython import const

def test_syntax(code):
    if False:
        print('Hello World!')
    try:
        exec(code)
    except SyntaxError:
        print('SyntaxError')
test_syntax('a = const(x)')
test_syntax('A = const(1); A = const(2)')
test_syntax('A = const(1 @ 2)')
test_syntax('A = const(1 / 2)')
test_syntax('A = const(1 ** -2)')
test_syntax('A = const(1 << -2)')
test_syntax('A = const(1 >> -2)')
test_syntax('A = const(1 % 0)')
test_syntax('A = const(1 // 0)')