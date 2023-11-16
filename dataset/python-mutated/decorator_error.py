def test_syntax(code):
    if False:
        for i in range(10):
            print('nop')
    try:
        exec(code)
    except SyntaxError:
        print('SyntaxError')
test_syntax('@micropython.a\ndef f(): pass')
test_syntax('@micropython.a.b\ndef f(): pass')