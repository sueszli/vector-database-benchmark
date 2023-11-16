def test(code):
    if False:
        for i in range(10):
            print('nop')
    try:
        exec(code)
        print('no Error')
    except SyntaxError:
        print('SyntaxError')
    except NotImplementedError:
        print('NotImplementedError')
try:
    eval('1and 0')
except SyntaxError:
    print('SyntaxError')
try:
    eval('1or 0')
except SyntaxError:
    print('SyntaxError')
try:
    eval('1if 1else 0')
except SyntaxError:
    print('SyntaxError')
try:
    eval('1if 0else 0')
except SyntaxError:
    print('SyntaxError')
test('"\\N{LATIN SMALL LETTER A}"')