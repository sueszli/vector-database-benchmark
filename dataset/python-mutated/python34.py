try:
    exec
except NameError:
    print('SKIP')
    raise SystemExit

def print_ret(x):
    if False:
        i = 10
        return i + 15
    print(x)
    return x
{print_ret(1): print_ret(2)}

def test_syntax(code):
    if False:
        print('Hello World!')
    try:
        exec(code)
    except SyntaxError:
        print('SyntaxError')
test_syntax('f(**a, b)')
test_syntax('() = []')
test_syntax('del ()')
import sys
print(sys.version[:3])
print(sys.version_info[0], sys.version_info[1])
print(repr(IndexError('foo')))