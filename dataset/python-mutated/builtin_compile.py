try:
    compile
except NameError:
    print('SKIP')
    raise SystemExit

def test():
    if False:
        return 10
    global x
    c = compile('print(x)', 'file', 'exec')
    try:
        exec(c)
    except NameError:
        print('NameError')
    x = 1
    exec(c)
    exec(c, {'x': 2})
    exec(c, {}, {'x': 3})
    exec(compile('if 1: 10 + 1\n', 'file', 'single'))
    exec(compile('print(10 + 2)', 'file', 'single'))
    print(eval(compile('10 + 3', 'file', 'eval')))
    try:
        compile('1', 'file', '')
    except ValueError:
        print('ValueError')
    try:
        exec(compile('noexist', 'file', 'exec'))
    except NameError:
        print('NameError')
    print(x)
    print(type(hash(compile('', '', 'exec'))))
test()