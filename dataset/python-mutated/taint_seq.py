def test1():
    if False:
        print('Hello World!')
    sink('tainted')

def test2():
    if False:
        return 10
    x = 'safe'
    a = x
    x = 'tainted'
    y = x
    z = y
    sink(z)
    sink(a)
    safe(z)

def test3():
    if False:
        print('Hello World!')
    sink(sanitize('tainted'))

def test4():
    if False:
        for i in range(10):
            print('nop')
    x = 'tainted'
    x = sanitize(x)
    sink(x)