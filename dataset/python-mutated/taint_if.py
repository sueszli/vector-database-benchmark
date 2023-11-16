def test1(cond):
    if False:
        i = 10
        return i + 15
    if cond:
        x = 'tainted'
    else:
        x = 'safe'
    sink(x)

def test2(cond):
    if False:
        print('Hello World!')
    x = 'tainted'
    if cond:
        x = sanitize(x)
    sink(x)
    x = sanitize(x)
    sink(x)

def test3(cond):
    if False:
        while True:
            i = 10
    x = 'tainted'
    if cond:
        sink(x)
        y = sanitize(x)
    else:
        y = sanitize(x)
    sink(x)
    sink(y)