import import1b

def func1():
    if False:
        for i in range(10):
            print('nop')
    print('func1')

def func2():
    if False:
        i = 10
        return i + 15
    try:
        import1b.throw()
    except ValueError:
        pass
    func1()
func2()