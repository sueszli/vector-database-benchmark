def foo():
    if False:
        while True:
            i = 10
    f = open()
    f.write('bar')
    f.close()
    f.close()
    f = open()
    f.write('baz')
    f.close()

def bar():
    if False:
        return 10
    f = open()
    f.write('bar')
    f.close()
    f = open()
    f.close()