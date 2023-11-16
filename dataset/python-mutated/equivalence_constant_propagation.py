CONST_GLOBAL = 'secret'
GLOBAL = 'secret'

def foo():
    if False:
        print('Hello World!')
    bar(CONST_GLOBAL)

def bar():
    if False:
        for i in range(10):
            print('nop')
    global GLOBAL
    GLOBAL = 'x'