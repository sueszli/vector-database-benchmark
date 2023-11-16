def ok(x):
    if False:
        for i in range(10):
            print('nop')
    y = x + 1
    sink(y)

def bad(x):
    if False:
        print('Hello World!')
    y = x + 'something'
    sink(y)