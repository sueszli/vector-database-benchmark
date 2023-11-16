def func(x):
    if False:
        print('Hello World!')
    b = x + 1
    return b + 2

def func2():
    if False:
        for i in range(10):
            print('nop')
    'Test function for issue 9936 '
    return (1, 2, 3)