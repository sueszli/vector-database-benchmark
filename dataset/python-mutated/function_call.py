def test(x):
    if False:
        print('Hello World!')
    return x

def foo(x):
    if False:
        for i in range(10):
            print('nop')
    return test(x)
y = test(2)
z = (lambda x: x)(1)
bar = lambda x: x
bar(1)