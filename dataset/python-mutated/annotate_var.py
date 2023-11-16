x: int
print('x' in globals())
x: int = 1
print(x)
t: tuple = (1, 2)
print(t)

def f():
    if False:
        i = 10
        return i + 15
    x: int
    try:
        print(x)
    except NameError:
        print('NameError')
f()

def f():
    if False:
        for i in range(10):
            print('nop')
    x.y: int
    print(x)
f()