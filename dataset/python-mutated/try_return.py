def f():
    if False:
        print('Hello World!')
    try:
        print(1)
        return
    except:
        print(2)
    print(3)
f()

def f(l, i):
    if False:
        print('Hello World!')
    try:
        return l[i]
    except IndexError:
        print('IndexError')
        return -1
print(f([1], 0))
print(f([], 0))