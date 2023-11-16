a = (1, 2, 3)
del a
b = [4, 5, 6]
del b[1]
del b[:]
l = [None] * 10
del l[-2:]
c = [0, 1, 2, 3, 4]
del c[:1]
del c[2:3]
d = [0, 1, 2, 3, 4, 5, 6]
del d[1:3:2]
e = ('a', 'b')

def foo():
    if False:
        print('Hello World!')
    global e
    del e
z = {}

def a():
    if False:
        return 10
    b = 1
    global z
    del z

    def b(y):
        if False:
            print('Hello World!')
        global z
        del y
        return z