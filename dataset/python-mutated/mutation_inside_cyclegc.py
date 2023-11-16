import weakref

class A(object):
    pass

def callback(x):
    if False:
        for i in range(10):
            print('nop')
    del lst[:]
keepalive = []
for i in range(100):
    lst = [str(i)]
    a = A()
    a.cycle = a
    keepalive.append(weakref.ref(a, callback))
    del a
    while lst:
        keepalive.append(lst[:])