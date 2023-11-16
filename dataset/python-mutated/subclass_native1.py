class mylist(list):
    pass
a = mylist([1, 2, 5])
a.attr = 'something'
print(a)
print(a.attr)
print(a[-1])
a[0] = -1
print(a)
print(len(a))
print(a + [20, 30, 40])

def foo():
    if False:
        print('Hello World!')
    print('hello from foo')
try:

    class myfunc(type(foo)):
        pass
except TypeError:
    print('TypeError')
try:

    class A(type, tuple):
        None
except TypeError:
    print('TypeError')