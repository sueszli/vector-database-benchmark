def fun1(val=5):
    if False:
        while True:
            i = 10
    print(val)
fun1()
fun1(10)

def fun2(p1, p2=100, p3='foo'):
    if False:
        for i in range(10):
            print('nop')
    print(p1, p2, p3)
fun2(1)
fun2(1, None)
fun2(0, 'bar', 200)
try:
    fun2()
except TypeError:
    print('TypeError')
try:
    fun2(1, 2, 3, 4)
except TypeError:
    print('TypeError')

def f(x=lambda : 1):
    if False:
        for i in range(10):
            print('nop')
    return x()
print(f())
print(f(f))
print(f(lambda : 2))