class X:

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'X'
x = X()

@micropython.viper
def a():
    if False:
        while True:
            i = 10
    x.i0 = 0
    x.i7 = 7
    x.s = 'hello'
    x.o = x
a()
print(x.i0)
print(x.i7)
print(x.s)
print(x.o)