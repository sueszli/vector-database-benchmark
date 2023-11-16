x = 1
print(x)
del x
try:
    print(x)
except NameError:
    print('NameError')
try:
    del x
except:
    print('NameError')

class C:

    def f():
        if False:
            while True:
                i = 10
        pass