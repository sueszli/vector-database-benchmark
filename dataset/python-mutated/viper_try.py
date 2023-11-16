@micropython.viper
def f():
    if False:
        for i in range(10):
            print('nop')
    try:
        fail
    finally:
        print('finally')
try:
    f()
except NameError:
    print('NameError')

@micropython.viper
def f():
    if False:
        return 10
    try:
        try:
            fail
        finally:
            print('finally')
    except NameError:
        print('NameError')
f()

@micropython.viper
def f():
    if False:
        while True:
            i = 10
    a = 100
    try:
        print(a)
        a = 200
        fail
    except NameError:
        print(a)
        a = 300
    print(a)
f()