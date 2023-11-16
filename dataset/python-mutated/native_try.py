@micropython.native
def f():
    if False:
        while True:
            i = 10
    try:
        fail
    finally:
        print('finally')
try:
    f()
except NameError:
    print('NameError')

@micropython.native
def f():
    if False:
        print('Hello World!')
    try:
        try:
            fail
        finally:
            print('finally')
    except NameError:
        print('NameError')
f()

@micropython.native
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