try:
    print('try 1')
    try:
        print('try 2')
        foo()
    except:
        print('except 2')
    bar()
except:
    print('except 1')
try:
    print('try 1')
    try:
        print('try 2')
        foo()
    except TypeError:
        print('except 2')
    bar()
except NameError:
    print('except 1')
try:
    try:
        raise Exception
    except (RuntimeError, SyntaxError):
        print('except 2')
except Exception:
    print('except 1')

def func1():
    if False:
        for i in range(10):
            print('nop')
    try:
        print('try func1')
        func2()
    except NameError:
        print('except func1')

def func2():
    if False:
        return 10
    try:
        print('try func2')
        foo()
    except TypeError:
        print('except func2')
func1()