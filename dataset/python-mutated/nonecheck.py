import cython

@cython.cclass
class MyClass:
    pass

@cython.nonecheck(False)
def func():
    if False:
        i = 10
        return i + 15
    obj: MyClass = None
    try:
        with cython.nonecheck(True):
            print(obj.myfunc())
    except AttributeError:
        pass
    print(obj.myfunc())